# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        # ==================================================================================
        print(kwargs)
        try:
            soft_poes_matrix = kwargs['soft_pose_matrix']
            train_ann_file = kwargs['train_ann_file']

            def soft_pose_matrix(ann_file):
                from pycocotools.coco import COCO
                import numpy as np
                import torch.nn.functional as F

                coco = COCO(ann_file)
                cat_ids = coco.getCatIds(catNms=['person'])
                img_ids = coco.getImgIds(catIds=cat_ids)
                imgs = coco.loadImgs(ids=img_ids)

                num_img = 0
                num_person = 0
                vis_person = 0
                soft_mat = np.zeros((17, 17))
                mean_area = 0
                max_area = 0
                min_area = 100000000
                ratio_list = []
                print('Generate Soft pose matrix')
                for img in imgs:
                    ann_ids = coco.getAnnIds(imgIds=[img['id']], catIds=cat_ids)
                    anns = coco.loadAnns(ids=ann_ids)
                    if len(anns) == 0:
                        continue

                    num_img += 1
                    num_person += len(anns)

                    for ann in anns:
                        kpts = ann['keypoints']
                        area = ann['area']
                        kpts = np.array(kpts).reshape(17, 3)
                        vis_num = (kpts[:, 2] > 0).sum()
                        if vis_num == 17:
                            vis_person += 1
                        else:
                            continue

                        kpts = kpts[:, :2]
                        assert kpts.shape == (17, 2)
                        dist = np.sqrt(((kpts[None, :, :] - kpts[:, None, :])**2).sum(axis=-1))
                        assert dist.shape == (17, 17)

                        soft_mat += dist / area * 50

                soft_mat /= vis_person
                soft_mat += np.eye(*soft_mat.shape)
                adj_mat = torch.from_numpy(1 / soft_mat).float()
                adj_mat -= torch.eye(*adj_mat.shape)
                adj_mat = F.softmax(adj_mat, dim=1) 
                adj_mat = adj_mat - torch.diag_embed(torch.diag(adj_mat)) + torch.eye(*adj_mat.shape)
                

                return adj_mat

            soft_matrix = soft_pose_matrix(train_ann_file)
            kwargs['soft_pose_matrix'] = soft_matrix
        except:
            pass
        # ==================================================================================

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    kwargs['epoch'] = self.epoch
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead',
            DeprecationWarning)
        super().__init__(*args, **kwargs)




# =======================================================================================
@RUNNERS.register_module()
class EfficientSampleEpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    # -----------------------------------------------------------------------------------------
    # def efficent_sample_forward_hook()
    def register_all_model(self):
        self.every_layer_output = []
        self.every_layer_grad = []
        for sub_module_tuple in self.model.module.named_children():
            self.register_every_layer_hook(sub_module_tuple)
        print('finish every layer hook')

    def register_every_layer_hook(self, children_module):
        def efficient_sample_backward_hook(layer, gin, gout):
            # print(type(layer), gout[0].abs().sum().item())
            # if type(layer) != torch.nn.modules.loss.MSELoss:
            print(torch.cuda.current_device(), layer, gin[0].abs().sum().item())
            # self.every_layer_grad.append(gout[0].abs().sum().item())

        if len(list(children_module[1].named_children())) == 0:
            # children_module[1].register_forward_hook(
            #     # lambda layer, input, output : print(output.shape, output.abs().sum())
            #     lambda layer, input, output : self.every_layer_output.append(output.abs().sum().item())
            # )
            children_module[1].register_backward_hook(
                # lambda layer, gin, gout : print(gout[0].shape)
                # lambda layer, gin, gout : print(gout[0].abs().sum().item())
                # lambda layer, gin, gout : self.every_layer_grad.append(gout[0].abs().sum().item())
                # lambda layer, gin, gout : print(layer, type(layer), gout[0].abs().sum().item())
                # lambda layer, gin, gout : print(type(layer), layer.requires_grad)
                efficient_sample_backward_hook
            )

            return 

        for sub_module_tuple in children_module[1].named_children():
            self.register_every_layer_hook(sub_module_tuple)
        
    # -----------------------------------------------------------------------------------------

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.all_layer_grad = []

        import numpy as np
        temp_record = []
        self.all_temp_layer_grad = []
        self.grad_result = [0 for _ in range(4)]
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        import random
        import os
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # self.register_every_layer_hook(self.model.module.named_children())
        self.register_all_model()
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            # print(i, '/', len(self.data_loader))
            # print(len(self.data_loader))
            # print(self.model.device)
            # print(self.model.device, ' : ' , data_batch['img'].abs().sum().item(), data_batch['img'].shape)
            # print(type(data_batch))

            # print(type(data_batch['img_metas'].data))
            image_meta = data_batch['img_metas'].data
            # print(len(image_meta))
            # print(len(image_meta[0]))
            temp_name = []
            temp_name.append(str(i))
            for j in range(len(image_meta[0])):
                # print(image_meta[0][j]['image_file'])
                temp_name.append(image_meta[0][j]['image_file'].split('/')[-1])
            self.image_meta = data_batch['img_metas'].data[0]
            
            # print(data_batch['img_metas'].data[0][0]['image_file'])
            temp_record.append(np.array([
                i, len(self.data_loader), torch.cuda.current_device(),
                data_batch['img'].abs().sum().item()
                # *temp_name
            ]))
            
            self.call_hook('before_train_iter')
            # ------------------------------------------------------------
            model_weight = 0
            for name, parameters in self.model.module.named_parameters():
                if parameters is None:
                    continue
                model_weight += parameters.abs().sum().item()
            # print('gpu_id : ', torch.cuda.current_device(), model_weight)
            # ------------------------------------------------------------

            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            # print(self.all_temp_layer_grad)
            t = self.all_temp_layer_grad[-1].tolist()
            t.append(data_batch['img'].abs().sum().item())
            self.all_temp_layer_grad[-1] = np.array(t)

            self._iter += 1
            # print(torch.randn(2, 2))
            # ----------------------------------------------------------
            # if i == ((len(self.data_loader) // 2) - 1):
            # if i == 1:
                # pass
            if True:
            # if i == 4:
                # pass
                # from mmcv.runner import get_dist_info, init_dist, set_random_seed
                # import random
                # import os
                # pass
                seed = 0
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                # torch.cuda.manual_seed_all(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                # # set_random_seed(seed)
            # if i == 10:
            #     break
            # ----------------------------------------------------------

        all_grad = 0
        for temp_grad in self.grad_result:
            all_grad += temp_grad

        # all_layers_grad = np.array(self.all_layer_grad)
        # np.savetxt(
        #     f"/home/chenbeitao/data/code/Test/txt/result{torch.cuda.current_device()}.txt", 
        #     np.array(all_layers_grad)
        # )
        
        print(np.array(temp_record))
        # np.savetxt(
        #     f"/home/chenbeitao/data/code/Test/txt/filename{torch.cuda.current_device()}.txt", 
        #     np.array(temp_record),
        #     '%s'
        # )
        # np.savetxt(
        #     f"/home/chenbeitao/data/code/Test/txt/seed_batch{torch.cuda.current_device()}.txt", 
        #     np.array(temp_record),
        # )

        # np.savetxt(
        #     f"/home/chenbeitao/data/code/Test/txt/all_grad{torch.cuda.current_device()}.txt", 
        #     # f"/home/chenbeitao/data/code/Test/txt/all_grad_backup{torch.cuda.current_device()}.txt", 
        #     # f"/home/chenbeitao/data/code/Test/txt/all_grad_backup.txt", 
        #     # f"/home/chenbeitao/data/code/Test/txt/all_grad_backup_DISABLE_NCCL.txt", 
        #     np.array(self.all_temp_layer_grad)
        # )
        
        # np.savetxt(
        #     # f"/home/chenbeitao/data/code/Test/txt/all_grad_higher_backup{torch.cuda.current_device()}.txt", 
        #     f"/home/chenbeitao/data/code/Test/txt/all_grad_higher{torch.cuda.current_device()}.txt", 
        #     np.array(self.all_temp_layer_grad)
        # )

        # np.savetxt(
        #     # f"/home/chenbeitao/data/code/Test/txt/every_layer_output{torch.cuda.current_device()}.txt", 
        #     # f"/home/chenbeitao/data/code/Test/txt/every_layer_output_backup{torch.cuda.current_device()}.txt", 
        #     f"/home/chenbeitao/data/code/Test/txt/every_layer_grad{torch.cuda.current_device()}.txt", 
        #     # f"/home/chenbeitao/data/code/Test/txt/every_layer_grad_backup{torch.cuda.current_device()}.txt", 
        #     np.array(self.every_layer_grad)
        # )
        print('every layer grad :', np.array(self.every_layer_grad).sum())
        print(torch.cuda.current_device(), self.all_temp_layer_grad)
        print(f'total sample : {len(self.data_loader)}, grad result : {self.grad_result}')
        print(f'all grad : {all_grad}')

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):            
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, 
            train_loader_cfg=None, cfg=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    kwargs['epoch'] = self.epoch
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
