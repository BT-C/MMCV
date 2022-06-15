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
class EfficientSampleEpochBasedRunnerDebug(BaseRunner):
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
        self.every_layer_name = []
        self.iter_every_layer_grad = []
        for sub_module_tuple in self.model.module.named_children():
            self.register_every_layer_hook(sub_module_tuple)
        print('finish every layer hook')

    def register_every_layer_hook(self, children_module):
        def efficient_sample_backward_hook(layer, gin, gout):
            # print(type(layer), gout[0].abs().sum().item())
            # if type(layer) != torch.nn.modules.loss.MSELoss:
            # print(torch.cuda.current_device(), layer, gout[0].abs().sum().item())
            print(torch.cuda.current_device(), layer)
            for param in layer.parameters():
                print(param.shape)
                if param.grad is not None:
                    print(param.shape, param.grad.abs().sum().item())
            # self.every_layer_grad.append(gout[0].abs().sum().item())

        def efficient_sample_grad_hook(grad):
            print('-' * 10, grad.shape, grad.abs().sum().item())
            # self.iter_every_layer_grad.append(grad.abs().sum().item())

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

            self.every_layer_name.append(children_module[0])
            for param in children_module[1].parameters():
                param.register_hook(efficient_sample_grad_hook)
                
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
        self.every_layer_grad = []

        import numpy as np
        temp_record = []
        self.all_temp_layer_grad = []
        self.grad_result = [0 for _ in range(4)]
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        import random
        import os
        import copy
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
            if i == ((len(self.data_loader) // 2) - 1):
            # if i == 1:
                pass
            # if True:
            # if i == 4:
                # pass
                # from mmcv.runner import get_dist_info, init_dist, set_random_seed
                # import random
                # import os
                # pass
                # seed = 0
                # random.seed(seed)
                # np.random.seed(seed)
                # torch.manual_seed(seed)
                # torch.cuda.manual_seed(seed)
                # # torch.cuda.manual_seed_all(seed)
                # os.environ['PYTHONHASHSEED'] = str(seed)
                # # set_random_seed(seed)
            # if i == 10:
            #     break
            # ----------------------------------------------------------
            # self.every_layer_output = []
            
            self.every_layer_grad.append(np.array(copy.deepcopy(self.iter_every_layer_grad)))
            self.iter_every_layer_grad = []
            # self.every_layer_name = []


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
        #     # f"/home/chenbeitao/data/code/Test/txt/all_grad{torch.cuda.current_device()}.txt", 
        #     f"/home/chenbeitao/data/code/Test/txt/all_grad_backup{torch.cuda.current_device()}.txt", 
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


        print('every layer name :', len(self.every_layer_name), self.every_layer_name)
        print('every layer grad :', np.array(self.every_layer_grad).sum())
        print('all layer grad : ', len(self.all_layer_grad), len(self.all_layer_grad[0]))
        # print(torch.cuda.current_device(), len(self.all_temp_layer_grad), self.all_temp_layer_grad)
        # print(f'total sample : {len(self.data_loader)}, grad result : {self.grad_result}')
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
        self.every_layer_name = []
        self.iter_every_layer_grad = []
        for sub_module_tuple in self.model.module.named_children():
            self.register_every_layer_hook(sub_module_tuple)
        print('finish every layer hook')

    def register_every_layer_hook(self, children_module):
        def efficient_sample_backward_hook(layer, gin, gout):
            # print(type(layer), gout[0].abs().sum().item())
            # if type(layer) != torch.nn.modules.loss.MSELoss:
            # print(torch.cuda.current_device(), layer, gout[0].abs().sum().item())
            print(torch.cuda.current_device(), layer)
            for param in layer.parameters():
                print(param.shape)
                if param.grad is not None:
                    print(param.shape, param.grad.abs().sum().item())
            # self.every_layer_grad.append(gout[0].abs().sum().item())

        def efficient_sample_grad_hook(grad):
            # print('-' * 10, grad.shape, len(self.iter_every_layer_grad))
            self.iter_every_layer_grad.append(grad.abs().sum().item())

        if len(list(children_module[1].named_children())) == 0:
            # children_module[1].register_forward_hook(
            #     # lambda layer, input, output : print(output.shape, output.abs().sum())
            #     lambda layer, input, output : self.every_layer_output.append(output.abs().sum().item())
            # )

            # children_module[1].register_backward_hook(
            #     # lambda layer, gin, gout : print(gout[0].shape)
            #     # lambda layer, gin, gout : print(gout[0].abs().sum().item())
            #     # lambda layer, gin, gout : self.every_layer_grad.append(gout[0].abs().sum().item())
            #     # lambda layer, gin, gout : print(layer, type(layer), gout[0].abs().sum().item())
            #     # lambda layer, gin, gout : print(type(layer), layer.requires_grad)
            #     efficient_sample_backward_hook
            # )

            self.every_layer_name.append(children_module[0])
            for param in children_module[1].parameters():
                param.register_hook(efficient_sample_grad_hook)
                
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
        self.every_layer_grad = []
        self.all_img_ids = []

        import numpy as np
        temp_record = []
        self.all_temp_layer_grad = []
        self.grad_result = [0 for _ in range(4)]
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        import random
        import os
        import copy
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # self.register_every_layer_hook(self.model.module.named_children())
        # self.register_all_model()                     # multi-hook will lead to multi-hook-call
        # for i, data_batch in enumerate(self.data_loader):
        for i, data_batch in enumerate(self.train_data_loader[-1]):
            self._inner_iter = i
            # print('gpu_id', torch.cuda.current_device(), i, '/', len(self.train_data_loader[-1]), end='\r')
            # print(' '*100, end='\r')
            # print(self.model.device, ' : ' , data_batch['img'].abs().sum().item(), data_batch['img'].shape)

            image_meta = data_batch['img_metas'].data[0]
            batch_size = kwargs['cfg'].data.train_dataloader['samples_per_gpu']
            temp_img_ids = [image_meta[j]['img_id'] for j in range(len(image_meta))]
            if len(temp_img_ids) < batch_size:
                add_item = temp_img_ids[-1]
                temp_length = len(temp_img_ids)
                for _ in range(batch_size - len(temp_img_ids)):
                    temp_img_ids.append(add_item)
            assert len(temp_img_ids) == batch_size
            temp_img_ids = np.array(temp_img_ids)


            self.all_img_ids.append(temp_img_ids)

            # temp_name = []
            # temp_name.append(str(i))
            # for j in range(len(image_meta)):
            #     # print(image_meta[0][j]['image_file'])
            #     temp_name.append(image_meta[j]['image_file'].split('/')[-1])
            # self.image_meta = data_batch['img_metas'].data[0]
            
            # print(data_batch['img_metas'].data[0][0]['image_file'])
            # temp_record.append(np.array([
            #     i, len(self.data_loader), torch.cuda.current_device(),
            #     data_batch['img'].abs().sum().item()
            #     # *temp_name
            # ]))
            
            self.call_hook('before_train_iter')
            # ------------------------------------------------------------
            model_weight = 0
            # for name, parameters in self.model.module.named_parameters():
            #     if parameters is None:
            #         continue
            #     model_weight += parameters.abs().sum().item()
            # print('gpu_id : ', torch.cuda.current_device(), model_weight)
            # ------------------------------------------------------------

            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            # print(self.all_temp_layer_grad)
            # t = self.all_temp_layer_grad[-1].tolist()
            # t.append(data_batch['img'].abs().sum().item())
            # self.all_temp_layer_grad[-1] = np.array(t)

            self._iter += 1
            # print(torch.randn(2, 2))
            # ----------------------------------------------------------
            if i == ((len(self.data_loader) // 2) - 1):
            # if i == 1:
                pass
            # if True:
            # if i == 4:
                # pass
                # from mmcv.runner import get_dist_info, init_dist, set_random_seed
                # import random
                # import os
                # pass
                # seed = 0
                # random.seed(seed)
                # np.random.seed(seed)
                # torch.manual_seed(seed)
                # torch.cuda.manual_seed(seed)
                # # torch.cuda.manual_seed_all(seed)
                # os.environ['PYTHONHASHSEED'] = str(seed)
                # # set_random_seed(seed)
            # if i == 10:
            #     break
            # ----------------------------------------------------------
            # self.every_layer_output = []
            
            self.every_layer_grad.append(np.array(self.iter_every_layer_grad))
            self.iter_every_layer_grad = []
            # self.every_layer_name = []

        all_grad = 0
        for temp_grad in self.grad_result:
            all_grad += temp_grad
        
        # if self._epoch == 2:
        # self.pick_grad_dataset(kwargs)
        self.call_hook('after_train_epoch')
        self._epoch += 1
        block_epoch = 10
        if self._epoch > 0 and self._epoch % block_epoch == 0:
            self.train_data_loader = [self.train_data_loader[0]]
            self.cal_grad = False
        elif self._epoch % block_epoch == 1:
            if self.cal_grad is None or self.cal_grad == False:
                self.pick_grad_dataset(kwargs)
                self._epoch -= 1
                self.cal_grad = True
        print('Epoch : ', self._epoch, self.cal_grad)
        
                
    def pick_grad_dataset(self, kwargs):
        print('start to pick grad dataset')
        import os
        import numpy as np
        import torch.distributed as dist
        ROOT = 0
        every_layer_grad = torch.from_numpy(np.array(self.every_layer_grad)).cuda() # (N/batch_size/world_size, 907)
        all_img_ids = torch.from_numpy(np.array(self.all_img_ids, dtype=np.int32)).cuda()   # (N/batch_size/world_size, batch_size)
        assert all_img_ids.shape[0] == every_layer_grad.shape[0]

        self.every_layer_grad = []        
        self.all_img_ids = []
        this_rank = dist.get_rank()
        world_size = dist.get_world_size()
        # print(f'rank : {this_rank}, world_size : {world_size}')
        communication_tensor = torch.zeros((1)).cuda()
        grad_gather_list = [torch.zeros_like(every_layer_grad) for _ in range(world_size)]
        imgids_gather_list = [torch.zeros_like(all_img_ids) for _ in range(world_size)]
        broadcast_tensor = torch.tensor([this_rank], dtype=torch.int32).cuda()
        broadcast_list = [torch.zeros_like(broadcast_tensor) for _ in range(world_size)]
        print(f'{this_rank} send every layer grad')
        dist.all_gather(grad_gather_list, every_layer_grad, async_op=False)
        dist.all_gather(imgids_gather_list, all_img_ids, async_op=False)
        if this_rank == 0:
            print('all grad from other gpu :', len(grad_gather_list))
            print(grad_gather_list[0].shape)

            all_grad_gather = torch.cat(grad_gather_list, dim=0) # (N/batch_size, 878(HRNet)907(higherHRNet))
            all_imgids_gather = torch.cat(imgids_gather_list, dim=0)
            assert all_grad_gather.shape[0] == all_imgids_gather.shape[0]

            mean = all_grad_gather.mean(dim=0)
            # import matplotlib.pyplot as plt
            # x = [_ for _ in range(len(mean))]
            y = mean
            # plt.plot(x, y)
            # plt.savefig(f"/home/chenbeitao/data/code/Test/txt/out_result{torch.cuda.current_device()}.jpg")

            # var = all_grad_gather.var(dim=0)
            # norm_all_grad_gather = (all_grad_gather - mean) / (var + 1e-5)
            out_all_grad_gather = all_grad_gather.cpu().numpy()
            np.savetxt(
                f"/home/chenbeitao/data/code/Test/txt/epoch/out_result{torch.cuda.current_device()}_{str(self._epoch)}.txt", 
                out_all_grad_gather
            )
            # choose_index = ((all_grad_gather > mean).sum(dim=1) > all_grad_gather.shape[1]/2)
            # (((out1 > out1.mean(axis=0)*(1 + i/100)).sum(axis=1) > (out1>out1.mean(axis=0)).sumaxis=1).mean() * i/30).sum())
            choose_index = ((all_grad_gather > mean*(1+self._epoch/1000)).sum(dim=1) > (all_grad_gather > mean).sum(axis=1).float().mean() * self._epoch / 600)
            top_grad_img_index = np.argsort((out_all_grad_gather - out_all_grad_gather.mean(axis=0)).sum(axis=1))[::-1][:2].tolist()

            choose_img_ids = all_imgids_gather[choose_index].reshape(-1).cpu().numpy().tolist()
            choose_top_grad_img_index = all_imgids_gather[top_grad_img_index].reshape(-1).cpu().numpy().tolist()
            img_ids_set = set()
            img_ids_set.update(choose_img_ids)
            img_ids = list(img_ids_set)

            from pycocotools.coco import COCO
            import json
            coco = COCO(kwargs['cfg'].data.train['ann_file'])
            cat_ids = coco.getCatIds(catNms=['person'])
            img_list = coco.loadImgs(ids=img_ids)
            final_ann = {
                'images' : [],
                'annotations' : [],
                'categories' : json.load(open(kwargs['cfg'].data.train['ann_file']))['categories'],
            }
            for j, img in enumerate(img_list):
                ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids)
                ann_list = coco.loadAnns(ids=ann_ids)
                final_ann['images'].append(img)
                final_ann['annotations'].extend(ann_list)

                if img['id'] in choose_top_grad_img_index:
                    file_name = img['file_name']
                    source_img_file = os.path.join(
                        '/home/chenbeitao/data/code/mmlab/mmpose/data/coco/train2017',
                        file_name
                    )
                    target_dir = f'/home/chenbeitao/data/code/Test/txt/grad-image/epoch_{self._epoch}'
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    target_img_file = os.path.join(
                        target_dir,
                        file_name
                    )
                    
                    os.popen(f'cp {source_img_file} {target_img_file}')

            target_file_path = os.path.join(
                '/mnt/hdd2/chenbeitao/code/mmlab/mmpose/data/temp-coco/train', 'temp_keypoints_train.json'
            )
            with open(target_file_path, 'w') as fd:
                json.dump(final_ann, fd)
            
            print('choose number : ', choose_index.sum().item())

            print(all_grad_gather.shape)
            # for j in range(len(grad_gather_list)):
            #     print(grad_gather_list[j].shape, grad_gather_list[j][:5, :5])
            
            # print('-'*50, 'sleep 5')
            # dist.broadcast(communication_tensor, src=0, async_op=False)
            dist.all_gather(broadcast_list, broadcast_tensor, async_op=False)
            # for j in range(world_size):
            #     if j == 0:
            #         continue
            #     dist.send(broadcast_tensor, dst=j)
            print(f'{this_rank} broadcast!')
        else:
            print(this_rank, ' wait to recv ')
            # time.sleep(1)
            # dist.broadcast(communication_tensor, src=0, async_op=False)
            dist.all_gather(broadcast_list, broadcast_tensor, async_op=False)
            # dist.recv(broadcast_tensor, src=0)
            print(f'{this_rank} received broadcast and finish gradent statistic')
        
        print(f'GPU {this_rank} epoch :{self._epoch}', broadcast_list)
        # print(f'GPU {this_rank} epoch :{self._epoch}', broadcast_tensor)

        from mmpose.datasets import build_dataloader, build_dataset
        import copy
        new_train_cfg = copy.deepcopy(kwargs['cfg'].data.train)
        new_train_cfg['ann_file'] = 'data/temp-coco/train/temp_keypoints_train.json'
        train_loader_cfg = kwargs['train_loader_cfg']
        new_datasets = build_dataset(new_train_cfg)
        new_data_loaders = build_dataloader(new_datasets, **train_loader_cfg)
        self.train_data_loader.append(new_data_loaders)

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

        self.train_data_loader = data_loaders
        self.cal_grad = True
        self.register_all_model()
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
                    kwargs['cfg'] = cfg
                    kwargs['train_loader_cfg'] = train_loader_cfg
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
