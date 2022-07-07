# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from matplotlib.pyplot import flag

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
class SLSOptimizerEpochBasedRunner(BaseRunner):
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

            from mmpose.core.optimizer.SLS_optimizer import SLSOptimizer
            if type(self.optimizer) == SLSOptimizer:
                temp_inputs, temp_kwargs = self.model.scatter(data_batch, kwargs, self.model.device_ids)
            
                # output = self.module.train_step(*inputs[0], **kwargs[0])
                # self.optimizer.loss_function = lambda : self.model.train_step(data_batch, self.optimizer, **kwargs)
                self.optimizer.loss_function = lambda : self.model.module.train_step(*temp_inputs[0], **temp_kwargs[0])
            
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
        # self.iter_every_layer_grad = []
        self.iter_every_layer_grad = torch.zeros((0)).cuda()
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
            # print('-' * 10, grad.shape, grad.abs().sum().item())

            # self.iter_every_layer_grad.append(grad.abs().sum().item())
            self.iter_every_layer_grad = torch.cat([self.iter_every_layer_grad, grad.reshape(-1)])

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

        # import random
        # import os
        # import copy
        # seed = 0
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)

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
            
            self.every_layer_grad.append(self.iter_every_layer_grad.unsqueeze(0))
            self.iter_every_layer_grad = torch.zeros((0)).cuda()
            # self.every_layer_name = []

        all_grad = 0
        for temp_grad in self.grad_result:
            all_grad += temp_grad
        
        # if self._epoch == 2:
        self.pick_grad_dataset(kwargs)
        self.call_hook('after_train_epoch')
        
        block_epoch = 20
        if self._epoch > 0 and self._epoch % block_epoch == 0:
            self.train_data_loader = [self.train_data_loader[0]]
            self.cal_grad = False
        elif self._epoch % block_epoch == 1:
            if self.cal_grad is None or self.cal_grad == False:
                self.pick_grad_dataset(kwargs)
                self._epoch -= 1
                self.cal_grad = True
        print('Epoch : ', self._epoch, self.cal_grad)
        self._epoch += 1
        
    def get_four_stages(self, all_grad_gather):
        #  'stage1' : 51, 'stage3': 108, 'stage4': 483
        ''' all_grad_gather : (N, 907) '''
        # temp_grad = all_grad_gather[:, ::-1]
        temp_grad = torch.flip(all_grad_gather, [1])
        return torch.cat(
            [
                temp_grad[:, :51].sum(axis=1, keepdim=True), 
                temp_grad[:, 51:108].sum(axis=1, keepdim=True),
                temp_grad[:, 108:483].sum(axis=1, keepdim=True),
                temp_grad[:, 483:].sum(axis=1, keepdim=True)
            ], dim=1
        )
                
    def pick_grad_dataset(self, kwargs):
        print('start to pick grad dataset')
        import os
        import numpy as np
        import torch.distributed as dist
        ROOT = 0
        # every_layer_grad = torch.from_numpy(np.array(self.every_layer_grad)).cuda() # (N/batch_size/world_size, 907)
        every_layer_grad = torch.cat(self.every_layer_grad, dim=0)
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
            # all_grad_gather = self.get_four_stages(all_grad_gather)
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
            # np.savetxt(
            #     f"/home/chenbeitao/data/code/Test/txt/orientation/grad_{torch.cuda.current_device()}_{str(self._epoch)}.txt", 
            #     # f"/home/chenbeitao/data/code/Test/txt/higher/stage4_pretrained_grad_{torch.cuda.current_device()}_{str(self._epoch)}.txt", 
            #     out_all_grad_gather
            # )
            
            # choose_index = ((all_grad_gather > mean).sum(dim=1) > all_grad_gather.shape[1]/2)
            # (((out1 > out1.mean(axis=0)*(1 + i/100)).sum(axis=1) > (out1>out1.mean(axis=0)).sumaxis=1).mean() * i/30).sum())
            # choose_index = ((all_grad_gather > mean*(1+self._epoch/1000)).sum(dim=1) > (all_grad_gather > mean).sum(axis=1).float().mean() * self._epoch / 600)
            # choose_index = ((all_grad_gather > mean).sum(dim=1) > (all_grad_gather > mean).sum(axis=1).float().mean())
            pos_flag = ((all_grad_gather > mean).sum(axis=1) == 4)
            neg_flag = ~pos_flag
            pos_flag[torch.where(neg_flag == True)[0][:pos_flag.sum()]] = True
            # choose_index = ((all_grad_gather > mean).sum(axis=1) == 4)
            choose_index = pos_flag
            
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
                    
                    # os.popen(f'cp {source_img_file} {target_img_file}')

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



def read_grad_oriental_file_single(q, block_id, file_list, epoch, gpu_id):
    import os
    import torch
    
    
    # print('read file')
    # ids = id.get()

    print(f'gpu id {gpu_id} start to read file')
    x_id, y_id = block_id[0], block_id[1]
    x_file_name = file_list[x_id]
    y_file_name = file_list[y_id]
    x_grad, y_grad = None, None
    x_grad = torch.load(
        os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{epoch}', x_file_name)
    )
    y_grad = x_grad
    # y_grad = torch.load(
    #     os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{epoch}', y_file_name)
    # )
            
    print(f'gpu {gpu_id} put tensor')
    q.put([block_id, [x_grad, y_grad]])

    # q.join()

def grad_son_read_file(file_name, in_pipe):
    y_grad = torch.load(file_name)
    in_pipe.send(y_grad)

def read_grad_oriental_multi_process(q, block_id, file_list, epoch, gpu_id):
        import os
        import torch
        from torch.multiprocessing import Process, Pipe        
        
        # print('read file')
        # ids = id.get()
        out_pipe, in_pipe = Pipe(True)
        print(f'gpu id {gpu_id} start to read file')
        x_id, y_id = block_id[0], block_id[1]
        x_file_name = file_list[x_id]
        y_file_name = file_list[y_id]
        x_grad, y_grad = None, None

        son_process = Process(
            target=grad_son_read_file, 
            args=(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{epoch}', y_file_name),
                in_pipe,
            )
        )
        son_process.start()

        x_grad = torch.load(
            os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{epoch}', x_file_name)
        )
        # y_grad = x_grad
        # y_grad = torch.load(
        #     os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{epoch}', y_file_name)
        # )
        while True:
            try: 
                y_grad = out_pipe.recv()
                print('grad_son_process recv: ', y_grad.shape)
            except:
                break
        son_process.join()
                
        print(f'gpu {gpu_id} put tensor')
        q.put([block_id, [x_grad, y_grad]])
    # q.join()


def read_grad_oriental_file_batch(q, block_id_list, file_list, epoch, gpu_id, process_id=0):
    import os
    import torch
        
    
    # ids = id.get()
    for i in range(block_id_list.shape[0]):
        block_id = block_id_list[i]
        x_id, y_id = block_id[0], block_id[1]
        x_file_name = file_list[x_id]
        y_file_name = file_list[y_id]
        # x_grad, y_grad = None, None
        print(x_id, y_id)
        print(f'process_id {process_id} start to read file')
        x_grad = torch.load(
            os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{epoch}', x_file_name)
        )
        y_grad = torch.load(
            os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{epoch}', y_file_name)
        )
                
        print(f'gpu {gpu_id} process_id {process_id} put tensor')
        q.put([block_id, [x_grad, y_grad]])

        # q.join()

def read_grad_oriental_file_one_by_one(q, block_id, file_list, epoch, gpu_id):
    import os
    import torch
    
    print(f'gpu id {gpu_id} start to read file')
    flag, x_id = block_id[0], block_id[1]
    x_file_name = file_list[x_id]
    x_grad = None
    x_grad = torch.load(
        os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{epoch}', x_file_name)
    )

    print(x_id)        
    print(f'gpu {gpu_id} put tensor')
    q.put([block_id, x_grad])
            


import os
from torch.utils.data import DataLoader, Dataset
import multiprocessing
import multiprocessing.pool

class GradDataset(Dataset):
    def __init__(self, file_list, block_ids, current_epoch, gpu_id):
        self.block_ids = block_ids
        self.file_list = file_list
        self.epoch = current_epoch
        self.gpu_id = gpu_id

    def __len__(self):
        return len(self.block_ids)

    def __getitem__(self, idx):
        import time
        x_id = self.block_ids[idx]
        x_file_name = self.file_list[x_id]

        print(f'gpu_id {self.gpu_id} start to read')
        t1 = time.time()
        x_grad = torch.load(
            os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self.epoch}', x_file_name)
        )
        print('put grad tensor from ', x_file_name, 'time :', time.time() - t1)
        # print(x_grad.shape)

        return [x_id, x_grad]

@RUNNERS.register_module()
class EfficientSampleGradOrientationEpochBasedRunner(BaseRunner):
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
        self.sum_grad = None
        # self.iter_every_layer_grad = torch.zeros((0)).cuda()
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
            # print('-' * 10, grad.shape, grad.abs().sum().item())

            # self.iter_every_layer_grad.append(grad.abs().sum().item())
            # pass
            # self.iter_every_layer_grad = torch.cat([self.iter_every_layer_grad, grad.reshape(-1)])
            # print(grad.shape, grad.reshape(-1).shape)
            self.iter_every_layer_grad.append(grad.reshape(-1))


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

        # for i, data_batch in enumerate(self.data_loader):
        for i, data_batch in enumerate(self.train_data_loader[-1]):
            self._inner_iter = i

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
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            # t = self.all_temp_layer_grad[-1].tolist()
            # t.append(data_batch['img'].abs().sum().item())
            # self.all_temp_layer_grad[-1] = np.array(t)

            self._iter += 1
            # ----------------------------------------------------------
            # self.every_layer_grad.append(self.iter_every_layer_grad.unsqueeze(0))
            # self.iter_every_layer_grad = torch.zeros((0)).cuda()
            # self.every_layer_grad.append(torch.tensor(self.iter_every_layer_grad))
            # self.every_layer_grad.append(torch.cat(self.iter_every_layer_grad, dim=0))

            added_tensor = torch.cat(self.iter_every_layer_grad, dim=0)
            if self.sum_grad is None:
                self.sum_grad = torch.zeros_like(added_tensor)
                self.l1_arr = []
            self.sum_grad += added_tensor.abs()
            if i % 10 == 0:
                print(self.sum_grad.abs().sum())

            self.iter_every_layer_grad = []
            if len(self.every_layer_grad) == 30:
                # self.save_grad_vector()        
                pass
                
        
        if len(self.every_layer_grad) > 0:
            # self.save_grad_vector()
            pass
        assert 0 == 1

        all_grad = 0
        for temp_grad in self.grad_result:
            all_grad += temp_grad
        
        # if self._epoch == 2:
        # self.get_grad_dist()
        # self.pick_grad_dataset(kwargs)
        self.call_hook('after_train_epoch')
        
        block_epoch = 20
        if self._epoch > 0 and self._epoch % block_epoch == 0:
            self.train_data_loader = [self.train_data_loader[0]]
            self.cal_grad = False
        elif self._epoch % block_epoch == 1:
            if self.cal_grad is None or self.cal_grad == False:
                self.pick_grad_dataset(kwargs)
                self._epoch -= 1
                self.cal_grad = True
        print('Epoch : ', self._epoch, self.cal_grad)
        self._epoch += 1

    def save_grad_vector(self):
        import os
        import torch.nn.functional as F
        import h5py

        # target_dir = f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}'
        for i in range(3):
            target_dir = f'/mnt/hdd{i}/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}'
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            # file_name = f'iter_{self._inner_iter}_gpu_{torch.cuda.current_device()}.pt'
            file_name = f'grad_tensor.hdf5'
            save_grad_ori = F.normalize(torch.cat(self.every_layer_grad, dim=0), p=2, dim=1)
            f = h5py.File(os.path.join(target_dir, file_name), 'a')
            # if self.max_val is None:
            #     self.max_val = 0
            # if self.min_val is None:
            #     self.min_val = 1000
            # self.max_val = max(save_grad_ori.abs().max(), self.max_val)
            # self.min_val = min(save_grad_ori.abs().min(), self.min_val)

            save_grad_ori = save_grad_ori.cpu()

            torch.save(save_grad_ori, os.path.join(target_dir, file_name))
            self.every_layer_grad = []
            torch.cuda.empty_cache()
        
    def get_four_stages(self, all_grad_gather):
        #  'stage1' : 51, 'stage3': 108, 'stage4': 483
        ''' all_grad_gather : (N, 907) '''
        # temp_grad = all_grad_gather[:, ::-1]
        temp_grad = torch.flip(all_grad_gather, [1])
        return torch.cat(
            [
                temp_grad[:, :51].sum(axis=1, keepdim=True), 
                temp_grad[:, 51:108].sum(axis=1, keepdim=True),
                temp_grad[:, 108:483].sum(axis=1, keepdim=True),
                temp_grad[:, 483:].sum(axis=1, keepdim=True)
            ], dim=1
        )
    
    def get_block_list(self, file_num):
        block_id = torch.ones((file_num, file_num))
        block_id = torch.triu(block_id)
        x_id, y_id = torch.where(block_id == 1)
        block_id = torch.cat([x_id.unsqueeze(1), y_id.unsqueeze(1)], dim=1)

        return block_id

    def get_all_grad_file(self, file_list, world_size):
        for i in range(len(file_list)):
            file_list[i] = f"iter_{file_list[i].split('_')[1].zfill(6)}_gpu_{file_list[i].split('_')[-1].split('.')[0]}.pt"
        file_list.sort()
        out_file = []
        for i in range(world_size):
            out_file.extend(file_list[i::world_size])
        assert len(out_file) == len(file_list)
        for i in range(len(out_file)):
            out_file[i] = f"iter_{int(out_file[i].split('_')[1])}_gpu_{out_file[i].split('_')[-1].split('.')[0]}.pt"

        return out_file
    
    def get_self_block_ids_and_file_list(self):
        import os
        import math
        import torch.distributed as dist

        gpu_id = torch.cuda.current_device()
        dirname, _, file_name = next(os.walk(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}'))
        file_num = len(file_name)
        block_ids = self.get_block_list(file_num) # (K, 2)
        file_list = self.get_all_grad_file(file_name, dist.get_world_size())
        block_length = math.ceil(block_ids.shape[0] / dist.get_world_size())
        self_block_ids = block_ids[gpu_id * block_length : (gpu_id + 1) * block_length]

        return self_block_ids, file_list, block_length

    
    def calculate_grad_dist(self, q, block_size, gpu_id, in_pipe):
        import torch.nn.functional as F
        import torch
        import time

        count = 0
        time.sleep(3)
        print('start get tensor')
        while True:
            res = q.get()
            if res is None:
                break

            print(f'gpu:{gpu_id} {count} / {block_size}')
            x_id, y_id = res[0][0], res[0][1]
            in_pipe.send(res)
            # x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
            # x_grad = F.normalize(x_grad, p=2, dim=1)
            # y_grad = F.normalize(y_grad, p=2, dim=1)
            # print(x_grad.device, y_grad.device)
            # for i in range(x_grad.shape[0]):
            #     temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
            #     self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
                
            count += 1
            if count == block_size:
                q.task_done()
                return 

    @staticmethod
    def err_call_back(err):
        print(f'catch error：{str(err)}')

    def get_grad_dist_dataloader(self):
        import torch.nn.functional as F

        gpu_id = torch.cuda.current_device()
        block_ids, file_list = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()

        grad_file_dataset = GradDataset(file_list, block_ids, self._epoch)
        grad_file_dataloader = DataLoader(
            grad_file_dataset, 
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        for i, data in enumerate(grad_file_dataloader):
            print(i, 'gpu_id :', gpu_id)
            x_id, y_id = data[0][0], data[0][1]
            x_grad, y_grad = data[1][0].squeeze(), data[1][1].squeeze()
            print(data[0])
            print(x_grad.shape)
            x_grad = x_grad.cuda()
            y_grad = y_grad.cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)
            y_grad = F.normalize(y_grad, p=2, dim=1)
            for i in range(x_grad.shape[0]):
                temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
                self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
            print('over cal')



        assert 0 == 1

    
    def get_grad_dist_test(self):
        # from multiprocessing import Process, Pool, Pipe
        import math
        import torch.nn.functional as F
        from torch.multiprocessing import Process, Pool, Pipe
        from threading import Thread
        
        # import multiprocessing
        import torch
        # torch.multiprocessing.set_start_method('spawn')
        # print(torch.multiprocessing.get_start_method())


        gpu_id = torch.cuda.current_device()
        block_ids, file_list = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()
        out_pipe, in_pipe = Pipe(True)

        pool = Pool(5)
        q = multiprocessing.Manager().Queue()
        # tensor_p = Process(target=self.calculate_grad_dist, args=(q, block_ids.shape[0], gpu_id, in_pipe))

        # id_list = [[i, i + 1] for i in range(60)]
        result_list = []
        for i in range(block_ids.shape[0]):
            ret = pool.apply_async(
                func=read_grad_oriental_file_single, 
                args=(q, block_ids[i], file_list, self._epoch, gpu_id), 
                error_callback=self.err_call_back
            )
            result_list.append(ret)
        # process_num = 5
        # task_list = []
        # for i in range(process_num):
        #     single_size = math.ceil(block_ids.shape[0] / process_num)
        #     p_task = Process(
        #         target=read_grad_oriental_file_batch, 
        #         args=(q, block_ids[i * single_size : (i + 1) * single_size], file_list, self._epoch, gpu_id, i)
        #     )
        #     task_list.append(p_task)
        #     p_task.start()
        
        # tensor_p.start()
        # self.calculate_grad_dist(q, block_ids.shape[0])

        count = 0
        block_size = block_ids.shape[0]
        print('GPU id :', torch.cuda.current_device())
        while True:
            res = q.get()
            if res is None:
                break

            print(f'gpu:{gpu_id} {count} / {block_size}')
            x_id, y_id = res[0][0], res[0][1]
            x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
            print(x_grad.shape, y_grad.shape)
            x_grad = F.normalize(x_grad, p=2, dim=1)
            y_grad = F.normalize(y_grad, p=2, dim=1)
            for i in range(x_grad.shape[0]):
                temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
                self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
            
            count += 1
            if count == block_size:
                break
    
        # count = 1
        # while True:
        #     try:
        #         res = out_pipe.recv()
        #         x_id, y_id = res[0][0], res[0][1]
        #         print(res[0])
        #         x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
        #         x_grad = F.normalize(x_grad, p=2, dim=1)
        #         y_grad = F.normalize(y_grad, p=2, dim=1)
        #         for i in range(x_grad.shape[0]):
        #             temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
        #             self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
        #         count += 1
        #         if count == block_ids.shape[0]:
        #             break
        #     except:
        #         break
        
        pool.close()
        pool.join()
        # for task in task_list:
        #     task.join()
        # q.join()
        # tensor_p.join()
        
        print('over')
        assert 0 == 1
   
    def get_grad_dist_one_by_one(self):
        # from multiprocessing import Process, Pool, Pipe
        import math
        import torch.nn.functional as F
        from torch.multiprocessing import Process, Pool, Pipe
        from threading import Thread
        
        # import multiprocessing
        import torch
        # torch.multiprocessing.set_start_method('spawn')
        # print(torch.multiprocessing.get_start_method())

        gpu_id = torch.cuda.current_device()
        block_ids, file_list = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()
        out_pipe, in_pipe = Pipe(True)

        pool = Pool(5)
        q = multiprocessing.Manager().Queue()
        # tensor_p = Process(target=self.calculate_grad_dist, args=(q, block_ids.shape[0], gpu_id, in_pipe))

        # id_list = [[i, i + 1] for i in range(60)]
        flag_block_ids =  torch.cat(
            [torch.tensor([i, block_ids[i][0], i, block_ids[i][1]]) for i in range(block_ids.shape[0])]
        ).reshape(-1, 2)
        result_list = []
        for i in range(flag_block_ids.shape[0]):
        # for i in range(10):
            ret = pool.apply_async(
                func=read_grad_oriental_file_one_by_one, 
                args=(q, flag_block_ids[i], file_list, self._epoch, gpu_id), 
                error_callback=self.err_call_back
            )
            result_list.append(ret)
            # ret = pool.apply_async(
            #     func=read_grad_oriental_file_one_by_one, 
            #     args=(q, [i, block_ids[i][1]], file_list, self._epoch, gpu_id), 
            #     error_callback=self.err_call_back
            # )
            # result_list.append(ret)

        # process_num = 5
        # task_list = []
        # for i in range(process_num):
        #     single_size = math.ceil(block_ids.shape[0] / process_num)
        #     p_task = Process(
        #         target=read_grad_oriental_file_batch, 
        #         args=(q, block_ids[i * single_size : (i + 1) * single_size], file_list, self._epoch, gpu_id, i)
        #     )
        #     task_list.append(p_task)
        #     p_task.start()
        
        # tensor_p.start()
        # self.calculate_grad_dist(q, block_ids.shape[0])

        count = 0
        block_size = block_ids.shape[0]
        buffer_block = [[-1, -1] for _ in range(block_size)]
        print('GPU id :', torch.cuda.current_device())
        while True:
            res = q.get()
            if res is None:
                break

            # print(f'gpu:{gpu_id} {count} / {block_size}')
            # x_id, y_id = res[0][0], res[0][1]
            # x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
            # print(x_grad.shape, y_grad.shape)
            flag_id, temp_block_id = res[0]
            temp_grad = res[1]
            if buffer_block[flag_id][0] != -1:
                print(f'gpu id {gpu_id} find pair')
                x_id, y_id = buffer_block[flag_id][0], temp_block_id
                x_grad, y_grad = buffer_block[flag_id][1], temp_grad
                x_grad, y_grad = x_grad.cuda(), y_grad.cuda()

                x_grad = F.normalize(x_grad, p=2, dim=1)
                y_grad = F.normalize(y_grad, p=2, dim=1)
                for i in range(x_grad.shape[0]):
                    temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
                    self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
                temp_grad = buffer_block[flag_id][1]
                del temp_grad
                buffer_block[flag_id][0] = buffer_block[flag_id][1] = -1
                print(f'gpu:{gpu_id} {count} / {block_size}')
                count += 1
            else:
                buffer_block[flag_id][0] = temp_block_id
                buffer_block[flag_id][1] = temp_grad
            
            if count == block_size:
                break
    
        # count = 1
        # while True:
        #     try:
        #         res = out_pipe.recv()
        #         x_id, y_id = res[0][0], res[0][1]
        #         print(res[0])
        #         x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
        #         x_grad = F.normalize(x_grad, p=2, dim=1)
        #         y_grad = F.normalize(y_grad, p=2, dim=1)
        #         for i in range(x_grad.shape[0]):
        #             temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
        #             self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
        #         count += 1
        #         if count == block_ids.shape[0]:
        #             break
        #     except:
        #         break
        
        pool.close()
        pool.join()
        # for task in task_list:
        #     task.join()
        # q.join()
        # tensor_p.join()
        
        print('over')
        assert 0 == 1

    def get_grad_dist(self):
        import os
        import torch.nn.functional as F

        gpu_id = torch.cuda.current_device()
        block_ids, file_list, block_length = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        block_num = len(file_list)
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()

        count = 0
        y_flag = True
        for i in range(block_ids[0][0], block_num):
            print(f'gpu:{gpu_id}-{i}')
            x_file_name = file_list[i]
            x_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', x_file_name)
            ).cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)

            init_j = block_ids[0][1] if y_flag else i + 1
            y_flag = False
            for j in range(init_j, block_num):
                count += 1

                y_file_name = file_list[j]
                y_grad = torch.load(
                    os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', y_file_name)
                ).cuda()
                y_grad = F.normalize(y_grad, p=2, dim=1)    

                print(f'gpu:{gpu_id} {count} / {len(block_ids)}')

                for t in range(x_grad.shape[0]):
                    temp_dist = ((x_grad[t] - y_grad) ** 2).sum(dim=1)
                    self.all_grad_dist[i*30 + t, j*30 : j*30 + y_grad.shape[0]] = temp_dist

                if count == block_length:
                    return 

    def get_grad_dist_loop_multiprocess(self):
        import os
        import torch.nn.functional as F
        from torch.multiprocessing import Process, Pool, Pipe
        import torch

        gpu_id = torch.cuda.current_device()
        block_ids, file_list, block_length = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        block_num = len(file_list)
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()

        count = 0
        y_flag = True
        for i in range(block_ids[0][0], block_num):
            print(f'gpu:{gpu_id}-{i}')
            x_file_name = file_list[i]
            x_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', x_file_name)
            ).cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)

            init_j = block_ids[0][1] if y_flag else i + 1
            y_flag = False
            temp_pool = Pool(10)
            q = multiprocessing.Manager().Queue()

            result_list = []
            for j in range(init_j, block_num):
                ret = temp_pool.apply_async(
                    func=read_grad_oriental_file_one_by_one,
                    args=(q, [0, j], file_list, self._epoch, gpu_id,),
                    error_callback=self.err_call_back
                )
                result_list.append(ret)

            
            j_num = 0
            while True:
                res = q.get()
                if res == None:
                    break
                count += 1
                j_num += 1

                print(f'gpu:{gpu_id} {count} / {len(block_ids)}')
                y_id = res[0][1]
                y_grad = res[1]
                y_grad = y_grad.cuda()
                y_grad = F.normalize(y_grad, p=2, dim=1)    

                for t in range(x_grad.shape[0]):
                    temp_dist = ((x_grad[t] - y_grad) ** 2).sum(dim=1)
                    self.all_grad_dist[i*30 + t, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist

                if j_num == block_num - init_j:
                    break
                if count == block_length:
                    return 

            temp_pool.close()
            temp_pool.join()

    def get_grad_dist_mul_dataloader(self):
        import os
        import torch.nn.functional as F
        from torch.multiprocessing import Process, Pool, Pipe
        import torch

        gpu_id = torch.cuda.current_device()
        block_ids, file_list, block_length = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        block_num = len(file_list)
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()

        count = 0
        y_flag = True
        for i in range(block_ids[0][0], block_num):
            
            x_file_name = file_list[i]
            x_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', x_file_name)
            ).cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)
            print(f'gpu:{gpu_id}-{i}')

            init_j = block_ids[0][1] if y_flag else i + 1
            grad_dataset = GradDataset(file_list, [t for t in range(init_j, block_num)], self._epoch, gpu_id)
            grad_dataloader = DataLoader(
                dataset=grad_dataset,
                pin_memory=True,
                batch_size=1,
                num_workers=2,
                prefetch_factor=1
            )

            for _, data in enumerate(grad_dataloader):
                y_id, y_grad = data

                count += y_grad.shape[0]
                print(f'gpu:{gpu_id} {count} / {len(block_ids)}')
                y_grad = y_grad.squeeze().cuda()
                y_grad = F.normalize(y_grad, p=2, dim=1)    

                for t in range(x_grad.shape[0]):
                    temp_dist = ((x_grad[t] - y_grad) ** 2).sum(dim=1)
                    self.all_grad_dist[i*30 + t, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist

                if count == block_length:
                    return 

    def test_get_grad_dist(self):
        import os
        import torch.nn.functional as F

        gpu_id = torch.cuda.current_device()
        block_ids, file_list = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        self.all_grad_dist = torch.zeros((grad_size, grad_size))
        for k, (x_id, y_id) in enumerate(block_ids):
            print(f'gpu:{gpu_id} {k} / {len(block_ids)}')
            # x_file_name = f'iter_{x_id * 30 + 29}_gpu_{gpu_id}.pt'
            # y_file_name = f'iter_{y_id * 30 + 29}_gpu_{}.pt'
            x_file_name = file_list[x_id]
            y_file_name = file_list[y_id]
            x_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', x_file_name)
            ).cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)
            y_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', y_file_name)
            ).cuda()
            y_grad = F.normalize(y_grad, p=2, dim=1)

            for i in range(x_grad.shape[0]):
                temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
                self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
            # break


    def pick_grad_dataset(self, kwargs):
        print('start to pick grad dataset')
        import os
        import numpy as np
        import torch.distributed as dist
        ROOT = 0
        # every_layer_grad = torch.from_numpy(np.array(self.every_layer_grad)).cuda() # (N/batch_size/world_size, 907)

        self.get_grad_dist()
        self.all_grad_dist = self.all_grad_dist.cuda()
        all_img_ids = torch.from_numpy(np.array(self.all_img_ids, dtype=np.int32)).cuda()   # (N/batch_size/world_size, batch_size)
        
        self.every_layer_grad = []        
        self.all_img_ids = []
        this_rank = dist.get_rank()
        world_size = dist.get_world_size()
        # print(f'rank : {this_rank}, world_size : {world_size}')
        
        grad_gather_list = [torch.zeros_like(self.all_grad_dist) for _ in range(world_size)]
        imgids_gather_list = [torch.zeros_like(all_img_ids) for _ in range(world_size)]
        broadcast_tensor = torch.tensor([this_rank], dtype=torch.int32).cuda()
        broadcast_list = [torch.zeros_like(broadcast_tensor) for _ in range(world_size)]
        print(f'{this_rank} send every layer grad')
        dist.all_gather(grad_gather_list, self.all_grad_dist.cuda(), async_op=False)
        dist.all_gather(imgids_gather_list, all_img_ids, async_op=False)
        if this_rank == 0:
            print('all grad from other gpu :', len(grad_gather_list))
            print(grad_gather_list[0].shape)

            # all_grad_gather = torch.cat(grad_gather_list, dim=0) # (N/batch_size, 878(HRNet)907(higherHRNet))
            all_grad_gather = torch.zeros_like(grad_gather_list[0])
            for temp_grad_dist in grad_gather_list:
                all_grad_gather += temp_grad_dist
            torch.save(all_grad_gather, f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}/grad-dist/grad_dist.pt')
            assert 0 == 1

            all_imgids_gather = torch.cat(imgids_gather_list, dim=0)
            # all_grad_gather = self.get_four_stages(all_grad_gather)
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
            # np.savetxt(
            #     f"/home/chenbeitao/data/code/Test/txt/orientation/grad_{torch.cuda.current_device()}_{str(self._epoch)}.txt", 
            #     # f"/home/chenbeitao/data/code/Test/txt/higher/stage4_pretrained_grad_{torch.cuda.current_device()}_{str(self._epoch)}.txt", 
            #     out_all_grad_gather
            # )
            
            # choose_index = ((all_grad_gather > mean).sum(dim=1) > all_grad_gather.shape[1]/2)
            # (((out1 > out1.mean(axis=0)*(1 + i/100)).sum(axis=1) > (out1>out1.mean(axis=0)).sumaxis=1).mean() * i/30).sum())
            # choose_index = ((all_grad_gather > mean*(1+self._epoch/1000)).sum(dim=1) > (all_grad_gather > mean).sum(axis=1).float().mean() * self._epoch / 600)
            # choose_index = ((all_grad_gather > mean).sum(dim=1) > (all_grad_gather > mean).sum(axis=1).float().mean())
            pos_flag = ((all_grad_gather > mean).sum(axis=1) == 4)
            neg_flag = ~pos_flag
            pos_flag[torch.where(neg_flag == True)[0][:pos_flag.sum()]] = True
            # choose_index = ((all_grad_gather > mean).sum(axis=1) == 4)
            choose_index = pos_flag
            
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
                    
                    # os.popen(f'cp {source_img_file} {target_img_file}')

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


@RUNNERS.register_module()
class EfficientSampleGradOrientationDoubleCircleEpochBasedRunner(BaseRunner):
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
        self.sum_grad = None
        # self.iter_every_layer_grad = torch.zeros((0)).cuda()
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
            # print('-' * 10, grad.shape, grad.abs().sum().item())

            self.iter_every_layer_grad.append(grad.abs().sum().item())
            # pass
            # self.iter_every_layer_grad = torch.cat([self.iter_every_layer_grad, grad.reshape(-1)])
            # print(grad.shape, grad.reshape(-1).shape)
            # self.iter_every_layer_grad.append(grad.reshape(-1))
            # self.iter_every_layer_grad.append(grad.mean())


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
        self.output_loss = []
        self.all_img_ids = []

        import numpy as np
        temp_record = []
        self.all_temp_layer_grad = []
        self.grad_result = [0 for _ in range(4)]
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        # for i, data_batch in enumerate(self.data_loader):
        # print('dataloader length :', len(self.train_data_loader[-1]))
        self.cal_grad = False
        temp_train_dataloader = self.train_data_loader[-1] if self.cal_grad else self.train_data_loader[0]
        print('dataloader length : ', len(temp_train_dataloader))
        # for i, data_batch in enumerate(self.train_data_loader[-1]):
        for i, data_batch in enumerate(temp_train_dataloader):
            
            self._inner_iter = i
            print(f'{i} / {len(temp_train_dataloader)}', end='\r')

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

            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')

            self._iter += 1
            # ----------------------------------------------------------
            # self.every_layer_grad.append(self.iter_every_layer_grad.unsqueeze(0))
            # self.iter_every_layer_grad = torch.zeros((0)).cuda()
            # self.every_layer_grad.append(torch.tensor(self.iter_every_layer_grad))
            self.every_layer_grad.append(torch.tensor(self.iter_every_layer_grad).unsqueeze(0))
            self.iter_every_layer_grad = []
            self.output_loss.append(self.outputs['loss'].item())

            # added_tensor = torch.cat(self.iter_every_layer_grad, dim=0)
            # if self.sum_grad is None:
            #     self.sum_grad = torch.zeros_like(added_tensor)
            #     self.l1_arr = []
            # self.sum_grad += added_tensor.abs()
            # if i % 10 == 0:
            #     print(self.sum_grad.abs().sum())

            # self.iter_every_layer_grad = []
            if len(self.every_layer_grad) == 30:
                # self.save_grad_vector()        
                pass
                
        
        if len(self.every_layer_grad) > 0:
            # self.save_grad_vector()
            pass
        

        all_grad = 0
        for temp_grad in self.grad_result:
            all_grad += temp_grad
        
        # if self._epoch == 2:
        # self.get_grad_dist()
        # self.pick_grad_dataset(kwargs)
        self.call_hook('after_train_epoch')
        
        self.pick_grad_dataset(kwargs)
        block_epoch = 50
        if self._epoch > 0 and self._epoch % block_epoch == 0:
            self.train_data_loader = [self.train_data_loader[0]]
            self.cal_grad = False
        elif self._epoch % block_epoch == 1:
            if self.cal_grad is None or self.cal_grad == False:
                self.pick_grad_dataset(kwargs)
                self._epoch -= 1
                self.cal_grad = True
        print('Epoch : ', self._epoch, self.cal_grad)
        
        self._epoch += 1

    def save_grad_vector(self):
        import os
        import torch.nn.functional as F
        import h5py

        # target_dir = f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}'
        for i in range(3):
            target_dir = f'/mnt/hdd{i}/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}'
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            # file_name = f'iter_{self._inner_iter}_gpu_{torch.cuda.current_device()}.pt'
            file_name = f'grad_tensor.hdf5'
            save_grad_ori = F.normalize(torch.cat(self.every_layer_grad, dim=0), p=2, dim=1)
            f = h5py.File(os.path.join(target_dir, file_name), 'a')
            # if self.max_val is None:
            #     self.max_val = 0
            # if self.min_val is None:
            #     self.min_val = 1000
            # self.max_val = max(save_grad_ori.abs().max(), self.max_val)
            # self.min_val = min(save_grad_ori.abs().min(), self.min_val)

            save_grad_ori = save_grad_ori.cpu()

            torch.save(save_grad_ori, os.path.join(target_dir, file_name))
            self.every_layer_grad = []
            torch.cuda.empty_cache()
        
    def get_four_stages(self, all_grad_gather):
        #  'stage1' : 51, 'stage3': 108, 'stage4': 483
        ''' all_grad_gather : (N, 907) '''
        # temp_grad = all_grad_gather[:, ::-1]
        temp_grad = torch.flip(all_grad_gather, [1])
        return torch.cat(
            [
                temp_grad[:, :51].sum(axis=1, keepdim=True), 
                temp_grad[:, 51:108].sum(axis=1, keepdim=True),
                temp_grad[:, 108:483].sum(axis=1, keepdim=True),
                temp_grad[:, 483:].sum(axis=1, keepdim=True)
            ], dim=1
        )
    
    def get_block_list(self, file_num):
        block_id = torch.ones((file_num, file_num))
        block_id = torch.triu(block_id)
        x_id, y_id = torch.where(block_id == 1)
        block_id = torch.cat([x_id.unsqueeze(1), y_id.unsqueeze(1)], dim=1)

        return block_id

    def get_all_grad_file(self, file_list, world_size):
        for i in range(len(file_list)):
            file_list[i] = f"iter_{file_list[i].split('_')[1].zfill(6)}_gpu_{file_list[i].split('_')[-1].split('.')[0]}.pt"
        file_list.sort()
        out_file = []
        for i in range(world_size):
            out_file.extend(file_list[i::world_size])
        assert len(out_file) == len(file_list)
        for i in range(len(out_file)):
            out_file[i] = f"iter_{int(out_file[i].split('_')[1])}_gpu_{out_file[i].split('_')[-1].split('.')[0]}.pt"

        return out_file
    
    def get_self_block_ids_and_file_list(self):
        import os
        import math
        import torch.distributed as dist

        gpu_id = torch.cuda.current_device()
        dirname, _, file_name = next(os.walk(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}'))
        file_num = len(file_name)
        block_ids = self.get_block_list(file_num) # (K, 2)
        file_list = self.get_all_grad_file(file_name, dist.get_world_size())
        block_length = math.ceil(block_ids.shape[0] / dist.get_world_size())
        self_block_ids = block_ids[gpu_id * block_length : (gpu_id + 1) * block_length]

        return self_block_ids, file_list, block_length

    
    def calculate_grad_dist(self, q, block_size, gpu_id, in_pipe):
        import torch.nn.functional as F
        import torch
        import time

        count = 0
        time.sleep(3)
        print('start get tensor')
        while True:
            res = q.get()
            if res is None:
                break

            print(f'gpu:{gpu_id} {count} / {block_size}')
            x_id, y_id = res[0][0], res[0][1]
            in_pipe.send(res)
            # x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
            # x_grad = F.normalize(x_grad, p=2, dim=1)
            # y_grad = F.normalize(y_grad, p=2, dim=1)
            # print(x_grad.device, y_grad.device)
            # for i in range(x_grad.shape[0]):
            #     temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
            #     self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
                
            count += 1
            if count == block_size:
                q.task_done()
                return 

    @staticmethod
    def err_call_back(err):
        print(f'catch error：{str(err)}')

    def get_grad_dist_dataloader(self):
        import torch.nn.functional as F

        gpu_id = torch.cuda.current_device()
        block_ids, file_list = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()

        grad_file_dataset = GradDataset(file_list, block_ids, self._epoch)
        grad_file_dataloader = DataLoader(
            grad_file_dataset, 
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        for i, data in enumerate(grad_file_dataloader):
            print(i, 'gpu_id :', gpu_id)
            x_id, y_id = data[0][0], data[0][1]
            x_grad, y_grad = data[1][0].squeeze(), data[1][1].squeeze()
            print(data[0])
            print(x_grad.shape)
            x_grad = x_grad.cuda()
            y_grad = y_grad.cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)
            y_grad = F.normalize(y_grad, p=2, dim=1)
            for i in range(x_grad.shape[0]):
                temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
                self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
            print('over cal')


    
    def get_grad_dist_test(self):
        # from multiprocessing import Process, Pool, Pipe
        import math
        import torch.nn.functional as F
        from torch.multiprocessing import Process, Pool, Pipe
        from threading import Thread
        
        # import multiprocessing
        import torch
        # torch.multiprocessing.set_start_method('spawn')
        # print(torch.multiprocessing.get_start_method())


        gpu_id = torch.cuda.current_device()
        block_ids, file_list = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()
        out_pipe, in_pipe = Pipe(True)

        pool = Pool(5)
        q = multiprocessing.Manager().Queue()
        # tensor_p = Process(target=self.calculate_grad_dist, args=(q, block_ids.shape[0], gpu_id, in_pipe))

        # id_list = [[i, i + 1] for i in range(60)]
        result_list = []
        for i in range(block_ids.shape[0]):
            ret = pool.apply_async(
                func=read_grad_oriental_file_single, 
                args=(q, block_ids[i], file_list, self._epoch, gpu_id), 
                error_callback=self.err_call_back
            )
            result_list.append(ret)
        # process_num = 5
        # task_list = []
        # for i in range(process_num):
        #     single_size = math.ceil(block_ids.shape[0] / process_num)
        #     p_task = Process(
        #         target=read_grad_oriental_file_batch, 
        #         args=(q, block_ids[i * single_size : (i + 1) * single_size], file_list, self._epoch, gpu_id, i)
        #     )
        #     task_list.append(p_task)
        #     p_task.start()
        
        # tensor_p.start()
        # self.calculate_grad_dist(q, block_ids.shape[0])

        count = 0
        block_size = block_ids.shape[0]
        print('GPU id :', torch.cuda.current_device())
        while True:
            res = q.get()
            if res is None:
                break

            print(f'gpu:{gpu_id} {count} / {block_size}')
            x_id, y_id = res[0][0], res[0][1]
            x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
            print(x_grad.shape, y_grad.shape)
            x_grad = F.normalize(x_grad, p=2, dim=1)
            y_grad = F.normalize(y_grad, p=2, dim=1)
            for i in range(x_grad.shape[0]):
                temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
                self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
            
            count += 1
            if count == block_size:
                break
    
        # count = 1
        # while True:
        #     try:
        #         res = out_pipe.recv()
        #         x_id, y_id = res[0][0], res[0][1]
        #         print(res[0])
        #         x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
        #         x_grad = F.normalize(x_grad, p=2, dim=1)
        #         y_grad = F.normalize(y_grad, p=2, dim=1)
        #         for i in range(x_grad.shape[0]):
        #             temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
        #             self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
        #         count += 1
        #         if count == block_ids.shape[0]:
        #             break
        #     except:
        #         break
        
        pool.close()
        pool.join()
        # for task in task_list:
        #     task.join()
        # q.join()
        # tensor_p.join()
        
        print('over')
   
    def get_grad_dist_one_by_one(self):
        # from multiprocessing import Process, Pool, Pipe
        import math
        import torch.nn.functional as F
        from torch.multiprocessing import Process, Pool, Pipe
        from threading import Thread
        
        # import multiprocessing
        import torch
        # torch.multiprocessing.set_start_method('spawn')
        # print(torch.multiprocessing.get_start_method())

        gpu_id = torch.cuda.current_device()
        block_ids, file_list = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()
        out_pipe, in_pipe = Pipe(True)

        pool = Pool(5)
        q = multiprocessing.Manager().Queue()
        # tensor_p = Process(target=self.calculate_grad_dist, args=(q, block_ids.shape[0], gpu_id, in_pipe))

        # id_list = [[i, i + 1] for i in range(60)]
        flag_block_ids =  torch.cat(
            [torch.tensor([i, block_ids[i][0], i, block_ids[i][1]]) for i in range(block_ids.shape[0])]
        ).reshape(-1, 2)
        result_list = []
        for i in range(flag_block_ids.shape[0]):
        # for i in range(10):
            ret = pool.apply_async(
                func=read_grad_oriental_file_one_by_one, 
                args=(q, flag_block_ids[i], file_list, self._epoch, gpu_id), 
                error_callback=self.err_call_back
            )
            result_list.append(ret)
            # ret = pool.apply_async(
            #     func=read_grad_oriental_file_one_by_one, 
            #     args=(q, [i, block_ids[i][1]], file_list, self._epoch, gpu_id), 
            #     error_callback=self.err_call_back
            # )
            # result_list.append(ret)

        # process_num = 5
        # task_list = []
        # for i in range(process_num):
        #     single_size = math.ceil(block_ids.shape[0] / process_num)
        #     p_task = Process(
        #         target=read_grad_oriental_file_batch, 
        #         args=(q, block_ids[i * single_size : (i + 1) * single_size], file_list, self._epoch, gpu_id, i)
        #     )
        #     task_list.append(p_task)
        #     p_task.start()
        
        # tensor_p.start()
        # self.calculate_grad_dist(q, block_ids.shape[0])

        count = 0
        block_size = block_ids.shape[0]
        buffer_block = [[-1, -1] for _ in range(block_size)]
        print('GPU id :', torch.cuda.current_device())
        while True:
            res = q.get()
            if res is None:
                break

            # print(f'gpu:{gpu_id} {count} / {block_size}')
            # x_id, y_id = res[0][0], res[0][1]
            # x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
            # print(x_grad.shape, y_grad.shape)
            flag_id, temp_block_id = res[0]
            temp_grad = res[1]
            if buffer_block[flag_id][0] != -1:
                print(f'gpu id {gpu_id} find pair')
                x_id, y_id = buffer_block[flag_id][0], temp_block_id
                x_grad, y_grad = buffer_block[flag_id][1], temp_grad
                x_grad, y_grad = x_grad.cuda(), y_grad.cuda()

                x_grad = F.normalize(x_grad, p=2, dim=1)
                y_grad = F.normalize(y_grad, p=2, dim=1)
                for i in range(x_grad.shape[0]):
                    temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
                    self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
                temp_grad = buffer_block[flag_id][1]
                del temp_grad
                buffer_block[flag_id][0] = buffer_block[flag_id][1] = -1
                print(f'gpu:{gpu_id} {count} / {block_size}')
                count += 1
            else:
                buffer_block[flag_id][0] = temp_block_id
                buffer_block[flag_id][1] = temp_grad
            
            if count == block_size:
                break
    
        # count = 1
        # while True:
        #     try:
        #         res = out_pipe.recv()
        #         x_id, y_id = res[0][0], res[0][1]
        #         print(res[0])
        #         x_grad, y_grad = res[1][0].cuda(), res[1][1].cuda()
        #         x_grad = F.normalize(x_grad, p=2, dim=1)
        #         y_grad = F.normalize(y_grad, p=2, dim=1)
        #         for i in range(x_grad.shape[0]):
        #             temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
        #             self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
        #         count += 1
        #         if count == block_ids.shape[0]:
        #             break
        #     except:
        #         break
        
        pool.close()
        pool.join()
        # for task in task_list:
        #     task.join()
        # q.join()
        # tensor_p.join()
        
        print('over')

    def get_grad_dist(self):
        import os
        import torch.nn.functional as F

        gpu_id = torch.cuda.current_device()
        block_ids, file_list, block_length = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        block_num = len(file_list)
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()

        count = 0
        y_flag = True
        for i in range(block_ids[0][0], block_num):
            print(f'gpu:{gpu_id}-{i}')
            x_file_name = file_list[i]
            x_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', x_file_name)
            ).cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)

            init_j = block_ids[0][1] if y_flag else i + 1
            y_flag = False
            for j in range(init_j, block_num):
                count += 1

                y_file_name = file_list[j]
                y_grad = torch.load(
                    os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', y_file_name)
                ).cuda()
                y_grad = F.normalize(y_grad, p=2, dim=1)    

                print(f'gpu:{gpu_id} {count} / {len(block_ids)}')

                for t in range(x_grad.shape[0]):
                    temp_dist = ((x_grad[t] - y_grad) ** 2).sum(dim=1)
                    self.all_grad_dist[i*30 + t, j*30 : j*30 + y_grad.shape[0]] = temp_dist

                if count == block_length:
                    return 

    def get_grad_dist_loop_multiprocess(self):
        import os
        import torch.nn.functional as F
        from torch.multiprocessing import Process, Pool, Pipe
        import torch

        gpu_id = torch.cuda.current_device()
        block_ids, file_list, block_length = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        block_num = len(file_list)
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()

        count = 0
        y_flag = True
        for i in range(block_ids[0][0], block_num):
            print(f'gpu:{gpu_id}-{i}')
            x_file_name = file_list[i]
            x_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', x_file_name)
            ).cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)

            init_j = block_ids[0][1] if y_flag else i + 1
            y_flag = False
            temp_pool = Pool(10)
            q = multiprocessing.Manager().Queue()

            result_list = []
            for j in range(init_j, block_num):
                ret = temp_pool.apply_async(
                    func=read_grad_oriental_file_one_by_one,
                    args=(q, [0, j], file_list, self._epoch, gpu_id,),
                    error_callback=self.err_call_back
                )
                result_list.append(ret)

            
            j_num = 0
            while True:
                res = q.get()
                if res == None:
                    break
                count += 1
                j_num += 1

                print(f'gpu:{gpu_id} {count} / {len(block_ids)}')
                y_id = res[0][1]
                y_grad = res[1]
                y_grad = y_grad.cuda()
                y_grad = F.normalize(y_grad, p=2, dim=1)    

                for t in range(x_grad.shape[0]):
                    temp_dist = ((x_grad[t] - y_grad) ** 2).sum(dim=1)
                    self.all_grad_dist[i*30 + t, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist

                if j_num == block_num - init_j:
                    break
                if count == block_length:
                    return 

            temp_pool.close()
            temp_pool.join()

    def get_grad_dist_mul_dataloader(self):
        import os
        import torch.nn.functional as F
        from torch.multiprocessing import Process, Pool, Pipe
        import torch

        gpu_id = torch.cuda.current_device()
        block_ids, file_list, block_length = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        block_num = len(file_list)
        self.all_grad_dist = torch.zeros((grad_size, grad_size)).cuda()

        count = 0
        y_flag = True
        for i in range(block_ids[0][0], block_num):
            
            x_file_name = file_list[i]
            x_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', x_file_name)
            ).cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)
            print(f'gpu:{gpu_id}-{i}')

            init_j = block_ids[0][1] if y_flag else i + 1
            grad_dataset = GradDataset(file_list, [t for t in range(init_j, block_num)], self._epoch, gpu_id)
            grad_dataloader = DataLoader(
                dataset=grad_dataset,
                pin_memory=True,
                batch_size=1,
                num_workers=2,
                prefetch_factor=1
            )

            for _, data in enumerate(grad_dataloader):
                y_id, y_grad = data

                count += y_grad.shape[0]
                print(f'gpu:{gpu_id} {count} / {len(block_ids)}')
                y_grad = y_grad.squeeze().cuda()
                y_grad = F.normalize(y_grad, p=2, dim=1)    

                for t in range(x_grad.shape[0]):
                    temp_dist = ((x_grad[t] - y_grad) ** 2).sum(dim=1)
                    self.all_grad_dist[i*30 + t, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist

                if count == block_length:
                    return 

    def test_get_grad_dist(self):
        import os
        import torch.nn.functional as F

        gpu_id = torch.cuda.current_device()
        block_ids, file_list = self.get_self_block_ids_and_file_list() # (N, 2) where N = file_num / world_size
        grad_size = len(file_list) * 30
        self.all_grad_dist = torch.zeros((grad_size, grad_size))
        for k, (x_id, y_id) in enumerate(block_ids):
            print(f'gpu:{gpu_id} {k} / {len(block_ids)}')
            # x_file_name = f'iter_{x_id * 30 + 29}_gpu_{gpu_id}.pt'
            # y_file_name = f'iter_{y_id * 30 + 29}_gpu_{}.pt'
            x_file_name = file_list[x_id]
            y_file_name = file_list[y_id]
            x_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', x_file_name)
            ).cuda()
            x_grad = F.normalize(x_grad, p=2, dim=1)
            y_grad = torch.load(
                os.path.join(f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}', y_file_name)
            ).cuda()
            y_grad = F.normalize(y_grad, p=2, dim=1)

            for i in range(x_grad.shape[0]):
                temp_dist = ((x_grad[i] - y_grad) ** 2).sum(dim=1)
                self.all_grad_dist[x_id*30 + i, y_id*30 : y_id*30 + y_grad.shape[0]] = temp_dist
            # break


    def pick_grad_dataset(self, kwargs):
        print('start to pick grad dataset')
        import os
        import numpy as np
        import torch.distributed as dist
        ROOT = 0
        # every_layer_grad = torch.from_numpy(np.array(self.every_layer_grad)).cuda() # (N/batch_size/world_size, 907)

        # self.get_grad_dist()
        # self.all_grad_dist = self.all_grad_dist.cuda()
        every_gpu_loss = torch.tensor(self.output_loss).cuda()
        every_layer_grad = torch.cat(self.every_layer_grad, dim=0).cuda()
        all_img_ids = torch.from_numpy(np.array(self.all_img_ids, dtype=np.int32)).cuda()   # (N/batch_size/world_size, batch_size)
        assert all_img_ids.shape[0] == every_layer_grad.shape[0]
        
        self.every_layer_grad = []        
        self.all_img_ids = []
        this_rank = dist.get_rank()
        world_size = dist.get_world_size()
        # print(f'rank : {this_rank}, world_size : {world_size}')
        
        # grad_gather_list = [torch.zeros_like(self.all_grad_dist) for _ in range(world_size)]
        loss_gather_list = [torch.zeros_like(every_gpu_loss) for _ in range(world_size)]
        grad_gather_list = [torch.zeros_like(every_layer_grad) for _ in range(world_size)]
        imgids_gather_list = [torch.zeros_like(all_img_ids) for _ in range(world_size)]
        broadcast_tensor = torch.tensor([this_rank], dtype=torch.int32).cuda()
        broadcast_list = [torch.zeros_like(broadcast_tensor) for _ in range(world_size)]
        print(f'{this_rank} send every layer grad')
        # dist.all_gather(grad_gather_list, self.all_grad_dist.cuda(), async_op=False)
        dist.all_gather(grad_gather_list, every_layer_grad, async_op=False)
        dist.all_gather(imgids_gather_list, all_img_ids, async_op=False)
        dist.all_gather(loss_gather_list, every_gpu_loss, async_op=False)
        if this_rank == 0:
            print('all grad from other gpu :', len(grad_gather_list))
            print(grad_gather_list[0].shape)

            all_grad_gather = torch.cat(grad_gather_list, dim=0) # (N/batch_size, 878(HRNet)907(higherHRNet))

            # all_grad_gather = torch.zeros_like(grad_gather_list[0])
            # for temp_grad_dist in grad_gather_list:
            #     all_grad_gather += temp_grad_dist
            # torch.save(all_grad_gather, f'/home/chenbeitao/data/code/Test/txt/orientation/epoch_{self._epoch}/grad-dist/grad_dist.pt')
            # torch.save(all_grad_gather, f'/home/chenbeitao/data/code/Test/txt/orientation/double-circle/grad_dist_epoch_{self._epoch}.pt')
            
            
            
            all_imgids_gather = torch.cat(imgids_gather_list, dim=0)
            all_loss = torch.cat(loss_gather_list, dim=0)
            torch.save(all_loss, f'/home/chenbeitao/data/code/Test/txt/orientation/double-circle/check_grad_dist_epoch_{self._epoch}_loss_normal.pt')
            torch.save(all_grad_gather, f'/home/chenbeitao/data/code/Test/txt/orientation/double-circle/check_grad_dist_epoch_{self._epoch}_grad_normal.pt')
            torch.save(all_imgids_gather, f'/home/chenbeitao/data/code/Test/txt/orientation/double-circle/check_grad_dist_epoch_{self._epoch}_image_id_normal.pt')
            assert 0 == 1
            # all_grad_gather = self.get_four_stages(all_grad_gather)
            assert all_grad_gather.shape[0] == all_imgids_gather.shape[0]

            mean = all_grad_gather.mean(dim=0)
            all_grad_dist = (all_grad_gather - mean).abs().sum(dim=1)
            all_grad_dist[all_grad_dist == 0] = all_grad_dist[all_grad_dist > 0].min() * 0.01
            mul_scale = 1
            if all_grad_dist.min() < 1:
                mul_scale = 10 ** (torch.floor(torch.log10(all_grad_dist.min())) * -1)
            all_grad_dist *= mul_scale
            assert all_grad_dist.shape[0] == all_grad_gather.shape[0]

            all_grad_dist_order, all_grad_dist_index = torch.sort(all_grad_dist)
            
            mul_scale = 1
            all_grad_L1_scale = all_grad_gather.abs().sum(dim=1)
            all_grad_L1_scale[all_grad_L1_scale == 0] = all_grad_L1_scale[all_grad_L1_scale > 0].min() * 0.01
            if all_grad_L1_scale.min() < 1:
                mul_scale = 10 ** (torch.floor(torch.log10(all_grad_L1_scale.min())) * -1)
            all_grad_L1_scale *= mul_scale
            # assert all_grad_dist.min() >= 1
            # assert all_grad_L1_scale.min() >= 1
            assert all_grad_L1_scale.shape == all_grad_dist.shape
            assert all_grad_L1_scale.shape == all_grad_dist_index.shape
            all_grad_L1_scale_order = all_grad_L1_scale[all_grad_dist_index]

            choose_index = torch.zeros_like(all_grad_dist_index)
            ''' first pick dataset '''
            p1_grad_vector = (all_grad_dist_order / all_grad_dist_order.min()) * all_grad_L1_scale_order
            first_random_thr = torch.randn(p1_grad_vector.shape).cuda() * p1_grad_vector.std() + p1_grad_vector.mean()
            choose_index[all_grad_dist_index[p1_grad_vector > first_random_thr]] = 1
            ''' second pick dataset '''
            p2_grad_vector = (all_grad_dist_order.max() / all_grad_dist_order) * all_grad_L1_scale_order
            # second_random_thr = torch.rand(p2_grad_vector.shape).cuda() * p2_grad_vector.max()
            second_random_thr = torch.randn(p2_grad_vector.shape).cuda() * p2_grad_vector.std() + p2_grad_vector.mean()
            # choose_index[p2_grad_vector > second_random_thr] = 1
            choose_index[all_grad_dist_index[p2_grad_vector > second_random_thr]] = 1

            choose_index = (choose_index == 1)
            print('choose num : ', choose_index.sum())


            # import matplotlib.pyplot as plt
            # x = [_ for _ in range(len(mean))]
            y = mean
            # plt.plot(x, y)
            # plt.savefig(f"/home/chenbeitao/data/code/Test/txt/out_result{torch.cuda.current_device()}.jpg")

            # var = all_grad_gather.var(dim=0)
            # norm_all_grad_gather = (all_grad_gather - mean) / (var + 1e-5)
            out_all_grad_gather = all_grad_gather.cpu().numpy()
            # np.savetxt(
            #     f"/home/chenbeitao/data/code/Test/txt/orientation/grad_{torch.cuda.current_device()}_{str(self._epoch)}.txt", 
            #     # f"/home/chenbeitao/data/code/Test/txt/higher/stage4_pretrained_grad_{torch.cuda.current_device()}_{str(self._epoch)}.txt", 
            #     out_all_grad_gather
            # )
            
            # choose_index = ((all_grad_gather > mean).sum(dim=1) > all_grad_gather.shape[1]/2)
            # (((out1 > out1.mean(axis=0)*(1 + i/100)).sum(axis=1) > (out1>out1.mean(axis=0)).sumaxis=1).mean() * i/30).sum())
            # choose_index = ((all_grad_gather > mean*(1+self._epoch/1000)).sum(dim=1) > (all_grad_gather > mean).sum(axis=1).float().mean() * self._epoch / 600)
            # choose_index = ((all_grad_gather > mean).sum(dim=1) > (all_grad_gather > mean).sum(axis=1).float().mean())
            
            choose_img_ids = all_imgids_gather[choose_index].reshape(-1).cpu().numpy().tolist()
            # choose_top_grad_img_index = all_imgids_gather[top_grad_img_index].reshape(-1).cpu().numpy().tolist()
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

                # if img['id'] in choose_top_grad_img_index:
                #     file_name = img['file_name']
                #     source_img_file = os.path.join(
                #         '/home/chenbeitao/data/code/mmlab/mmpose/data/coco/train2017',
                #         file_name
                #     )
                #     target_dir = f'/home/chenbeitao/data/code/Test/txt/grad-image/epoch_{self._epoch}'
                #     if not os.path.exists(target_dir):
                #         os.makedirs(target_dir)
                #     target_img_file = os.path.join(
                #         target_dir,
                #         file_name
                #     )
                    
                #     os.popen(f'cp {source_img_file} {target_img_file}')

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
        # print('new_data_loader :', len(new_data_loaders))
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
            train_loader_cfg=None, cfg=None, single_pick_data_loader=None, **kwargs):
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

        self.train_data_loader = [single_pick_data_loader[0], data_loaders[0]]
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
