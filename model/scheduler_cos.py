#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import math
from utils import logger
import numpy as np


class CosineScheduler(object):
    """
    Cosine learning rate scheduler: https://arxiv.org/abs/1608.03983
    """

    def __init__(self, opts, **kwargs) -> None:
        super().__init__()
        self.opts = opts
        self.round_places = 8
        self.lr_multipliers = getattr(opts, "optim_lr_multipliers", None)
        is_iter_based = getattr(opts, "scheduler_is_iteration_based", True)
        warmup_iterations = getattr(opts, "scheduler_warmup_iterations", 500)
        max_iterations = getattr(opts, "scheduler_max_iterations", 2000)

        self.min_lr = getattr(opts, "scheduler_cosine_min_lr", 1e-5)
        self.max_lr = getattr(opts, "scheduler_cosine_max_lr", 0.4)

        self.warmup_iterations = max(warmup_iterations, 0)
        if self.warmup_iterations > 0:
            warmup_init_lr = getattr(opts, "scheduler_warmup_init_lr", 1e-7)
            self.warmup_init_lr = warmup_init_lr
            self.warmup_step = (self.max_lr - self.warmup_init_lr) / self.warmup_iterations

        self.period = (max_iterations - self.warmup_iterations + 1) if is_iter_based \
            else getattr(opts, "scheduler_period_epochs", 50)
        self.is_iter_based = is_iter_based

    def update_lr(self, optimizer, epoch: int, curr_iter: int,task_name=None):
        lr = self.get_lr(epoch=epoch, curr_iter=curr_iter)
        lr = max(0.0, lr)
        if self.lr_multipliers is not None:
            assert len(self.lr_multipliers) == len(optimizer.param_groups)
            for g_id, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = round(lr * self.lr_multipliers[g_id], self.round_places)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = round(lr, self.round_places)
            if task_name == 'object_detection':
                optimizer.param_groups[0]['lr'] = lr * 0.000001
        return optimizer

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
        else:
            if self.is_iter_based:
                curr_iter = curr_iter - self.warmup_iterations
                curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                            1 + math.cos(math.pi * curr_iter / self.period))
            else:
                curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                        1 + math.cos(math.pi * epoch / self.period))
        return max(0.0, curr_lr)

    @staticmethod
    def retrieve_lr(optimizer) -> list:
        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list


class CyclicLRScheduler(object):
    """
    Cosine learning rate scheduler: https://arxiv.org/abs/1608.03983
    """

    def __init__(self, opts, **kwargs) -> None:
        super().__init__()
        cycle_steps = getattr(opts, "scheduler_cyclic_steps", [25])
        if cycle_steps is not None and isinstance(cycle_steps, int):
            cycle_steps = [cycle_steps]
        gamma = getattr(opts, "scheduler_cyclic_gamma", 0.5)
        anneal_type = getattr(opts, "scheduler_cyclic_last_cycle_type", "linear")
        min_lr = getattr(opts, "scheduler_cyclic_min_lr", 0.1)
        end_lr = getattr(opts, "scheduler_cyclic_last_cycle_end_lr", 1e-3)
        ep_per_cycle = getattr(opts, "scheduler_cyclic_epochs_per_cycle", 5)
        warmup_iterations = getattr(opts, "scheduler_warmup_iterations", 0)
        n_cycles = getattr(opts, "scheduler_cyclic_total_cycles", 10) - 1
        max_epochs = getattr(opts, "scheduler_max_epochs", 100)
        if min_lr < end_lr:
            logger.error("Min LR should be greater than end LR. Got: {} and {}".format(min_lr, end_lr))

        self.min_lr = min_lr
        self.cycle_length = ep_per_cycle
        self.end_lr = end_lr
        self.max_lr = self.min_lr * self.cycle_length
        self.last_cycle_anneal_type = anneal_type
        self.warmup_iterations = max(warmup_iterations, 0)
        if self.warmup_iterations > 0:
            warmup_init_lr = getattr(opts, "scheduler.warmup_init_lr", 1e-7)
            self.warmup_init_lr = warmup_init_lr
            self.warmup_step = (self.min_lr - self.warmup_init_lr) / self.warmup_iterations

        self.n_cycles = n_cycles
        self.cyclic_epochs = self.cycle_length * self.n_cycles
        self.max_epochs = max_epochs
        self.last_cycle_epochs = self.max_epochs - self.cyclic_epochs
        assert self.max_epochs == self.cyclic_epochs + self.last_cycle_epochs

        self.steps = [self.max_epochs] if cycle_steps is None else cycle_steps
        self.gamma = gamma if cycle_steps is not None else 1

        self._lr_per_cycle()
        self.epochs_lr_stepped = []

    def update_lr(self, optimizer, epoch: int, curr_iter: int):
        lr = self.get_lr(epoch=epoch, curr_iter=curr_iter)
        lr = max(0.0, lr)
        if self.lr_multipliers is not None:
            assert len(self.lr_multipliers) == len(optimizer.param_groups)
            for g_id, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = round(lr * self.lr_multipliers[g_id], self.round_places)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = round(lr, self.round_places)
        return optimizer

    def _lr_per_cycle(self) -> None:
        lrs = list(np.linspace(self.max_lr, self.min_lr, self.cycle_length, dtype=np.float))
        lrs = [lrs[-1]] + lrs[:-1]
        self.cycle_lrs = lrs

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
        else:
            if epoch <= self.cyclic_epochs:
                if epoch in self.steps and epoch not in self.epochs_lr_stepped:
                    self.min_lr *= (self.gamma ** (self.steps.index(epoch) + 1))
                    self.max_lr *= (self.gamma ** (self.steps.index(epoch) + 1))
                    self._lr_per_cycle()
                    self.epochs_lr_stepped.append(epoch)
                idx = (epoch % self.cycle_length)
                curr_lr = self.cycle_lrs[idx]
            else:
                base_lr = self.min_lr
                if self.last_cycle_anneal_type == 'linear':
                    lr_step = (base_lr - self.end_lr) / self.last_cycle_epochs
                    curr_lr = base_lr - (epoch - self.cyclic_epochs + 1) * lr_step
                elif self.last_cycle_anneal_type == 'cosine':
                    curr_epoch = epoch - self.cyclic_epochs
                    period = self.max_epochs - self.cyclic_epochs + 1
                    curr_lr = self.end_lr + 0.5 * (base_lr - self.end_lr) * (
                                1 + math.cos(math.pi * curr_epoch / period))
                else:
                    raise NotImplementedError
        return max(0.0, curr_lr)

    @staticmethod
    def retrieve_lr(optimizer) -> list:
        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list


if __name__ == '__main__':
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
    import torch.nn as nn
    from torchvision.models import resnet18
    import matplotlib.pyplot as plt

    #
    # model=resnet18(pretrained=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    '''
    mode='cosineAnnWarm'
    if mode=='cosineAnn':
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    elif mode=='cosineAnnWarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2)

        以T_0=5, T_mult=1为例:
        T_0:学习率第一次回到初始值的epoch位置.
        T_mult:这个控制了学习率回升的速度
            - 如果T_mult=1,则学习率在T_0,2*T_0,3*T_0,....,i*T_0,....处回到最大值(初始学习率)
                - 5,10,15,20,25,.......处回到最大值
            - 如果T_mult>1,则学习率在T_0,(1+T_mult)*T_0,(1+T_mult+T_mult**2)*T_0,.....,(1+T_mult+T_mult**2+...+T_0**i)*T0,处回到最大值
                - 5,15,35,75,155,.......处回到最大值
        example:
            T_0=5, T_mult=1
    '''
    import argparse
    from pathlib import Path
    from utils.common_utils import load_config_file

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    parser = argparse.ArgumentParser(description='Training arguments', add_help=True)
    parser.add_argument('--common_config_file', type=str,
                        default=r'E:\new python\MobileVIT_WP2\mobilevitv3_small_multiserver.yaml',
                        help="Configuration file")
    opts = parser.parse_args()
    opts = load_config_file(opts)
    scheduler = CosineScheduler(opts)
    plt.figure()
    max_epoch = 200
    iters = 3785
    cur_lr_list = []
    iter = 0
    for epoch in range(max_epoch):
        print('epoch_{}'.format(epoch))
        for batch in range(iters):
            # scheduler.step(epoch + batch / iters)
            # optimizer.step()
            # cur_lr=optimizer.param_groups[-1]['lr']

            cur_lr = scheduler.get_lr(epoch, iter)
            iter = iter + 1
            print('cur_lr:', cur_lr)
            cur_lr_list.append(cur_lr)
        print('epoch_{}_end'.format(epoch))
    x_list = list(range(len(cur_lr_list)))
    plt.plot(x_list, cur_lr_list)
    plt.show()
    t = 5
