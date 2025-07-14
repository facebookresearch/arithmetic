# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

from torch import optim


class AdamWCosineWithWarmup(optim.AdamW):
    def __init__(
        self,
        params,
        lr=5e-5,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.1,
        warmup_updates=1000,
        warmup_init_lr=1e-8,
        min_lr=5e-6,
        total_iterations=-1,
    ):
        super().__init__(params, lr=warmup_init_lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr

        # then, apply cosine scheduler
        self.min_lr = min_lr
        self.max_lr = lr
        self.total_iterations = total_iterations

        # total number of updates
        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * (self.max_lr - self.warmup_init_lr) / self.warmup_updates
        else:
            t = num_updates - self.warmup_updates
            T = self.total_iterations
            assert t <= self.total_iterations, t
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * t / T))

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(param_group["num_updates"])


def get_optimizer(parameters, params):
    optim_params = {}
    optim_fn = AdamWCosineWithWarmup
    optim_params["lr"] = params.lr
    optim_params["weight_decay"] = params.weight_decay
    optim_params["total_iterations"] = params.steps_per_epoch * params.epochs
    optim_params["min_lr"] = 0.1 * params.lr

    return optim_fn(parameters, **optim_params)
