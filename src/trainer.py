# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from logging import getLogger
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

import wandb

from .optim import get_optimizer
from .utils import to_xy, to_angle
from .dataset import data_gen

logger = getLogger()


class CustomLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(CustomLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        pred_norm = torch.norm(inputs, dim=1, keepdim=True)
        loss_regularizer = (pred_norm**2) + 1.0 / (pred_norm**2 + 1e-6)
        loss_mse = torch.nn.functional.mse_loss(inputs.float(), targets.float(), reduction="none").sum(dim=1).unsqueeze(1)
        if self.reduction == "mean":
            loss_regularizer = torch.mean(loss_regularizer)
            loss_mse = torch.mean(loss_mse)
        return loss_regularizer, loss_mse

    def __repr__(self):
        return "MSE + norm"


class AngleLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(AngleLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        angle_inputs = to_angle(inputs)
        angle_targets = to_angle(targets)
        difference = angle_inputs - angle_targets
        difference = torch.where(
            difference > torch.pi, difference - 2 * torch.pi, torch.where(difference < -torch.pi, difference + 2 * torch.pi, difference)
        )
        loss = (difference**2).unsqueeze(1)
        if self.reduction == "mean":
            loss = torch.mean(loss)
        return loss


class MSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss_mse = torch.nn.functional.mse_loss(inputs.float(), targets.float(), reduction="none").sum(dim=1).unsqueeze(1)
        if self.reduction == "mean":
            loss_mse = torch.mean(loss_mse)
        return loss_mse


class Trainer:
    def __init__(self, model, params):
        self.model = model
        self.params = params

        if params.loss_type == "MSE":
            self.criterion = MSELoss(reduction="none")
        elif params.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss(reduction="none")
        elif params.loss_type == "angle":
            self.criterion = AngleLoss(reduction="none")
        else:
            self.criterion = CustomLoss(reduction="none")

        self.set_parameters()

        if params.world_size > 1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=False
            )

        self.set_optimizer()

        self.scaler = None
        if params.amp:
            self.scaler = torch.amp.GradScaler("cuda")

        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            if "loss" in m:
                biggest = False
            elif "acc" in m:
                biggest = True
            else:
                raise NotImplementedError(f"{m} has not been implemented yet.")
            self.metrics.append((m, biggest))
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        self.epoch = 0
        self.n_total_iter = 0
        self.stats = {"loss": []}
        if params.loss_type.startswith("custom"):
            self.stats |= {"loss_regularizer": []}
            self.stats |= {"loss_mse": []}
        if params.amp:
            self.stats["scaler"] = []

        self.dataloader = data_gen(params, train_eval_test_type="train")
        self.reload_checkpoint()

    def set_parameters(self):
        self.parameters = {}
        named_params = []
        named_params.extend([(k, p) for k, p in self.model.named_parameters() if p.requires_grad])
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            assert len(v) >= 1

    def set_optimizer(self):
        self.optimizer = get_optimizer(self.parameters["model"], self.params)

    def optimize(self, loss):
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            exit()

        optimizer = self.optimizer

        if not self.params.amp:
            optimizer.zero_grad()
            loss.backward()
            if self.params.clip_grad_norm > 0:
                clip_grad_norm_(self.parameters["model"], self.params.clip_grad_norm)
            optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            if self.params.clip_grad_norm > 0:
                self.scaler.unscale_(optimizer)
                clip_grad_norm_(self.parameters["model"], self.params.clip_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
            self.stats["scaler"].append(self.scaler.get_scale())
            optimizer.zero_grad()

    def save_checkpoint(self, name, include_optimizer=True):
        if not self.params.is_master:
            return

        path = os.path.join(self.params.output_dir, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "params": {k: v for k, v in self.params.__dict__.items()},
            "easy_hard": self.dataloader.dataset.easy_hard,
        }

        logger.warning(f"Saving model parameters ...")
        data["model"] = self.model.state_dict()

        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        torch.save(data, path)

    def reload_checkpoint(self):
        checkpoint_path = (
            os.path.join(self.params.output_dir, "checkpoint.pth")
            if getattr(self.params, "reload_exp", None) is None
            else os.path.join(self.params.reload_exp, "checkpoint.pth")
        )

        if not os.path.isfile(checkpoint_path):
            return
        else:
            new_file = checkpoint_path

        logger.warning(f"Reloading checkpoint from {new_file} ...")
        data = torch.load(new_file, map_location="cpu", weights_only=False)

        if self.params.world_size > 1:
            torch.distributed.barrier()

        data["model"] = {
            k.replace("module._orig_mod.", "XYZXYZ")
            .replace("_orig_mod.", "XYZXYZ")
            .replace("XYZXYZ", "module._orig_mod." if self.params.world_size > 1 else "_orig_mod."): v
            for k, v in data["model"].items()
        }

        self.model.load_state_dict(data["model"])
        logger.warning("Reloading checkpoint optimizer ...")
        self.optimizer.load_state_dict(data["optimizer"])

        if self.params.amp:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])
        else:
            assert self.scaler is None and "scaler" not in data

        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.dataloader.dataset.easy_hard = data.get("easy_hard", "all")
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[metric]))
                if self.params.wandb:
                    self.save_checkpoint("best-%s" % metric)

    def end_epoch(self):
        if self.params.wandb:
            self.save_checkpoint("checkpoint")
        self.epoch += 1

    def train_epoch(self):
        model = self.model
        params = self.params

        model.train()
        self.dataloader.dataset.epoch = self.epoch

        steps = 0

        while True:
            for x, s_k, y in self.dataloader:
                x, s_k, y = x.cuda(), s_k.cuda(), y.cuda()
                with torch.amp.autocast("cuda", enabled=self.params.amp):
                    output = model(x, s_k)["output"]
                    if params.loss_type == "CE":
                        loss = self.criterion(output, y)
                        all_losses = [(loss, "loss")]
                    elif params.loss_type in ["MSE", "angle"]:
                        loss = self.criterion(output.float(), to_xy(y, self.params.Q))
                        all_losses = [(loss, "loss")]
                    else:
                        loss_regularizer, loss_mse = self.criterion(output.float(), to_xy(y, self.params.Q))
                        loss = loss_mse + float(params.loss_type.split(";")[1]) * loss_regularizer
                        all_losses = [(loss, "loss"), (loss_regularizer, "loss_regularizer"), (loss_mse, "loss_mse")]
                for loss_type, loss_str in all_losses:
                    self.stats[f"{loss_str}"].append(loss_type.mean().item())
                loss = loss.mean()
                self.optimize(loss)

                self.n_total_iter += 1
                if self.n_total_iter % 200 == 0:
                    if self.dataloader.dataset.easy_hard == "easy" and np.mean(self.stats["loss"]) < 1e-2:
                        self.dataloader.dataset.close_files()
                        self.dataloader.dataset.easy_hard = self.params.train_with_cl.split("_")[1]
                        self.dataloader.dataset.open_files(train_eval_test_type="train")
                        logger.info("Switching from easy to hard data...")

                    s_iter = "%7i - " % self.n_total_iter
                    s_stat = " || ".join(
                        [
                            "{}: {:7.4f}".format(k.upper().replace("_", "-"), np.mean(v))
                            for k, v in self.stats.items()
                            if type(v) is list and len(v) > 0
                        ]
                    )

                    if params.wandb and params.is_master:
                        wandb.log(
                            {"train/step": self.n_total_iter}
                            | {f"train/{k}": np.mean(v) for k, v in self.stats.items() if type(v) is list and len(v) > 0}
                        )

                    for k in self.stats.keys():
                        if type(self.stats[k]) is list:
                            del self.stats[k][:]

                    s_lr = (" - LR: ") + " / ".join("{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups)

                    logger.info(s_iter + s_stat + s_lr)

                steps += 1
                if steps >= params.steps_per_epoch:
                    return
