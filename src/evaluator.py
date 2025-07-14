# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
import torch
from .dataset import data_gen
from .utils import to_angle, to_xy
import numpy as np


logger = getLogger()


def mod_diff(f_x, f_x_i, Q):
    assert f_x.shape == f_x_i.shape
    diff = torch.abs(f_x - f_x_i)
    diff = torch.minimum(diff, Q - diff)
    return diff.sum().cpu()


class Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.params = trainer.params

    def breakdown(self, metrics_list):
        stats = {}
        for el in metrics_list:
            train_eval_test_type = el["train_eval_test_type"]
            s_k = el["s_k"]

            for key in [
                "loss",
                "acc",
                "relative_dist_0_001Q",
                "relative_dist_0_002Q",
                "relative_dist_0_003Q",
                "relative_dist_0_005Q",
                "relative_dist_0_01Q",
                "relative_dist_0_02Q",
                "relative_dist_0_03Q",
                "relative_dist_0_05Q",
                "relative_dist_0_1Q",
            ]:
                metric = "_" + key
                for comb in ["", f"_secretkey_{s_k}"]:
                    stats[f"{train_eval_test_type}{metric}{comb}"] = stats.get(f"{train_eval_test_type}{metric}{comb}", 0) + el[key]
                    stats[f"COUNT_{train_eval_test_type}{metric}{comb}"] = stats.get(f"COUNT_{train_eval_test_type}{metric}{comb}", 0) + 1

        final_stats = {}
        for key in stats.keys():
            if not key.startswith("COUNT_"):
                if stats.get(f"COUNT_{key}", 0) == 0:
                    final_stats[key] = 0.0
                else:
                    final_stats[key] = 1.0 * stats[key] / stats[f"COUNT_{key}"]
        return final_stats

    def sort_embedded_numbers(self, scores):
        def sorting_key(s):
            parts = s.split("_")
            key = []
            for part in parts:
                if part.isdigit():
                    key.append(part.zfill(100))
                else:
                    key.append(part)
            # Add a length key to prioritize shorter strings
            key = (len([p for p in parts if p.isdigit()]),) + tuple(key)
            return key

        return dict(sorted(scores.items(), key=lambda item: sorting_key(item[0])))

    def run_all_evals(self):
        with torch.no_grad():
            metrics = self.evaluate("test")
        metrics_list = [None for _ in range(self.params.world_size)]
        if self.params.world_size > 1:
            torch.distributed.all_gather_object(metrics_list, metrics)
        else:
            metrics_list = [metrics]
        metrics_list = [item for sublist in metrics_list for item in sublist]
        scores = {"epoch": self.trainer.epoch} | self.breakdown(metrics_list)
        if getattr(self.params, "add_secret", False):
            scores["recovered"] = self.recover()
        scores = self.sort_embedded_numbers(scores)
        return scores

    def evaluate(self, train_eval_test_type):
        params = self.params
        model = self.model.module if params.world_size > 1 else self.model
        scores = []

        model.eval()

        iterator = data_gen(params, train_eval_test_type=train_eval_test_type)
        for x, s_k, y in iterator:
            x, s_k, y = x.cuda(), s_k.cuda(), y.cuda()
            output = model(x, s_k)["output"]
            if params.loss_type == "CE":
                loss = torch.nn.functional.cross_entropy(output, y, reduction="none")
                pred_int = output.argmax(dim=-1)
            else:
                loss = torch.nn.functional.mse_loss(output, to_xy(y, params.Q), reduction="none").sum(dim=1)
                pred_int = (torch.round(to_angle(output) * params.Q / (2 * torch.pi))) % params.Q
            acc = (pred_int == y).long()
            acc_relative_dist_0_001Q = (torch.abs(pred_int - y) % params.Q <= 0.001 * params.Q).long()
            acc_relative_dist_0_002Q = (torch.abs(pred_int - y) % params.Q <= 0.002 * params.Q).long()
            acc_relative_dist_0_003Q = (torch.abs(pred_int - y) % params.Q <= 0.003 * params.Q).long()
            acc_relative_dist_0_005Q = (torch.abs(pred_int - y) % params.Q <= 0.005 * params.Q).long()
            acc_relative_dist_0_01Q = (torch.abs(pred_int - y) % params.Q <= 0.01 * params.Q).long()
            acc_relative_dist_0_02Q = (torch.abs(pred_int - y) % params.Q <= 0.02 * params.Q).long()
            acc_relative_dist_0_03Q = (torch.abs(pred_int - y) % params.Q <= 0.03 * params.Q).long()
            acc_relative_dist_0_05Q = (torch.abs(pred_int - y) % params.Q <= 0.05 * params.Q).long()
            acc_relative_dist_0_1Q = (torch.abs(pred_int - y) % params.Q <= 0.1 * params.Q).long()

            for j in range(x.shape[0]):
                score = {
                    "train_eval_test_type": train_eval_test_type,
                    "loss": loss[j].item(),
                    "acc": acc[j].item(),
                    "relative_dist_0_001Q": acc_relative_dist_0_001Q[j].item(),
                    "relative_dist_0_002Q": acc_relative_dist_0_002Q[j].item(),
                    "relative_dist_0_003Q": acc_relative_dist_0_003Q[j].item(),
                    "relative_dist_0_005Q": acc_relative_dist_0_005Q[j].item(),
                    "relative_dist_0_01Q": acc_relative_dist_0_01Q[j].item(),
                    "relative_dist_0_02Q": acc_relative_dist_0_02Q[j].item(),
                    "relative_dist_0_03Q": acc_relative_dist_0_03Q[j].item(),
                    "relative_dist_0_05Q": acc_relative_dist_0_05Q[j].item(),
                    "relative_dist_0_1Q": acc_relative_dist_0_1Q[j].item(),
                    "s_k": s_k[j].item(),
                }
                scores.append(score)

        iterator.dataset.close_files()

        return scores

    def get_inputs(self, x):
        for i in range(self.params.N):
            xi = x.clone()
            xi[:, i] += self.params.Q // 2
            xi[:, i] %= self.params.Q
            yield xi

    def inference(self, x, s_k, y=None):
        # Get integer preds and loss from model
        self.model.eval()
        output = self.model(x, s_k)["output"]
        loss = None
        if self.params.loss_type == "CE":
            preds = output.argmax(dim=-1)
            if y is not None:
                loss = torch.nn.functional.cross_entropy(output, y, reduction="none")
        else:
            preds = (torch.round(to_angle(output) * self.params.Q / (2 * torch.pi))) % self.params.Q
            if y is not None:
                loss = torch.nn.functional.mse_loss(output, to_xy(y, self.params.Q), reduction="none").sum(dim=1)
        return preds, loss

    def std_secret(self, guess):
        # Calculate std of a secret guess
        guess = np.array(guess).astype(int)
        err_pred_all = []
        for x, _, y in data_gen(self.params, train_eval_test_type="test"):
            x = x.numpy()
            y = y.numpy()
            err_pred = (x @ guess - y) % self.params.Q
            err_pred[err_pred > self.params.Q // 2] -= self.params.Q
            err_pred_all.extend(err_pred)
        return np.std(err_pred_all).item()

    def compute_outputs(self, x, s_k, f_x, f_xi):
        current_f_x, _ = self.inference(x, s_k)
        current_f_xi = []
        for xi in self.get_inputs(x):
            preds, _ = self.inference(xi, s_k)
            current_f_xi.append(preds)

        if f_x is None:
            f_x = current_f_x
        else:
            f_x = torch.cat([f_x, current_f_x], dim=0)
        if f_xi is None:
            f_xi = current_f_xi
        else:
            for i in range(len(f_xi)):
                f_xi[i] = torch.cat([f_xi[i], current_f_xi[i]], dim=0)
        return f_x, f_xi

    @torch.no_grad()
    def dist_run(self, f_x, f_xi):
        scores = [mod_diff(f_x, f_x_i, Q=self.params.Q) for f_x_i in f_xi]
        sorted_i_score = sorted(enumerate(scores), key=lambda i: i[1], reverse=True)
        logger.info(f"Scores: {[(ind, score.item()) for ind, score in sorted_i_score]}")
        # USING SECRET HAMMING HERE!
        sorted_i_score = sorted_i_score[: self.params.hamming_weight]

        guess = np.zeros(self.params.N)
        for i, _ in sorted_i_score:
            guess[i] = 1

        std_error = self.std_secret(guess)

        if std_error < 2 * getattr(self.params, "sigma", 3.0):
            logger.info(f"Successful Secret Guess: {np.where(guess)}")
            logger.info(f"Std error is: {round(std_error, 3)}")
            return True
        else:
            return False

    @torch.no_grad()
    def recover(self):
        f_x, f_xi = None, None
        for x, s_k, _ in data_gen(self.params, train_eval_test_type="test"):
            x, s_k = x.cuda(), s_k.cuda()
            f_x, f_xi = self.compute_outputs(x, s_k, f_x, f_xi)
        return self.dist_run(f_x, f_xi)
