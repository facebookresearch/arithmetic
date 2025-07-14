# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch

import wandb

from src.slurm import init_distributed_mode
from src.utils import bool_flag, initialize_exp, fix_randomness
from src.model import build_model
from src.trainer import Trainer
from src.evaluator import Evaluator
import getpass


def get_train_parser(args=None):
    parser = argparse.ArgumentParser(description="", add_help=False)
    parser.add_argument("--Q", type=int, default=257)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--sample_strategy", type=str, default="1_sqrtx_only_zero")
    parser.add_argument("--S", type=int, default=100_000_000, help="total set size")
    parser.add_argument("--dataset_path", type=str, default=f"/private/home/{getpass.getuser()}/arithmetic/new_datasets")
    parser.add_argument("--positional_emb", type=bool_flag, default=True)
    parser.add_argument("--test_size", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--train_with_cl", type=str, default="all")
    parser.add_argument("--function", type=str, default="linear")
    parser.add_argument("--add_s1", type=bool_flag, default=True, help="do the model need to learn sum x_i mod Q?")
    parser.add_argument("--add_secret", type=bool_flag, default=False, help="do the model need to learn sum x_i \cdot s_i mod Q?")
    parser.add_argument("--hamming_weight", type=int, default=None, help="secret hardness")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--gaussian_error", type=float, default=0.0, help="gaussian error")

    parser.add_argument("--emb_input", type=str, default="angular")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps_per_epoch", type=int, default=10_000)
    parser.add_argument("--amp", type=bool_flag, default=True, help="Use AMP wrapper")

    parser.add_argument("--loss_type", type=str, default="custom;0.1", help="if you want to setup a custom loss, you should try with `custom;eps`")
    parser.add_argument("--layer_norm_pre", type=bool_flag, default=False)
    parser.add_argument("--layer_norm_post", type=bool_flag, default=False)

    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=0, help="Clip gradients norm (0 to disable)")

    parser.add_argument("--backbone", type=str, default="transformer")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_enc_heads", type=int, default=None)
    parser.add_argument("--n_enc_layers", type=int, default=4)

    parser.add_argument("--validation_metrics", type=str, default="test_loss", help="you can choose between `loss` and `acc`")
    parser.add_argument("--wandb", type=bool_flag, default=True)
    parser.add_argument("--wandb_host", type=str, default=None)
    return parser


def secret_handling(params):
    params.num_secrets = 2  # even if not used
    if getattr(params, "add_secret", False):
        assert params.hamming_weight is not None


def main_train(params):
    fix_randomness(params)
    init_distributed_mode(params)
    secret_handling(params)
    logger = initialize_exp(params)
    assert torch.cuda.is_available()

    if params.is_master and params.wandb:
        wandb.login(host=params.wandb_host)
        wandb.init(
            reinit=True,
            config=params,
            resume="allow",
            project="-".join(str(params.output_dir).split("/")[4:6]),
            id=str(params.output_dir).split("/")[-1],
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("test/*", step_metric="train/step")

    model = build_model(params)
    trainer = Trainer(model, params)
    evaluator = Evaluator(trainer)

    starting_epoch = trainer.epoch
    for epoch in range(starting_epoch, params.epochs):
        logger.info(f"============ Starting epoch {trainer.epoch} ... ============")
        trainer.train_epoch()
        logger.info(f"============ End of epoch {trainer.epoch} ============")

        scores = evaluator.run_all_evals()

        if params.is_master:
            if params.wandb:
                wandb.log({"train/step": trainer.n_total_iter} | {"test/" + k.replace("test_", ""): v for k, v in scores.items()})

            logger.info(scores)
            trainer.save_best_model(scores)
            trainer.end_epoch()

    trainer.dataloader.dataset.close_files()

    if params.world_size > 1:
        torch.distributed.barrier()
    exit()
