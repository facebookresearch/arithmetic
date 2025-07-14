# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import argparse
import json

from .logger import create_logger

import torch
from pathlib import Path
import random

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


def cos(x, Q):
    result = torch.where(x < Q, torch.cos(2 * torch.pi * x / Q), torch.zeros_like(x))
    return result


def sin(x, Q):
    result = torch.where(x < Q, torch.sin(2 * torch.pi * x / Q), torch.zeros_like(x))
    return result


def to_xy(x, Q):
    return torch.stack((cos(x, Q), sin(x, Q)), dim=-1)


def to_angle(xy):
    return (torch.atan2(xy[:, 1], xy[:, 0]) + 2 * torch.pi) % (2 * torch.pi)


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.strip().lower() in FALSY_STRINGS:
        return False
    elif s.strip().lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    if params.wandb:
        # dump parameters
        pickle.dump(params, open(os.path.join(params.output_dir, "params.pkl"), "wb"))
    # create a logger
    logger = create_logger(os.path.join(params.output_dir, "train.log"), rank=getattr(params, "rank", 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.output_dir)
    return logger


def fix_randomness(params):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.use_deterministic_algorithms(True, warn_only=False)

    # Enable CuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = random.randint(0, 2**16)
    torch.manual_seed(seed)

    params.seed = seed


def get_grid_params(experiment):
    job_id = str(Path(experiment)).split("/")[-1]
    experiment_file = os.path.join(experiment, f"{job_id}_0_log.out")
    if os.path.exists(experiment_file):
        with open(experiment_file, "r") as open_file:
            for line in open_file:
                if line.find("__grid_params__: ") > -1:
                    return json.loads(line[len("__grid_params__: ") :].strip().replace("'", '"').replace("True", "true").replace("False", "false"))
        return None
    return None


# [bla1, bla2]+ [None, None] =>[blal, bla2]+ [blal, bla2]
def distributed_broadcast(items, src=0):
    torch.distributed.broadcast_object_list(items, src=src)
