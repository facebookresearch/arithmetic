# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import random
import traceback
import time
import hashlib
import subprocess
import datetime


def get_random_port(s):
    hashed = hashlib.sha256(s.encode())
    seed = int(hashed.hexdigest(), 16) % (2**32 - 1)
    random.seed(seed)
    return random.randint(10001, 20000)


def init_distributed_mode(params):
    params.is_slurm_job = "SLURM_JOB_ID" in os.environ
    print("SLURM job: %s" % str(params.is_slurm_job))

    # SLURM job
    if params.is_slurm_job:
        params.rank = int(os.environ["SLURM_PROCID"])
        params.local_rank = int(os.environ["SLURM_LOCALID"])
        params.world_size = int(os.environ["SLURM_NTASKS"])
        params.n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        params.node_id = int(os.environ["SLURM_NODEID"])

        os.environ["RANK"] = str(params.rank)
        os.environ["LOCAL_RANK"] = str(params.local_rank)
        os.environ["WORLD_SIZE"] = str(params.world_size)

        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        params.master_addr = hostnames.split()[0].decode("utf-8")
        os.environ["MASTER_ADDR"] = params.master_addr

        torch.cuda.set_device(params.local_rank)

        if "SLURM_ARRAY_JOB_ID" in os.environ:
            jobname = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
        elif "SLURM_JOB_ID" in os.environ:
            jobname = str(os.environ["SLURM_JOB_ID"])
        else:
            assert False, "missing slurm job ids"

        max_retries = 5
        retry_delay = 5

        for i in range(max_retries):
            try:
                params.dist_url = "tcp://%s:%s" % (os.environ["MASTER_ADDR"], get_random_port(f"{jobname} = {i}"))
                torch.distributed.init_process_group(
                    backend="nccl",
                    init_method=params.dist_url,
                    world_size=params.world_size,
                    rank=params.rank,
                    timeout=datetime.timedelta(seconds=5400),
                )
                print(f"Begin torch.dist.init with init_method={params.dist_url}")
                break
            except Exception as e:
                print(f"torch.distributed.init_process_group failed. Attempt {i+1} / {max_retries}")
                print(f"Error: {str(e)}")
                traceback.print_exc()
                if i < max_retries - 1:
                    print(f"Retry in {retry_delay} seconds")
                    time.sleep(retry_delay)
                else:
                    print("Max retries exceeded. Upload failed")
                    raise

        torch.distributed.barrier()
    else:
        local_rank = 0
        params.rank = local_rank
        params.local_rank = local_rank
        params.world_size = 1
        params.n_nodes = 1
        params.node_id = 0
        torch.cuda.set_device(local_rank)

    params.is_master = params.node_id == 0 and params.local_rank == 0

    print("myrank: ", params.rank, "local_rank: ", params.local_rank, " device_count: ", torch.cuda.device_count(), "world_size:", params.world_size)
