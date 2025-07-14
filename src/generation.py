# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import h5py
import os


def generate_and_write(N, Q, sample_strategy, size, seed, file_name):
    rng = np.random.default_rng(seed)
    if sample_strategy.startswith("1_sqrtx") or sample_strategy.startswith("min_1"):
        list_to_choose = range(1, N + 1)
        if sample_strategy.startswith("1_sqrtx"):
            p = [1.0 / (N - n + 1) ** 0.5 for n in list_to_choose]
            p = [1.0 * x / sum(p) for x in p]
        elif sample_strategy.startswith("min_1"):
            p = [1.0 for n in list_to_choose]
            p = [1.0 * x / sum(p) for x in p]
        ns = rng.choice(list_to_choose, p=p, size=size)
        X = np.zeros((size, N), dtype=np.int32)
        for length in list_to_choose:
            X[ns == length, :length] = rng.integers(1, Q, size=((ns == length).sum(), length))
        shuffled_size = np.arange(size)
        rng.shuffle(shuffled_size)
        X = X[shuffled_size]
        random_indices = np.argsort(rng.random((size, N)), axis=1)
        X = np.take_along_axis(X, random_indices, axis=1)
    elif sample_strategy == "uniform":
        X = rng.integers(0, Q, size=(size, N))
        X = np.int32(X)
    easy_indices = np.where((X > 0).sum(axis=1) <= (N // 2))[0]
    hard_indices = np.where((X > 0).sum(axis=1) > (N // 2))[0]
    all_indices = np.arange(size)
    with h5py.File(file_name, "w") as h5f:
        h5f.create_dataset("X", data=X)
        h5f.create_dataset("easy_indices", data=easy_indices)
        h5f.create_dataset("hard_indices", data=hard_indices)
        h5f.create_dataset("all_indices", data=all_indices)
    print(file_name)


if __name__ == "__main__":
    for N in [4]:
        for Q in [257]:
            for sample_strategy in ["1_sqrtx_only_zero", "min_1_only_zero", "uniform"]:
                if sample_strategy == "uniform":
                    seeds = [0, 1]
                else:
                    seeds = [0]
                for seed in seeds:
                    size = 100_000 if seed > 0 else 100_000_000
                    file_name = f"new_datasets/N={N}|Q={Q}|sample_strategy={sample_strategy}|size={size}|seed={seed}.h5"
                    if os.path.isfile(file_name):
                        print(f"{file_name} exists already")
                    else:
                        generate_and_write(N, Q, sample_strategy, size, seed, file_name)
