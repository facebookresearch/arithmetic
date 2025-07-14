# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import math
import h5py
import os


class ModDataset(Dataset):
    def __init__(self, params, size, seed, train_eval_test_type, epoch=0):
        self.params = params
        self.size = size
        self.seed = seed
        self.is_train = train_eval_test_type == "train"
        self.epoch = epoch
        if not self.is_train:
            self.sampled = 0
        if getattr(self.params, "train_with_cl", "all") != "all":
            self.easy_hard = "easy"
        else:
            self.easy_hard = "all"
        self.open_files(train_eval_test_type)
        self.build_secret()
        self.build_error()

    def open_files(self, train_eval_test_type):
        if train_eval_test_type == "train":
            sample_strategies = [self.params.sample_strategy, "uniform"]
            size = 100_000_000
        elif train_eval_test_type == "test":
            sample_strategies = ["uniform", "uniform"]
            size = 100_000
        self.h5f = {}
        for idx, sample_strategy in enumerate(sample_strategies):
            file_name = (
                f"{self.params.dataset_path}/N={self.params.N}|Q={self.params.Q}|sample_strategy={sample_strategy}|size={size}|seed={self.seed}.h5"
            )
            assert os.path.isfile(file_name), file_name
            self.h5f[idx] = h5py.File(file_name, "r")
        self.h5f_indices_shape = {key: self.h5f[0][f"{key}_indices"][:] for key in ["all", "easy", "hard"]}

    def close_files(self):
        for v in self.h5f.values():
            v.close()

    def build_secret(self):
        rng = np.random.default_rng(0)
        if getattr(self.params, "add_secret", False):
            indices = np.arange(self.params.N)
            rng.shuffle(indices)
            indices = indices[: self.params.hamming_weight]
            self.secret = np.int32(np.zeros(self.params.N))
            self.secret[indices] = 1
        else:
            self.secret = np.int32(np.zeros(self.params.N))

    def build_error(self):
        rng = np.random.default_rng(0)
        if self.is_train:
            size = 100_000_000
        else:
            size = 100_000
        self.error = np.int32(rng.normal(scale=getattr(self.params, "gaussian_error", 0.0), size=size).round())

    def __len__(self):
        return math.ceil(self.size / self.params.batch_size)

    def collate_fn(self, elements):
        x, s_k, y = zip(*elements)
        x = torch.LongTensor(np.array(x))[0]
        s_k = torch.LongTensor(np.array(s_k)).T
        y = torch.LongTensor(np.array(y))[0]
        return x, s_k, y

    def process(self, indices, secret_keys, batch_size):
        X = np.empty(shape=(batch_size, self.params.N), dtype=np.int32)
        y = np.empty(shape=(batch_size,), dtype=np.int32)

        X[secret_keys == 0] = self.h5f[0]["X"][indices[secret_keys == 0], :]
        X[secret_keys == 1] = self.h5f[1]["X"][indices[secret_keys == 1], :]

        y[secret_keys == 1] = np.int32(np.sum((X[secret_keys == 1] * self.secret), axis=1) % self.params.Q)
        if getattr(self.params, "function", "linear") == "linear":
            y[secret_keys == 0] = np.int32(np.sum(X[secret_keys == 0], axis=1) % self.params.Q)
        elif getattr(self.params, "function", "linear") == "linear_10":
            y[secret_keys == 0] = np.int32(10 * np.sum(X[secret_keys == 0], axis=1) % self.params.Q)
        elif getattr(self.params, "function", "linear") == "square":
            y[secret_keys == 0] = np.int32(np.sum(X[secret_keys == 0] ** 2, axis=1) % self.params.Q)
        elif getattr(self.params, "function", "linear") == "all_square":
            y[secret_keys == 0] = np.int32(np.sum(X[secret_keys == 0], axis=1) ** 2 % self.params.Q)
        y += self.error[indices]
        return X, y

    def __getitem__(self, index):
        if self.is_train:
            batch_size = self.params.batch_size
        else:
            batch_size = min(self.size - self.sampled, self.params.batch_size)
        if self.is_train:
            rng_for_the_train_index = np.random.default_rng([self.params.rank, self.epoch, index])
            indices = np.sort(
                rng_for_the_train_index.choice(
                    self.params.S * self.h5f_indices_shape[self.easy_hard].shape[0] // 100_000_000, size=batch_size, replace=False
                )
            )
            indices = self.h5f_indices_shape[self.easy_hard][indices]
        else:
            indices = np.array([i for i in range(self.sampled, self.sampled + batch_size)])
            self.sampled += batch_size

        if getattr(self.params, "add_s1", True) and not getattr(self.params, "add_secret", False):
            secret_keys = np.zeros_like(indices)
        elif not getattr(self.params, "add_s1", True) and getattr(self.params, "add_secret", False):
            secret_keys = np.ones_like(indices)

        X, y = self.process(indices, secret_keys, batch_size)
        if self.is_train:
            shuffled_indices = np.arange(batch_size)
            rng_for_the_train_index.shuffle(shuffled_indices)
            X, y = X[shuffled_indices], y[shuffled_indices]
        return X, secret_keys, y


def data_gen(params, train_eval_test_type):
    if train_eval_test_type == "train":
        size = params.batch_size * params.steps_per_epoch * params.epochs
        seed = 0
    elif train_eval_test_type == "test":
        size = params.test_size
        seed = 1
    dataset = ModDataset(params=params, size=size, seed=seed, train_eval_test_type=train_eval_test_type)
    return DataLoader(dataset, batch_size=1, shuffle=False, timeout=0, pin_memory=True, collate_fn=dataset.collate_fn)
