# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import functional as F
from ..utils import to_xy
from logging import getLogger


logger = getLogger()


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_enc_heads, attention_dropout, mlp_dropout):
        super().__init__()
        assert hidden_dim % n_enc_heads == 0
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.resid_dropout = nn.Dropout(mlp_dropout)
        self.n_enc_heads = n_enc_heads
        self.hidden_dim = hidden_dim
        self.attention_dropout = attention_dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, T, self.n_enc_heads, C // self.n_enc_heads).transpose(1, 2)
        q = q.view(B, T, self.n_enc_heads, C // self.n_enc_heads).transpose(1, 2)
        v = v.view(B, T, self.n_enc_heads, C // self.n_enc_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attention_dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, dim, factor=4, mlp_dropout=0.0):
        super().__init__()
        inner_dim = int(factor * dim)
        self.c_fc = nn.Linear(dim, inner_dim, bias=False)
        self.c_proj = nn.Linear(inner_dim, dim, bias=False)
        self.mlp_dropout = nn.Dropout(mlp_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.mlp_dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, hidden_dim, n_enc_heads, attention_dropout, mlp_dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, bias=False)
        self.attn = SelfAttention(hidden_dim, n_enc_heads, attention_dropout, mlp_dropout)
        self.ln_2 = nn.LayerNorm(hidden_dim, bias=False)
        self.mlp = MLP(hidden_dim, mlp_dropout=mlp_dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        hidden_dim = params.hidden_dim
        n_enc_layers = params.n_enc_layers
        n_enc_heads = params.n_enc_heads
        attention_dropout = 0.0
        mlp_dropout = 0.0
        self.Q = params.Q
        self.N = params.N
        self.positional_emb = params.positional_emb
        if n_enc_heads is None:
            assert hidden_dim % 64 == 0
            n_enc_heads = hidden_dim // 64

        self.emb_input = params.emb_input
        self.emb_output = "token" if params.loss_type == "CE" else "angular"

        if self.emb_input == "angular":
            self.emb = nn.Linear(2, hidden_dim)
        else:
            self.emb = nn.Embedding(params.Q, hidden_dim)
        if self.positional_emb:
            self.pos_emb = nn.Embedding(params.N, hidden_dim)
        self.secret_emb = nn.Embedding(params.num_secrets, hidden_dim)
        self.layers = nn.ModuleList([Block(hidden_dim, n_enc_heads, attention_dropout, mlp_dropout) for _ in range(n_enc_layers)])
        if params.layer_norm_pre:
            self.norm = nn.LayerNorm(hidden_dim, bias=False)
        self.layer_norm_pre = params.layer_norm_pre
        if self.emb_output == "angular":
            if params.layer_norm_post:
                self.ln_f = nn.LayerNorm(2, bias=False)
            self.layer_norm_post = params.layer_norm_post
            self.head = nn.Linear(hidden_dim, 2, bias=False)
        else:
            self.head = nn.Linear(hidden_dim * (self.N + 1), self.Q, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_enc_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, secret_keys):
        if self.emb_input == "angular":
            x = to_xy(inputs, self.Q)
            x = self.emb(x)
        elif self.emb_input == "token":
            x = self.emb(inputs)

        if self.positional_emb:
            input_shape = inputs.size()
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=inputs.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            x += self.pos_emb(position_ids)

        x = torch.cat([x, self.secret_emb(secret_keys)], dim=1)

        for layer in self.layers:
            x = layer(x)
        if self.layer_norm_pre:
            x = self.norm(x)
        if self.emb_output == "angular":
            pooled = torch.max(x, dim=1)[0]
            logits = self.head(pooled)
            if self.layer_norm_post:
                logits = self.ln_f(logits)
        else:
            logits = self.head(x.reshape(-1, x.shape[1] * x.shape[2]))
        return {"output": logits}


def model_selection(params):
    if params.backbone == "transformer":
        model = Transformer(params)
    model = torch.compile(model)
    return model
