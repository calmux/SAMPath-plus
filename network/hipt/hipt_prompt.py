
import os
import math
from functools import partial, reduce
from operator import mul

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _pair

from . import vision_transformer as vits
from . import vision_transformer4k as vits4k


class PromptedTransformer(vits.VisionTransformer):
    def __init__(
            self,
            vit_config,
            num_tokens=1,
            drop_out=0.,
            project_prompt_dim=-1,
            deep_prompt=False,
    ):
        super().__init__(**vit_config)
        self.vit_config = vit_config

        self.num_prefix_tokens = 1

        patch_size = _pair(vit_config["patch_size"])

        self.num_prompt_tokens = num_tokens  # number of prompted tokens
        self.deep_prompt = deep_prompt

        self.prompt_dropout = nn.Dropout(drop_out)

        # if project the prompt embeddings
        if project_prompt_dim > 0:
            # only for prepend / add
            prompt_dim = project_prompt_dim
            self.prompt_proj = nn.Linear(
                prompt_dim, vit_config["embed_dim"])
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = vit_config["embed_dim"]
            self.prompt_proj = nn.Identity()

        if num_tokens > 0:
            # initiate prompt:
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            pass

        if self.deep_prompt:  # noqa
            total_d_layer = vit_config["depth"] - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                total_d_layer, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.prepare_tokens(x)

        prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
        x = torch.cat((
            x[:, :self.num_prefix_tokens, :],
            prompt,
            x[:, self.num_prefix_tokens:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        # x = self.norm_pre(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            if self.num_prompt_tokens > 0:
                super().train(False)
                self.prompt_proj.train()
                self.prompt_dropout.train()
            else:
                super().train(mode)
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        # attn_weights = []
        hidden_states = None
        # weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config["depth"]

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.blocks[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]: