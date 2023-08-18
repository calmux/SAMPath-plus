import copy
from functools import reduce
from operator import mul
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything.modeling.common import LayerNorm2d


class PromptSAM(nn.Module):

    def __init__(
            self,
            model_type: str = "vit_b",
            checkpoint: str = "",
            prompt_dim: int = 256,
            num_classes: int = 20,
            extra_encoder = None,
            freeze_image_encoder = True,
            freeze_prompt_encoder = True,
            freeze_mask_decoder = False,
            mask_HW = (1024, 1024),
            feature_input = False,
            prompt_decoder = False,
            dense_prompt_decoder=False,
            no_sam=False
    ):
        super().__init__()

        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.mask_HW = mask_HW
        self.feature_input = feature_input

        self.extra_encoder = extra_encoder
        # change prompt
        mask_tokens = nn.Embedding(num_classes + 1, prompt_dim)
        self.model.mask_decoder.mask_tokens = mask_tokens
        self.model.mask_decoder.num_mask_tokens = num_classes + 1

        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList(
            [
                # self.model.mask_decoder.output_hypernetworks_mlps[0].clone()
                copy.deepcopy(self.model.mask_decoder.output_hypernetworks_mlps[0])
                for i in range(self.model.mask_decoder.num_mask_tokens)
            ]
        )

        self.model.mask_decoder.iou_prediction_head.layers[-1] = nn.Linear(prompt_dim,
                                         