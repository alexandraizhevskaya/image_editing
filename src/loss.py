import numpy as np
import math

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import clip
from PIL import Image

from typing import Union, List


class ClipLoss(torch.nn.Module):
    def __init__(self,
                 device: str,
                 direction_lambda: int = 1.,
                 global_lambda: int = 0.,
                 clip_model: str = 'ViT-B/32'
                 ) -> None:
        super(ClipLoss, self).__init__()

        # load model
        self.device = device
        self.model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.model.requires_grad_(False)

        # set lambdas
        self.global_lambda = global_lambda
        self.direction_lambda = direction_lambda

        # set target direction
        self.target_direction = None

        # set img transformations
        self.preprocess_image = transforms.Compose(
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +
            self.clip_preprocess.transforms[:2] +
            self.clip_preprocess.transforms[4:]
            )

    def encode_text(self,
                    strings: str
                    ) -> torch.Tensor:
        tokens = clip.tokenize(strings).to(self.device)
        return self.model.encode_text(tokens)

    def encode_image(self,
                     images: torch.Tensor
                     ) -> torch.Tensor:
        images = self.preprocess_image(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_feats(self,
                       class_str: str,
                       normalize: bool = True
                       ) -> torch.Tensor:
        text_feats = self.encode_text([class_str]).detach()
        if normalize:
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
        return text_feats

    def get_image_feats(self,
                           img: torch.Tensor,
                           normalize: bool = True
                           ) -> torch.Tensor:
        image_feats = self.encode_image(img)
        if normalize:
            image_feats /= image_feats.clone().norm(dim=-1, keepdim=True)
        return image_feats

    def compute_text_direction(self,
                               src_class: str,
                               tgt_class: str
                               ) -> torch.Tensor:
        src_feats = self.get_text_feats(src_class)
        tgt_feats = self.get_text_feats(tgt_class)
        text_direction = (tgt_feats - src_feats).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        return text_direction

    def compute_image_direction(self,
                                src_img: torch.Tensor,
                                tgt_img: torch.Tensor
                                ) -> torch.Tensor:
        src_encoding = self.get_image_feats(src_img)
        tgt_encoding = self.get_image_feats(tgt_img)
        image_edit_direction = (tgt_encoding - src_encoding)
        if image_edit_direction.sum() == 0:
            tgt_encoding = self.get_image_feats(tgt_img + 1e-6)
            image_edit_direction = (tgt_encoding - src_encoding)
        image_edit_direction /= (image_edit_direction.clone().norm(dim=-1, keepdim=True))
        return image_edit_direction

    def clip_directional_loss(self,
                              src_img: torch.Tensor,
                              src_class: str,
                              tgt_img: torch.Tensor,
                              tgt_class: str
                              ) -> torch.Tensor:
        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(src_class, tgt_class)
        edit_direction = self.compute_image_direction(src_img, tgt_img)
        return (1. - torch.nn.CosineSimilarity()(edit_direction, self.target_direction)).mean()

    def global_clip_loss(self,
                         img: torch.Tensor,
                         text: Union[str, List]
                         ) -> torch.Tensor:

        if not isinstance(text, list):
            text = [text]
        tokens = clip.tokenize(text).to(self.device)
        image  = self.preprocess_image(img)
        clip_logits, _ = self.model(image, tokens)
        return (1. - clip_logits).mean()

    def forward(self,
                src_img: torch.Tensor,
                src_class: str,
                tgt_img: torch.Tensor,
                tgt_class: str,
                ):
        clip_loss = 0.0
        if self.global_lambda:
            clip_loss += self.global_lambda * self.global_clip_loss(tgt_img, [tgt_class])
        if self.direction_lambda:
            clip_loss += self.direction_lambda * self.clip_directional_loss(src_img, src_class, tgt_img, tgt_class)
        return clip_loss
        