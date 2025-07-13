import sys
import os

import numpy as np

import torch
import torchvision.transforms as transforms

import copy
from typing import Optional, List

from model import Generator
from src.loss import ClipLoss


class StyleGan2Generator(torch.nn.Module):
    def __init__(self,
                 weights_path: str,
                 latent_dim: int = 512,
                 n_mlp: int = 8,
                 size: int = 256,
                 channel_multiplier: int = 2,
                 device: str = 'cuda:0'
                 ) -> None:
        super(StyleGan2Generator, self).__init__()

        # load stylegan2 generator
        self.generator = Generator(size, latent_dim, n_mlp, channel_multiplier=channel_multiplier).to(device)
        weights = torch.load(weights_path, map_location=device)
        self.generator.load_state_dict(weights["g_ema"], strict=True)
        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def get_all_layers(self) -> List:
        return list(self.generator.children())

    def get_training_layers(self,
                            change_type: Optional[str]
                            ) -> List:

        # for this task, I leave texture layers or all layers
        if change_type == 'texture':
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])
        else:
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])

    def freeze_layers(self,
                      layer_list: Optional[List] = None
                      ) -> None:
        if not layer_list:
            layer_list = self.get_all_layers()
        for layer in layer_list:
            self.requires_gradient(layer, False)

    def unfreeze_layers(self,
                        layer_list: Optional[List] = None
                        ) -> None:
        if not layer_list:
            layer_list = self.get_all_layers()
        for layer in layer_list:
            self.requires_gradient(layer, True)

    @staticmethod
    def requires_gradient(model, mode=True):
        for p in model.parameters():
            p.requires_grad = mode

    def get_style(self,
              styles: List
              ) -> List:
        styles = [self.generator.style(s) for s in styles]
        return styles

    def forward(self,
                styles: torch.tensor,
                return_latents: bool = False,
                truncation: float = 1.,
                truncation_latent: Optional[torch.tensor] = None,
                input_is_latent: bool = False,
                noise: Optional[torch.tensor] = None,
                randomize_noise: bool = True
                ) -> torch.tensor:
        # just apply generator
        return self.generator(styles, return_latents=return_latents, truncation=truncation, truncation_latent=self.mean_latent, noise=noise, randomize_noise=randomize_noise, input_is_latent=input_is_latent)


class Net(torch.nn.Module):
    def __init__(self,
                 args
                 ) -> None:
        super(Net, self).__init__()

        self.args = args
        self.device = args.device

        self.generator_frozen = StyleGan2Generator(args.weights_path, size=args.size).to(self.device)
        if not args.trained_weights_path:
            self.generator_trainable = StyleGan2Generator(args.weights_path, size=args.size).to(self.device)
        else:
            self.generator_trainable = StyleGan2Generator(args.trained_weights_path, size=args.size).to(self.device)
            
        # freeze trainer model
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()

        # unfreeze the layers which weights we are going to optimize
        self.generator_trainable.freeze_layers()
        self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.change_type))
        self.generator_trainable.train()

        # directional clip loss
        self.clip_loss = ClipLoss(device=self.device,
                                  direction_lambda=args.direction_lambda,
                                  global_lambda=args.global_lambda,
                                  clip_model=args.clip_model_name
                                  )

        # source and target
        self.src_class = args.src_class
        self.tgt_class = args.tgt_class

    def forward(
        self,
        styles: torch.tensor,
        return_latents: bool = False,
        truncation: float = 1.,
        truncation_latent: Optional[torch.tensor] = None,
        input_is_latent: bool = False,
        noise: Optional[torch.tensor] = None,
        randomize_noise: bool = True,
    ):

        # convert to w
        with torch.no_grad():
            if input_is_latent:
                w_styles = styles
            else:
                w_styles = self.generator_frozen.get_style(styles)

            # generate orig image
            orig_image = self.generator_frozen(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]

        # generate conditioned image
        edited_image = self.generator_trainable(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]

        # calculate directional loss
        cliploss = torch.sum(self.clip_loss(orig_image, self.src_class, edited_image, self.tgt_class))
        return orig_image, edited_image, cliploss
        