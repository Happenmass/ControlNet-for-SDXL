from typing import List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig

from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
import cv2
import einops
import numpy as np

class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, hint, batch, control=None, control_scales = None, only_mid_control=False):

        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        # # save noise for visualization
        # noise_copy = noise.detach().clone()
        # noise_copy = einops.rearrange(noise_copy, 'b c h w -> b h w c')
        # noise_copy = noise_copy.cpu().numpy()
        # noise_copy = noise_copy * 255.0
        # noise_copy = noise_copy.astype(np.uint8)
        # cv2.imwrite("test_noise_origin.png", noise_copy[0])

        # # save input for visualization
        # input_copy = input.detach().clone()
        # input_copy = einops.rearrange(input_copy, 'b c h w -> b h w c')
        # input_copy = input_copy.cpu().numpy()
        # input_copy = input_copy * 255.0
        # input_copy = input_copy.astype(np.uint8)
        # cv2.imwrite("test_input.png", input_copy[0])


        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        # # save noised_input for visualization
        # noised_input_copy = noised_input.detach().clone()
        # noised_input_copy = einops.rearrange(noised_input_copy, 'b c h w -> b h w c')
        # noised_input_copy = noised_input_copy.cpu().numpy()
        # noised_input_copy = noised_input_copy * 255.0
        # noised_input_copy = noised_input_copy.astype(np.uint8)
        # cv2.imwrite("test_noise.png", noised_input_copy[0])
        model_output = denoiser(
            network, noised_input, sigmas, cond, hint, control, control_scales, only_mid_control, **additional_model_inputs
        )
        # # save model_output for visualization
        # samples_copy = model_output.detach().clone()
        # samples_copy = einops.rearrange(samples_copy, 'b c h w -> b h w c')
        # samples_copy = samples_copy.cpu().numpy()
        # samples_copy = samples_copy * 255.0
        # samples_copy = samples_copy.astype(np.uint8)
        # cv2.imwrite("test_out.png", samples_copy[0])

        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
