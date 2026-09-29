"""Complete two-frame heteroscedastic decomposition network."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .blocks import FixedLinearScaler
from .decoders import CMEDecoder, FlatCoronaDecoder, NoiseDecoder, NormalizationDecoder
from .encoder import SharedImageEncoder
from .latent import LatentSeparator
from .outputs import TFHDNOutput
from .temporal_fusion import MultiscaleTemporalFusion


class TwoFrameHeteroscedasticDecompositionNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.architecture_name = config.name
        channels = config.channels
        self.minimum_noise_scale = config.minimum_noise_scale
        self.time_scale_seconds = config.time_scale_seconds
        self.scaler = FixedLinearScaler(config.input_scale)
        self.log_component_limit = config.log_component_limit
        self.encoder = SharedImageEncoder(
            channels,
            config.blocks_per_level,
            config.group_norm_groups,
            config.positional_grid_size,
        )
        self.temporal_fusion = MultiscaleTemporalFusion(channels, config.maximum_offset_pixels)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, channels[-1]),
            nn.SiLU(),
            nn.Linear(channels[-1], channels[-1]),
        )
        self.separator = LatentSeparator(channels[-1])
        self.normalization_decoder = NormalizationDecoder(
            channels[-1],
            len(channels),
            config.group_norm_groups,
            config.normalization_grid_size,
        )
        self.flat_corona_decoder = FlatCoronaDecoder(channels, config.group_norm_groups)
        self.noise_decoder = NoiseDecoder(
            channels[-1], len(channels), config.group_norm_groups, config.noise_grid_size
        )
        self.cme_decoder = CMEDecoder(channels, config.group_norm_groups)

    def forward(
        self,
        images: torch.Tensor,
        delta_t: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        radial_coordinate: torch.Tensor | None = None,
    ) -> TFHDNOutput:
        if images.ndim != 4 or images.shape[1] != 2:
            raise ValueError("images must have shape [B, 2, H, W]")
        if valid_mask is not None:
            images = images.masked_fill(~valid_mask.bool(), 0.0)
        if radial_coordinate is None:
            height, width = images.shape[-2:]
            y = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=images.dtype)
            x = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            radius = (grid_x.square() + grid_y.square()).sqrt()
            radius = radius / radius.max().clamp_min(1.0)
            radial_coordinate = radius.expand(images.shape[0], 2, -1, -1)
        if radial_coordinate.shape != images.shape:
            raise ValueError("radial_coordinate must have shape [B, 2, H, W]")
        radial_coordinate = radial_coordinate.to(device=images.device, dtype=images.dtype)
        scaled = self.scaler(images)
        first = self.encoder(scaled[:, 0:1], radial_coordinate[:, 0:1])
        second = self.encoder(scaled[:, 1:2], radial_coordinate[:, 1:2])
        if delta_t is None:
            delta_t = images.new_zeros(images.shape[0])
        time = self.time_embedding(delta_t.reshape(-1, 1) / self.time_scale_seconds)
        time = time.unsqueeze(-1).unsqueeze(-1)
        first[-1] = first[-1] + time
        second[-1] = second[-1] + time
        fused_first, fused_second = self.temporal_fusion(first, second)
        latent = self.separator(fused_first[-1], fused_second[-1])
        output_size = images.shape[-2:]

        log_normalization = self.normalization_decoder(latent.common, output_size).float()
        log_flat_1 = self.flat_corona_decoder(latent.common, latent.dynamic_1, fused_first)
        log_flat_2 = self.flat_corona_decoder(latent.common, latent.dynamic_2, fused_second)
        log_flat = torch.cat((log_flat_1, log_flat_2), dim=1).float()

        # Fix the scalar B*C ambiguity: C has unit geometric mean in every
        # frame, so the physical amplitude must be carried by B.
        if valid_mask is None:
            gauge_mask = torch.ones_like(log_flat, dtype=torch.bool)
        else:
            gauge_mask = valid_mask.bool()
        gauge_weight = gauge_mask.to(log_flat.dtype)
        gauge_mean = (log_flat * gauge_weight).sum(dim=(-2, -1), keepdim=True) / gauge_weight.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1.0)
        log_flat = log_flat - gauge_mean
        # Preserve B*C exactly while selecting the gauge. This also makes crop
        # training and full-frame inference differ only in component scale,
        # never in the reconstructed physical image.
        log_normalization = log_normalization + gauge_mean

        limit = self.log_component_limit
        log_normalization = log_normalization.clamp(-limit, limit)
        log_flat = log_flat.clamp(-limit, limit)
        normalization_scaled = log_normalization.exp()
        flat = log_flat.exp()
        clean_scaled = normalization_scaled * flat

        scale_1, df_1 = self.noise_decoder(latent.noise_1, output_size)
        scale_2, df_2 = self.noise_decoder(latent.noise_2, output_size)
        # Convert decoder outputs to float32 before returning to physical pB
        # units.  In particular, 1e-8 and 1e-10 underflow in float16.
        noise_scale = self.scaler.inverse(
            torch.cat((scale_1, scale_2), dim=1).float()
        ).clamp_min(
            self.minimum_noise_scale
        )
        noise_df = torch.cat((df_1, df_2), dim=1).float()
        cme = torch.cat(
            (
                self.cme_decoder(latent.common, latent.dynamic_1, fused_first),
                self.cme_decoder(latent.common, latent.dynamic_2, fused_second),
            ),
            dim=1,
        )
        return TFHDNOutput(
            clean_pb=self.scaler.inverse(clean_scaled.float()),
            flat_corona=flat,
            normalization_field=self.scaler.inverse(normalization_scaled.float()),
            noise_location=torch.zeros_like(clean_scaled, dtype=torch.float32),
            noise_scale=noise_scale,
            noise_df=noise_df,
            cme_probability=cme,
            auxiliary={
                "common": latent.common,
                "dynamic_1": latent.dynamic_1,
                "dynamic_2": latent.dynamic_2,
                "noise_1": latent.noise_1,
                "noise_2": latent.noise_2,
                "normalization_scaled": normalization_scaled,
                "log_normalization": log_normalization,
                "log_flat_corona": log_flat,
                "gauge_mean": gauge_mean,
            },
        )
