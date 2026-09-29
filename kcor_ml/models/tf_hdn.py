"""Complete two-frame heteroscedastic decomposition network."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .blocks import LearnedLinearScaler
from .decoders import CMEDecoder, FlatCoronaDecoder, NoiseDecoder, NormalizationDecoder
from .encoder import SharedImageEncoder
from .latent import LatentSeparator
from .outputs import TFHDNOutput
from .temporal_fusion import MultiscaleTemporalFusion


class TwoFrameHeteroscedasticDecompositionNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        channels = config.channels
        self.minimum_noise_scale = config.minimum_noise_scale
        self.time_scale_seconds = config.time_scale_seconds
        self.scaler = LearnedLinearScaler(config.input_scale)
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
            channels[-1], len(channels), config.group_norm_groups
        )
        self.flat_corona_decoder = FlatCoronaDecoder(channels, config.group_norm_groups)
        self.noise_decoder = NoiseDecoder(channels[-1], len(channels), config.group_norm_groups)
        self.cme_decoder = CMEDecoder(channels, config.group_norm_groups)

    def forward(
        self,
        images: torch.Tensor,
        delta_t: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> TFHDNOutput:
        if images.ndim != 4 or images.shape[1] != 2:
            raise ValueError("images must have shape [B, 2, H, W]")
        if valid_mask is not None:
            images = images.masked_fill(~valid_mask.bool(), 0.0)
        scaled = self.scaler(images)
        first = self.encoder(scaled[:, 0:1])
        second = self.encoder(scaled[:, 1:2])
        if delta_t is None:
            delta_t = images.new_zeros(images.shape[0])
        time = self.time_embedding(delta_t.reshape(-1, 1) / self.time_scale_seconds)
        time = time.unsqueeze(-1).unsqueeze(-1)
        first[-1] = first[-1] + time
        second[-1] = second[-1] + time
        fused_first, fused_second = self.temporal_fusion(first, second)
        latent = self.separator(fused_first[-1], fused_second[-1])
        output_size = images.shape[-2:]

        normalization_scaled = self.normalization_decoder(latent.common, output_size)
        flat_1 = self.flat_corona_decoder(latent.common, latent.dynamic_1, fused_first)
        flat_2 = self.flat_corona_decoder(latent.common, latent.dynamic_2, fused_second)
        flat = torch.cat((flat_1, flat_2), dim=1)
        clean_scaled = normalization_scaled * flat

        location_1, scale_1, df_1 = self.noise_decoder(latent.noise_1, output_size)
        location_2, scale_2, df_2 = self.noise_decoder(latent.noise_2, output_size)
        noise_location = self.scaler.inverse(torch.cat((location_1, location_2), dim=1))
        noise_scale = self.scaler.inverse(torch.cat((scale_1, scale_2), dim=1)).clamp_min(
            self.minimum_noise_scale
        )
        noise_df = torch.cat((df_1, df_2), dim=1)
        cme = torch.cat(
            (
                self.cme_decoder(latent.common, latent.dynamic_1, fused_first),
                self.cme_decoder(latent.common, latent.dynamic_2, fused_second),
            ),
            dim=1,
        )
        return TFHDNOutput(
            clean_pb=self.scaler.inverse(clean_scaled),
            flat_corona=flat,
            normalization_field=self.scaler.inverse(normalization_scaled),
            noise_location=noise_location,
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
            },
        )
