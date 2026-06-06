"""DDPM — Denoising Diffusion Probabilistic Model.

Implements:
- Linear (and cosine) noise schedule
- UNet with residual blocks, sinusoidal time embedding and self-attention
- Standard DDPM training loss (predict ε)
- DDIM deterministic sampler for fast inference
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999)

class GaussianDiffusion:
    """Pre-computes all diffusion schedule tensors and provides
    forward-process sampling and loss computation.

    Args:
        timesteps: T — total diffusion steps.
        beta_start / beta_end: Endpoints of the linear schedule.
        schedule:  "linear" or "cosine".
        device:    Buffers are stored on this device.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.T = timesteps
        self.device = device

        def buf(x: torch.Tensor) -> torch.Tensor:
            return x.float().to(device)

        self.betas = buf(betas)
        self.alphas_cumprod = buf(alphas_cumprod)
        self.alphas_cumprod_prev = buf(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = buf(alphas_cumprod.sqrt())
        self.sqrt_one_minus_alphas_cumprod = buf((1.0 - alphas_cumprod).sqrt())
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_var = buf(posterior_var.clamp(min=1e-20).log())
        self.posterior_mean_coef1 = buf(
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = buf(
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)
        )

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t given x_0 at timesteps t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_1m_acp * noise, noise

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor
    ) -> torch.Tensor:
        """Simple ε-prediction loss: E[||ε - ε_θ(x_t, t)||²]."""
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        x_t, noise = self.q_sample(x0, t)
        pred_noise = model(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        t_tensor = torch.full((x_t.size(0),), t, device=self.device, dtype=torch.long)
        pred_noise = model(x_t, t_tensor)
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        sqrt_acp = self.sqrt_alphas_cumprod[t]
        sqrt_1m_acp = self.sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x_t - sqrt_1m_acp * pred_noise) / sqrt_acp
        x0_pred = x0_pred.clamp(-1, 1)
        mean = coef1 * x0_pred + coef2 * x_t
        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        log_var = self.posterior_log_var[t]
        return mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, model: nn.Module, shape: tuple, verbose: bool = False
    ) -> torch.Tensor:
        """Full DDPM reverse diffusion loop (slow, T steps)."""
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple,
        ddim_steps: int = 50,
        eta: float = 0.0,
        x_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDIM deterministic sampling (Song et al., 2020).

        Args:
            model:      The noise-predicting UNet.
            shape:      Output shape (B, C, H, W).
            ddim_steps: Number of inference steps (much fewer than T).
            eta:        Stochasticity parameter (0 = fully deterministic).
            x_T:        Optional starting noise tensor. If None, sampled from N(0,I).

        Returns:
            Sampled images in [-1, 1].
        """
        step_size = self.T // ddim_steps
        timesteps = list(reversed(range(0, self.T, step_size)))
        x = torch.randn(shape, device=self.device) if x_T is None else x_T

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            pred_noise = model(x, t_tensor)
            acp = self.alphas_cumprod[t]
            x0_pred = (x - (1 - acp).sqrt() * pred_noise) / acp.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                acp_prev = self.alphas_cumprod[t_prev]
            else:
                acp_prev = torch.ones(1, device=self.device)

            sigma = eta * ((1 - acp_prev) / (1 - acp) * (1 - acp / acp_prev)).sqrt()
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = acp_prev.sqrt() * x0_pred + (1 - acp_prev - sigma ** 2).clamp(0).sqrt() * pred_noise + sigma * noise

        return x

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        return embedding

class TimeCondResBlock(nn.Module):
    """Residual block conditioned on a time embedding vector."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    """Multi-head self-attention over 2-D feature maps."""

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        out, _ = self.attn(h, h, h)
        return x + out.permute(0, 2, 1).view(B, C, H, W)

class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))

class UNet(nn.Module):
    """Conditional UNet for noise prediction in DDPM.

    Args:
        image_channels:        Input/output image channels.
        base_channels:         Width at the coarsest encoder level.
        channel_mults:         Multipliers per resolution (finest to coarsest).
        attention_resolutions: Spatial sizes at which to apply self-attention.
        num_res_blocks:        Number of residual blocks per resolution level.
        dropout:               Dropout rate in residual blocks.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 2, 2),
        attention_resolutions: tuple[int, ...] = (16,),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(image_channels, base_channels, 3, 1, 1)
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.down_samples: nn.ModuleList = nn.ModuleList()
        channels = [base_channels]
        cur_ch = base_channels
        cur_res = 64  # assume 64×64 input

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = TimeCondResBlock(cur_ch, out_ch, time_dim, dropout)
                attn = SelfAttention2d(out_ch) if cur_res in attention_resolutions else nn.Identity()
                self.down_blocks.append(nn.ModuleList([block, attn]))
                channels.append(out_ch)
                cur_ch = out_ch
            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample(cur_ch))
                channels.append(cur_ch)
                cur_res //= 2

        self.mid_res1 = TimeCondResBlock(cur_ch, cur_ch, time_dim, dropout)
        self.mid_attn = SelfAttention2d(cur_ch)
        self.mid_res2 = TimeCondResBlock(cur_ch, cur_ch, time_dim, dropout)

        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.up_samples: nn.ModuleList = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                block = TimeCondResBlock(cur_ch + skip_ch, out_ch, time_dim, dropout)
                attn = SelfAttention2d(out_ch) if cur_res in attention_resolutions else nn.Identity()
                self.up_blocks.append(nn.ModuleList([block, attn]))
                cur_ch = out_ch
            if level > 0:
                self.up_samples.append(Upsample(cur_ch))
                cur_res *= 2

        self.out_norm = nn.GroupNorm(8, cur_ch)
        self.out_conv = nn.Conv2d(cur_ch, image_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.input_conv(x)
        skips = [h]

        ds_iter = iter(self.down_samples)
        block_idx = 0
        for level in range(len(self.channel_mults)):
            for _ in range(self.num_res_blocks):
                block, attn = self.down_blocks[block_idx]
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)
                block_idx += 1
            if level < len(self.channel_mults) - 1:
                h = next(ds_iter)(h)
                skips.append(h)

        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        us_iter = iter(self.up_samples)
        block_idx = 0
        for level in range(len(self.channel_mults) - 1, -1, -1):
            for _ in range(self.num_res_blocks + 1):
                block, attn = self.up_blocks[block_idx]
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = attn(h)
                block_idx += 1
            if level > 0:
                h = next(us_iter)(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)

class EMA:
    """Exponential Moving Average of model parameters for better sample quality."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data * (1 - self.decay))

    def apply_shadow(self) -> None:
        """Replace model weights with EMA weights (for sampling)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, original: dict[str, torch.Tensor]) -> None:
        """Restore original weights after sampling."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(original[name])
