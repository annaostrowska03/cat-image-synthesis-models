"""VQ-VAE with EMA codebook updates and PixelCNN prior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """EMA-updated vector quantizer (commitment loss only for encoder).

    Args:
        num_embeddings: Codebook size K.
        embedding_dim:  Dimensionality of each codebook vector.
        commitment_cost: β — weight of the commitment loss term.
        decay:          EMA decay rate for codebook updates.  Set to 0 to
                        use the original straight-through gradient instead.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed", embed.clone())

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder output.

        Args:
            z_e: Encoder output of shape (B, D, H, W).

        Returns:
            z_q:        Quantized output (B, D, H, W), straight-through gradient.
            indices:    Codebook indices (B, H, W).
            vq_loss:    Scalar VQ loss (commitment only when using EMA).
        """
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)

        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        indices = distances.argmin(1)  # (B*H*W,)

        z_q = self.embedding[indices].view(B, H, W, D).permute(0, 3, 1, 2)

        if self.training and self.decay > 0:
            n = torch.bincount(indices, minlength=self.num_embeddings).float()
            dw = torch.zeros(self.num_embeddings, D, device=flat.device)
            dw.scatter_add_(0, indices.unsqueeze(1).expand(-1, D), flat)
            self.ema_cluster_size.mul_(self.decay).add_(n * (1 - self.decay))
            self.ema_embed.mul_(self.decay).add_(dw * (1 - self.decay))
            n_smooth = (
                (self.ema_cluster_size + 1e-5)
                / (self.ema_cluster_size.sum() + self.num_embeddings * 1e-5)
                * self.ema_cluster_size.sum()
            )
            self.embedding.copy_(self.ema_embed / n_smooth.unsqueeze(1))
            vq_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        else:
            vq_loss = (
                F.mse_loss(z_e.detach(), flat.view(B, H, W, D).permute(0, 3, 1, 2))
                + self.commitment_cost
                * F.mse_loss(z_e, flat.detach().view(B, H, W, D).permute(0, 3, 1, 2))
            )

        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices.view(B, H, W), vq_loss

    @property
    def perplexity_last(self) -> float:
        """Exponential entropy of cluster usage (higher = better codebook use)."""
        probs = self.ema_cluster_size / (self.ema_cluster_size.sum() + 1e-10)
        entropy = -(probs * (probs + 1e-10).log()).sum()
        return entropy.exp().item()

class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class Encoder(nn.Module):
    """Strided-conv encoder, output shape (B, D, H/4, W/4)."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 4, 2, 1),  # /2
            nn.ReLU(True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),  # /2
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            _ResBlock(hidden_channels),
            _ResBlock(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    """Transposed-conv decoder, output shape (B, C_out, 4H, 4W)."""

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_channels: int = 128,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 3, 1, 1),
            _ResBlock(hidden_channels),
            _ResBlock(hidden_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),  # ×2
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels // 2, out_channels, 4, 2, 1),    # ×2
            nn.Tanh(),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.net(z_q)

class VQVAE(nn.Module):
    """Complete VQ-VAE autoencoder.

    Args:
        in_channels:      Image channels (3 for RGB).
        hidden_channels:  Encoder/Decoder hidden width.
        embedding_dim:    Codebook vector dimensionality.
        num_embeddings:   Codebook size.
        commitment_cost:  β for commitment loss.
        decay:            EMA decay for codebook (0 = no EMA).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, indices, vq_loss

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Return flat codebook indices for a batch of images."""
        z_e = self.encoder(x)
        _, indices, _ = self.vq(z_e)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to images.

        Args:
            indices: (B, H_z, W_z) long tensor.
        """
        B, H, W = indices.shape
        z_q = self.vq.embedding[indices.view(-1)].view(B, H, W, -1).permute(0, 3, 1, 2)
        return self.decoder(z_q)

class MaskedConv2d(nn.Conv2d):
    """2-D convolution with an autoregressive mask (type A or B)."""

    def __init__(self, mask_type: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        H, W = self.weight.shape[2], self.weight.shape[3]
        mask = torch.ones_like(self.weight)
        mask[:, :, H // 2, W // 2 + (1 if mask_type == "B" else 0):] = 0
        mask[:, :, H // 2 + 1:] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data.mul_(self.mask)
        return super().forward(x)

class PixelCNN(nn.Module):
    """Shallow PixelCNN prior over VQ-VAE codebook indices.

    Models p(z₁, z₂, …, z_{H×W}) autoregressively over the discrete
    spatial map of codebook indices.

    Args:
        num_embeddings: Vocabulary size (= VQ-VAE codebook size).
        n_layers:       Number of residual gated layers.
        n_filters:      Number of feature channels.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        n_layers: int = 15,
        n_filters: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, n_filters)
        layers: list[nn.Module] = [
            MaskedConv2d("A", n_filters, n_filters, 7, 1, 3, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
        ]
        for _ in range(n_layers):
            layers += [
                MaskedConv2d("B", n_filters, n_filters, 3, 1, 1, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(True),
            ]
        layers += [nn.Conv2d(n_filters, num_embeddings, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Return logits over the codebook for each spatial position.

        Args:
            indices: Long tensor (B, H, W).

        Returns:
            Logits of shape (B, num_embeddings, H, W).
        """
        x = self.embedding(indices).permute(0, 3, 1, 2)  # (B, F, H, W)
        return self.net(x)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        device: torch.device,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively sample a codebook index map.

        Args:
            shape:       (B, H, W) of the desired index map.
            device:      Computation device.
            temperature: Softmax temperature (lower = sharper).

        Returns:
            Long tensor of shape (B, H, W).
        """
        B, H, W = shape
        indices = torch.zeros(B, H, W, dtype=torch.long, device=device)
        for i in range(H):
            for j in range(W):
                logits = self(indices)[:, :, i, j] / temperature
                probs = torch.softmax(logits, dim=-1)
                indices[:, i, j] = torch.multinomial(probs, 1).squeeze(1)
        return indices
