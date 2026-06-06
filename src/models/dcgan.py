"""DCGAN Generator and Discriminator for 64x64 cat image synthesis.

Architecture follows the original DCGAN paper (Radford et al., 2015) with
optional WGAN-GP support (gradient penalty computed externally during training).
The discriminator outputs a raw logit; use BCEWithLogitsLoss or a WGAN critic loss.
"""

import torch
import torch.nn as nn


def _weights_init(m: nn.Module) -> None:
    """Initialize Conv and BatchNorm layers as suggested in the DCGAN paper."""
    classname = type(m).__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    """DCGAN Generator.

    Maps a latent vector z of shape (B, z_dim) to an RGB image of shape
    (B, 3, image_size, image_size).  Supports image_size ∈ {64, 128}.

    Args:
        z_dim:         Dimensionality of the input noise vector.
        base_channels: Channel width at the coarsest feature map (4×4).
                       The channel count halves with each upsampling step.
        image_size:    Output spatial resolution (64 or 128).
        image_channels: Number of output channels (3 for RGB).
    """

    def __init__(
        self,
        z_dim: int = 128,
        base_channels: int = 64,
        image_size: int = 64,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        assert image_size in (64, 128), "image_size must be 64 or 128"
        bc = base_channels
        n_up = 4 if image_size == 64 else 5
        first_channels = bc * (2 ** (n_up - 1))  # 512 for 64px, 1024 for 128px

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(z_dim, first_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(first_channels),
            nn.ReLU(True),
        ]
        in_ch = first_channels
        for _ in range(n_up - 1):
            out_ch = in_ch // 2
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ]
            in_ch = out_ch

        layers += [
            nn.ConvTranspose2d(in_ch, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.unsqueeze(-1).unsqueeze(-1))

class Discriminator(nn.Module):
    """DCGAN Discriminator (critic).

    Maps an RGB image (B, 3, image_size, image_size) to a scalar logit per
    sample (B, 1).  No Sigmoid is applied so the output can be used with
    BCEWithLogitsLoss **or** as a WGAN-style critic without modification.

    Args:
        base_channels: Channel width at the first conv layer.
        image_size:    Input spatial resolution (64 or 128).
        image_channels: Number of input channels.
    """

    def __init__(
        self,
        base_channels: int = 64,
        image_size: int = 64,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        assert image_size in (64, 128), "image_size must be 64 or 128"
        bc = base_channels
        n_down = 4 if image_size == 64 else 5

        layers: list[nn.Module] = [
            nn.Conv2d(image_channels, bc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = bc
        for _ in range(n_down - 1):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch

        layers += [nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False)]
        self.net = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), 1)

def compute_gradient_penalty(
    discriminator: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP gradient penalty (Gulrajani et al., 2017).

    Args:
        discriminator: The Discriminator / critic network.
        real:          Batch of real images (B, C, H, W).
        fake:          Batch of generated images (B, C, H, W).
        device:        Computation device.

    Returns:
        Scalar gradient-penalty loss term.
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(B, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
