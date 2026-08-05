from __future__ import annotations

"""Conditional directional diffusion model used by the official DLD workflow."""

import math
from typing import Any

import torch
from torch import Tensor, nn

from lnl_toolbox.models.cifar_resnet import cifar_resnet34


def _betas_for_alpha_bar(num_timesteps: int, max_beta: float = 0.999) -> Tensor:
    """Build the cosine schedule used by the official DLD implementation."""

    def alpha_bar(time: float) -> float:
        return math.cos((time + 0.008) / 1.008 * math.pi / 2.0) ** 2

    values = []
    for index in range(num_timesteps):
        first = index / num_timesteps
        second = (index + 1) / num_timesteps
        values.append(min(1.0 - alpha_bar(second) / alpha_bar(first), max_beta))
    return torch.tensor(values, dtype=torch.float32)


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        if dimensions <= 0 or dimensions % 2:
            raise ValueError("time embedding dimension must be a positive even number")
        self.dimensions = int(dimensions)

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps = timesteps.to(dtype=torch.float32).reshape(-1, 1)
        half = self.dimensions // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(1, half - 1)
        )
        angles = timesteps * frequencies.reshape(1, -1)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ConditionalLabelNetwork(nn.Module):
    """MLP conditioned on a label state, representation and diffusion time."""

    def __init__(
        self,
        classes: int,
        feature_dim: int,
        num_timesteps: int,
        *,
        hidden_width: int = 512,
        time_dim: int = 64,
        image_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(int(classes), int(feature_dim), int(num_timesteps), int(hidden_width)) <= 0:
            raise ValueError("diffusion dimensions must be positive")
        self.classes = int(classes)
        self.feature_dim = int(feature_dim)
        self.num_timesteps = int(num_timesteps)
        self.image_dim = int(feature_dim if image_dim is None else image_dim)
        self.time_embedding = _SinusoidalTimeEmbedding(int(time_dim))
        self.network = nn.Sequential(
            nn.Linear(self.classes + self.feature_dim + self.image_dim + int(time_dim), int(hidden_width)),
            nn.SiLU(),
            nn.Linear(int(hidden_width), int(hidden_width)),
            nn.SiLU(),
            nn.Linear(int(hidden_width), self.classes),
        )

    def forward(
        self,
        y_t: Tensor,
        features: Tensor,
        timesteps: Tensor,
        image_features: Tensor | None = None,
    ) -> Tensor:
        if y_t.ndim != 2 or features.ndim != 2 or y_t.shape[0] != features.shape[0]:
            raise ValueError("y_t and features must be aligned [B,C] and [B,D]")
        if y_t.shape[1] != self.classes or features.shape[1] != self.feature_dim:
            raise ValueError("conditional diffusion input dimensions do not match the model")
        if image_features is None:
            image_features = torch.zeros(
                (y_t.shape[0], self.image_dim), dtype=y_t.dtype, device=y_t.device
            )
        if image_features.ndim != 2 or image_features.shape != (y_t.shape[0], self.image_dim):
            raise ValueError("image_features must have shape [B,image_dim]")
        image_features = image_features.to(device=y_t.device, dtype=y_t.dtype)
        timesteps = timesteps.to(device=y_t.device, dtype=torch.long).reshape(-1)
        if timesteps.shape != (y_t.shape[0],) or timesteps.min().item() < 0 or timesteps.max().item() >= self.num_timesteps:
            raise ValueError("diffusion timesteps are outside the configured range")
        time = self.time_embedding(timesteps)
        return self.network(torch.cat((y_t, features, image_features, time), dim=1))


class DirectionalDiffusion(nn.Module):
    """Two-network residual/noise conditional diffusion model.

    DLD trains one network to predict the label-direction residual and one to
    predict Gaussian noise.  Keeping the networks independent matches the
    official ``num_models=2`` objective and makes the two optimizer states
    independently checkpointable.
    """

    def __init__(
        self,
        classes: int,
        feature_dim: int,
        *,
        num_timesteps: int = 1000,
        hidden_width: int = 512,
        time_dim: int = 64,
        beta_start: float = 1e-3,
        beta_end: float = 2e-2,
        schedule: str = "cosine",
        image_base_width: int = 16,
        image_initialization: str = "torch_default",
    ) -> None:
        super().__init__()
        if int(num_timesteps) <= 0:
            raise ValueError("num_timesteps must be positive")
        if not 0.0 < float(beta_start) <= float(beta_end) < 1.0:
            raise ValueError("diffusion beta range must satisfy 0 < start <= end < 1")
        schedule = str(schedule).strip().lower()
        if schedule not in {"average", "cosine"}:
            raise ValueError("DLD schedule must be 'average' or 'cosine'")
        if int(image_base_width) <= 0:
            raise ValueError("image_base_width must be positive")
        self.classes = int(classes)
        self.feature_dim = int(feature_dim)
        self.num_timesteps = int(num_timesteps)
        self.schedule = schedule
        self.residual_model = ConditionalLabelNetwork(
            self.classes, self.feature_dim, self.num_timesteps,
            hidden_width=hidden_width, time_dim=time_dim,
        )
        self.noise_model = ConditionalLabelNetwork(
            self.classes, self.feature_dim, self.num_timesteps,
            hidden_width=hidden_width, time_dim=time_dim,
        )
        # The official model has a separate trainable image encoder for the
        # residual and noise networks.  The frozen feature model remains the
        # two-view pre-correction encoder; these encoders are the diffusion
        # condition and therefore must participate in optimization.
        self.residual_image_encoder = cifar_resnet34(
            num_classes=self.feature_dim,
            base_width=int(image_base_width),
            initialization=image_initialization,
        )
        self.noise_image_encoder = cifar_resnet34(
            num_classes=self.feature_dim,
            base_width=int(image_base_width),
            initialization=image_initialization,
        )

        if schedule == "cosine":
            betas = _betas_for_alpha_bar(self.num_timesteps)
            alphas = 1.0 - betas
            alpha_product = torch.cumprod(alphas, dim=0)
            alpha_cumsum = 1.0 - alpha_product.sqrt()
            beta2_cumsum = 1.0 - alpha_product
            alpha_previous = torch.nn.functional.pad(alpha_cumsum[:-1], (1, 0), value=1.0)
            beta2_previous = torch.nn.functional.pad(beta2_cumsum[:-1], (1, 0), value=1.0)
            alpha_steps = alpha_cumsum - alpha_previous
            beta2_steps = beta2_cumsum - beta2_previous
            if self.num_timesteps > 1:
                alpha_steps[0] = alpha_steps[1]
                beta2_steps[0] = beta2_steps[1]
        else:
            alpha_steps = torch.full((self.num_timesteps,), 1.0 / self.num_timesteps)
            beta2_steps = alpha_steps.clone()
            alpha_cumsum = alpha_steps.cumsum(dim=0).clamp(0.0, 1.0)
            beta2_cumsum = beta2_steps.cumsum(dim=0).clamp(0.0, 1.0)
        self.register_buffer("alphas", alpha_steps)
        self.register_buffer("alpha_cumsum", alpha_cumsum.clamp(0.0, 1.0))
        self.register_buffer("beta2_steps", beta2_steps.clamp_min(0.0))
        self.register_buffer("beta2_cumsum", beta2_cumsum.clamp(0.0, 1.0))
        self.register_buffer("beta_cumsum", beta2_cumsum.clamp_min(1e-12).sqrt())

    def _coefficient(self, values: Tensor, timesteps: Tensor, batch: int) -> Tensor:
        timesteps = timesteps.to(device=values.device, dtype=torch.long).reshape(-1)
        if timesteps.shape != (batch,) or timesteps.min().item() < 0 or timesteps.max().item() >= self.num_timesteps:
            raise ValueError("diffusion timesteps are outside the configured range")
        return values.index_select(0, timesteps).reshape(batch, 1)

    def q_sample(
        self,
        y_input: Tensor,
        y0: Tensor,
        timesteps: Tensor,
        *,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if y_input.shape != y0.shape or y_input.ndim != 2 or y_input.shape[1] != self.classes:
            raise ValueError("y_input and y0 must have shape [B,C]")
        if noise is None:
            noise = torch.randn_like(y0)
        if noise.shape != y0.shape:
            raise ValueError("diffusion noise must align with y0")
        residual = y_input - y0
        alpha = self._coefficient(self.alpha_cumsum, timesteps, y0.shape[0]).to(y0.dtype)
        beta = self._coefficient(self.beta_cumsum, timesteps, y0.shape[0]).to(y0.dtype)
        y_t = y_input + alpha * residual + beta * noise
        return y_t, residual, noise

    def forward_t(
        self,
        y_input: Tensor,
        y0: Tensor,
        features: Tensor,
        timesteps: Tensor,
        *,
        noise: Tensor | None = None,
        images: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        y_t, residual, noise = self.q_sample(y_input, y0, timesteps, noise=noise)
        residual_image_features = self._encode_image(images, self.residual_image_encoder)
        noise_image_features = self._encode_image(images, self.noise_image_encoder)
        predicted_residual = self.residual_model(
            y_t, features, timesteps, residual_image_features
        )
        predicted_noise = self.noise_model(
            y_t, features, timesteps, noise_image_features
        )
        return predicted_residual, predicted_noise, residual, noise, y_t

    def _encode_image(self, images: Tensor | None, encoder: nn.Module) -> Tensor | None:
        if images is None:
            return None
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        output = encoder(images)
        if output.ndim != 2 or output.shape[1] != self.feature_dim:
            raise ValueError("image encoder output dimension does not match feature_dim")
        return output

    @torch.no_grad()
    def sample(
        self,
        features: Tensor,
        *,
        sampling_timesteps: int = 10,
        y_input: Tensor | None = None,
        residual_model: nn.Module | None = None,
        noise_model: nn.Module | None = None,
        images: Tensor | None = None,
    ) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape [B,D]")
        steps = max(1, min(int(sampling_timesteps), self.num_timesteps))
        residual_model = self.residual_model if residual_model is None else residual_model
        noise_model = self.noise_model if noise_model is None else noise_model
        y = (
            torch.zeros(
                (features.shape[0], self.classes),
                dtype=features.dtype,
                device=features.device,
            )
            if y_input is None
            else y_input.clone()
        )
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, steps, device=features.device
        ).round().to(dtype=torch.long)
        for position, timestep in enumerate(timesteps):
            batch_timesteps = torch.full(
                (features.shape[0],), int(timestep.item()),
                dtype=torch.long, device=features.device,
            )
            residual_image_features = self._encode_image(images, self.residual_image_encoder)
            noise_image_features = self._encode_image(images, self.noise_image_encoder)
            predicted_residual = residual_model(
                y, features, batch_timesteps, residual_image_features
            )
            predicted_noise = noise_model(
                y, features, batch_timesteps, noise_image_features
            )
            next_timestep = -1 if position + 1 == len(timesteps) else int(timesteps[position + 1].item())
            if next_timestep < 0:
                y = y - self._coefficient(self.alpha_cumsum, batch_timesteps, y.shape[0]) * predicted_residual - self._coefficient(self.beta_cumsum, batch_timesteps, y.shape[0]) * predicted_noise
            else:
                next_batch = torch.full(
                    (features.shape[0],), next_timestep, dtype=torch.long, device=features.device
                )
                alpha_step = self._coefficient(self.alpha_cumsum, batch_timesteps, y.shape[0]) - self._coefficient(self.alpha_cumsum, next_batch, y.shape[0])
                beta_step = self._coefficient(self.beta_cumsum, batch_timesteps, y.shape[0]) - self._coefficient(self.beta_cumsum, next_batch, y.shape[0])
                y = y - alpha_step * predicted_residual - beta_step * predicted_noise
        return torch.softmax(y, dim=1)


# A descriptive alias keeps configuration code readable while retaining the
# concise class name used by existing model modules.
DirectionalLabelDiffusion = DirectionalDiffusion


__all__ = [
    "ConditionalLabelNetwork",
    "DirectionalDiffusion",
    "DirectionalLabelDiffusion",
]
