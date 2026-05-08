"""Lightning module for the conditional Δhm diffusion U-Net.

Training objective: v-prediction with cosine noise schedule.
Inference: DDIM sampler, ensemble of N samples per chip aggregated to median +
2.5% / 97.5% quantiles + std.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore[no-redef]

from diffusers import DDPMScheduler, DDIMScheduler

from .diffusion_unet import ConditionalDiffusionUNet


def _maybe_build_location_encoder(
    locenc_kwargs: Optional[Mapping[str, Any]],
):
    """Lazy import + construct the existing LocationEncoder, or return None."""
    if not locenc_kwargs:
        return None
    from src.locationencoder import LocationEncoder
    backbone = locenc_kwargs.get("backbone", ("sphericalharmonics", "siren"))
    out_channels = int(locenc_kwargs.get("out_channels", 8))
    hparams = dict(locenc_kwargs.get("hparams") or {})
    hparams.setdefault("legendre_polys", 10)
    hparams.setdefault("dim_hidden", 64)
    hparams.setdefault("num_layers", 2)
    hparams.setdefault("optimizer", dict(lr=1e-4, wd=1e-3))
    hparams["num_classes"] = out_channels
    return LocationEncoder(backbone[0], backbone[1], hparams), out_channels


class DiffusionLightningModule(pl.LightningModule):
    """v-prediction conditional diffusion for 20yr Δhm forecasting."""

    def __init__(
        self,
        cond_channels: int,
        sample_size: int = 64,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 2, 4),
        attention_at_low_two: bool = True,
        layers_per_block: int = 2,
        attention_head_dim: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 30,
        ensemble_n: int = 16,
        location_encoder_kwargs: Optional[Mapping[str, Any]] = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.ensemble_n = ensemble_n
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)

        loc = _maybe_build_location_encoder(location_encoder_kwargs)
        if loc is None:
            self.location_encoder = None
            self.locenc_out_channels = 0
        else:
            self.location_encoder, self.locenc_out_channels = loc

        self.unet = ConditionalDiffusionUNet(
            cond_channels=cond_channels,
            sample_size=sample_size,
            base_channels=base_channels,
            channel_mults=tuple(channel_mults),
            attention_at_low_two=attention_at_low_two,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
        )

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type="v_prediction",
            beta_schedule="squaredcos_cap_v2",
        )

        # EMA state — tracked manually as a dict of detached tensors so it
        # round-trips through Lightning's checkpoint without needing to be
        # an nn.Module submodule (which would double the param count in
        # the trainer summary).
        self._ema_state: Optional[dict] = None
        if self.use_ema:
            self._ema_state = {
                name: p.detach().clone()
                for name, p in self.unet.named_parameters()
                if p.requires_grad
            }

    def assemble_conditioning(
        self,
        input_dynamic: torch.Tensor,   # [B, T, C_dyn, H, W]
        input_static: torch.Tensor,    # [B, C_static, H, W]
        lonlat: Optional[torch.Tensor],   # [B, H, W, 2]
        hm_t_normalized: torch.Tensor, # [B, 1, H, W]
    ) -> torch.Tensor:
        """Concatenate all conditioning channels into [B, C_cond, H, W].

        Replaces non-finite values (NaN/±Inf) with 0 — chips that overlap
        ocean/no-data carry NaNs in HM and a few static channels; convs would
        otherwise propagate NaN through the entire U-Net output. The valid_mask
        on the loss already drops these pixels from the gradient.
        """
        B, T, C_dyn, H, W = input_dynamic.shape
        dyn_flat = input_dynamic.reshape(B, T * C_dyn, H, W)
        parts = [dyn_flat, input_static]
        if self.location_encoder is not None and lonlat is not None:
            ll_flat = lonlat.reshape(B * H * W, 2)
            loc = self.location_encoder(ll_flat)
            loc_grid = loc.reshape(B, H, W, self.locenc_out_channels).permute(0, 3, 1, 2).contiguous()
            parts.append(loc_grid)
        parts.append(hm_t_normalized)
        cond = torch.cat(parts, dim=1)
        return torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_conditioning_from_batch(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.assemble_conditioning(
            batch["input_dynamic"],
            batch["input_static"],
            batch.get("lonlat"),
            batch["hm_t_normalized"],
        )

    def _denoising_step(
        self,
        batch: Mapping[str, Any],
        log_prefix: str,
    ) -> torch.Tensor:
        cond = self._build_conditioning_from_batch(batch)
        x_0 = batch["target_dhm"]                  # [B, 1, H, W]
        valid = batch["valid_mask"].unsqueeze(1)   # [B, 1, H, W]
        B = x_0.shape[0]
        t = torch.randint(
            0, self.num_train_timesteps, (B,), device=x_0.device, dtype=torch.long
        )
        noise = torch.randn_like(x_0)
        noisy = self.train_scheduler.add_noise(x_0, noise, t)
        v_target = self.train_scheduler.get_velocity(x_0, noise, t)
        v_pred = self.unet(noisy, cond, t)

        if valid.any():
            loss = F.mse_loss(v_pred[valid.expand_as(v_pred)], v_target[valid.expand_as(v_target)])
        else:
            loss = F.mse_loss(v_pred, v_target)

        self.log(f"{log_prefix}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._denoising_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._denoising_step(batch, "val")

    # --- EMA helpers ---

    def _ema_to_device(self, device):
        if self._ema_state is None:
            return
        for name, t in self._ema_state.items():
            if t.device != device:
                self._ema_state[name] = t.to(device)

    def on_train_start(self):
        self._ema_to_device(self.device)

    def on_train_batch_end(self, *_args, **_kwargs):
        if self._ema_state is None:
            return
        d = self.ema_decay
        with torch.no_grad():
            for name, p in self.unet.named_parameters():
                if not p.requires_grad:
                    continue
                ema = self._ema_state[name]
                if ema.device != p.device:
                    ema = ema.to(p.device)
                    self._ema_state[name] = ema
                ema.mul_(d).add_(p.detach(), alpha=1.0 - d)

    def on_save_checkpoint(self, ckpt):
        if self._ema_state is not None:
            ckpt["ema_state"] = {k: v.detach().cpu() for k, v in self._ema_state.items()}

    def on_load_checkpoint(self, ckpt):
        if self.use_ema and "ema_state" in ckpt:
            self._ema_state = {k: v.clone() for k, v in ckpt["ema_state"].items()}

    def _swap_to_ema(self):
        """Backup current unet weights and load EMA into the unet. Returns the backup."""
        if self._ema_state is None:
            return None
        backup = {}
        with torch.no_grad():
            for name, p in self.unet.named_parameters():
                if not p.requires_grad:
                    continue
                backup[name] = p.detach().clone()
                ema = self._ema_state[name]
                if ema.device != p.device:
                    ema = ema.to(p.device)
                p.data.copy_(ema)
        return backup

    def _restore_from_backup(self, backup):
        if backup is None:
            return
        with torch.no_grad():
            for name, p in self.unet.named_parameters():
                if not p.requires_grad or name not in backup:
                    continue
                p.data.copy_(backup[name])

    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,   # [B, C_cond, H, W]
        n_samples: int = 1,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        use_ema: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run DDIM sampling. Returns [n_samples, B, 1, H, W].

        If `use_ema` is True (or None and self.use_ema is True), swap to EMA
        weights for sampling, then restore the live training weights.
        """
        steps = num_inference_steps or self.num_inference_steps
        scheduler = DDIMScheduler.from_config(self.train_scheduler.config)
        scheduler.set_timesteps(steps, device=conditioning.device)

        if use_ema is None:
            use_ema = self.use_ema and self._ema_state is not None
        backup = self._swap_to_ema() if use_ema else None

        try:
            B, C_cond, H, W = conditioning.shape
            cond_tiled = conditioning.repeat_interleave(n_samples, dim=0)  # [N*B, ...]
            x = torch.randn(
                n_samples * B, 1, H, W, device=conditioning.device,
                dtype=conditioning.dtype, generator=generator,
            )
            for t in scheduler.timesteps:
                t_batch = t.expand(x.shape[0]).to(x.device)
                v_pred = self.unet(x, cond_tiled, t_batch)
                x = scheduler.step(v_pred, t, x).prev_sample
            return x.view(n_samples, B, 1, H, W)
        finally:
            self._restore_from_backup(backup)

    def configure_optimizers(self):
        try:
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay, fused=True,
            )
        except (TypeError, RuntimeError):
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
        return opt
