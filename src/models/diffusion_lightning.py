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


class CorrDiffMeanHead(nn.Module):
    """Small deterministic conv stack that predicts the conditional-mean Δhm μ
    from the same conditioning the diffusion U-Net sees.

    Trained jointly with an MSE loss on (μ, target_dhm). The diffusion model
    learns the residual r = target − μ.detach(), so its entropy budget is no
    longer spent re-modelling the conditional mean — that frees variance for
    rare-event sampling, which is the v8/v12 bottleneck.
    """
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, cond):
        return self.net(cond)


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
        pixel_weight_alpha: float = 0.0,
        pixel_weight_eps: float = 0.05,
        pattern_loss_weight: float = 0.0,
        pattern_thresholds: Sequence[float] = (0.05, 0.4),
        pattern_scales: Sequence[int] = (8, 16),
        pattern_temperature: float = 0.02,
        dhm_mean: float = 0.0,
        dhm_std: float = 1.0,
        cfg_dropout_prob: float = 0.0,
        min_snr_gamma: float = 0.0,
        dhm_transform: str = "none",
        dhm_log_scale: float = 0.05,
        exloss_lambda: float = 0.0,
        wasserstein_loss_weight: float = 0.0,
        wasserstein_n_quantiles: int = 256,
        use_magnitude_cond: bool = False,
        m_dropout_prob: float = 0.0,
        m_norm_scale: float = 0.5,
        use_mean_head: bool = False,
        mean_head_hidden: int = 64,
        mean_loss_weight: float = 1.0,
        mean_head_pixel_weight_alpha: float = 0.0,
        mean_head_pixel_weight_eps: float = 0.01,
        tile_mean_loss_weight: float = 0.0,
        tile_mean_scales: Sequence[int] = (8, 16),
        hist_loss_weight: float = 0.0,
        hist_bin_edges: Sequence[float] = (-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0),
        hist_scales: Sequence[int] = (16, 32),
        hist_temperature: float = 0.01,
        tv_loss_weight: float = 0.0,
        tv_loss_target_floor: float = 0.05,
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
        self.pixel_weight_alpha = float(pixel_weight_alpha)
        self.pixel_weight_eps = float(pixel_weight_eps)
        self.pattern_loss_weight = float(pattern_loss_weight)
        self.pattern_thresholds = tuple(float(t) for t in pattern_thresholds)
        self.pattern_scales = tuple(int(s) for s in pattern_scales)
        self.pattern_temperature = float(pattern_temperature)
        self.dhm_mean = float(dhm_mean)
        self.dhm_std = float(dhm_std)
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.min_snr_gamma = float(min_snr_gamma)
        if dhm_transform not in ("none", "signed_log1p"):
            raise ValueError(
                f"dhm_transform must be 'none' or 'signed_log1p', got {dhm_transform!r}"
            )
        self.dhm_transform = str(dhm_transform)
        self.dhm_log_scale = float(dhm_log_scale)
        self.exloss_lambda = float(exloss_lambda)
        self.wasserstein_loss_weight = float(wasserstein_loss_weight)
        self.wasserstein_n_quantiles = int(wasserstein_n_quantiles)
        self.use_magnitude_cond = bool(use_magnitude_cond)
        self.m_dropout_prob = float(m_dropout_prob)
        self.m_norm_scale = float(m_norm_scale)
        self.use_mean_head = bool(use_mean_head)
        self.mean_loss_weight = float(mean_loss_weight)
        self.mean_head_pixel_weight_alpha = float(mean_head_pixel_weight_alpha)
        self.mean_head_pixel_weight_eps = float(mean_head_pixel_weight_eps)
        self.tile_mean_loss_weight = float(tile_mean_loss_weight)
        self.tile_mean_scales = tuple(int(s) for s in tile_mean_scales)
        self.hist_loss_weight = float(hist_loss_weight)
        self.register_buffer(
            "hist_bin_edges",
            torch.tensor(list(hist_bin_edges), dtype=torch.float32),
        )
        self.hist_scales = tuple(int(s) for s in hist_scales)
        self.hist_temperature = float(hist_temperature)
        self.tv_loss_weight = float(tv_loss_weight)
        self.tv_loss_target_floor = float(tv_loss_target_floor)

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

        if self.use_mean_head:
            self.mean_head = CorrDiffMeanHead(cond_channels, hidden=int(mean_head_hidden))
        else:
            self.mean_head = None

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
        m_scalar: Optional[torch.Tensor] = None,   # [B] raw |Δhm| scalar; only used if use_magnitude_cond
    ) -> torch.Tensor:
        """Concatenate all conditioning channels into [B, C_cond, H, W].

        When use_magnitude_cond is True, an extra channel encoding the per-chip
        max|Δhm| (FIDE-style block-maxima conditioning) is appended after
        normalisation by m_norm_scale. m_scalar=None ⇒ the channel is filled with
        zeros (the "null" / unknown M sentinel used during dropout/inference
        without explicit M).

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
        if self.use_magnitude_cond:
            if m_scalar is None:
                m_chan = torch.zeros((B, 1, H, W), device=input_dynamic.device, dtype=input_dynamic.dtype)
            else:
                m_norm = (m_scalar.float() / max(self.m_norm_scale, 1e-8)).view(B, 1, 1, 1)
                m_chan = m_norm.expand(B, 1, H, W).to(input_dynamic.dtype)
            parts.append(m_chan)
        cond = torch.cat(parts, dim=1)
        return torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_conditioning_from_batch(
        self, batch: Mapping[str, Any], drop_m: bool = False,
    ) -> torch.Tensor:
        m = None
        if self.use_magnitude_cond and "target_max_dhm" in batch:
            m = batch["target_max_dhm"]
            if drop_m and self.training and self.m_dropout_prob > 0:
                # FIDE-style: drop the magnitude scalar (replace with null) for a
                # subset of the batch so the model also learns the m-unconditional
                # distribution. Spatial conditioning is *not* dropped.
                B = m.shape[0]
                keep = (torch.rand(B, device=m.device) > self.m_dropout_prob)
                m = torch.where(keep, m, torch.zeros_like(m))
        return self.assemble_conditioning(
            batch["input_dynamic"],
            batch["input_static"],
            batch.get("lonlat"),
            batch["hm_t_normalized"],
            m_scalar=m,
        )

    def _snr_weights(self, t):
        """Min-SNR-γ per-timestep weight for v-prediction.

        Hang et al. 2023 — for v-prediction the loss should be weighted by
        min(γ, SNR(t)) / (1 + SNR(t)) so high-SNR (low-noise) steps get
        relatively more gradient. Without this the v-loss is roughly uniform
        across t, which under-weights the low-noise regime where rare
        high-magnitude features actually live.
        """
        if self.min_snr_gamma <= 0:
            return None
        alphas_cumprod = self.train_scheduler.alphas_cumprod.to(t.device)[t]
        snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)
        w = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / (1.0 + snr)
        return w.view(-1, 1, 1, 1)

    def _v_loss(self, v_pred, v_target, valid, target_dhm, t):
        """Per-pixel + per-timestep weighted v-prediction MSE.

        - pixel_weight_alpha > 0 scales each pixel's MSE by (|target_dhm|^α + ε),
          normalised so the average pixel weight on valid pixels is 1.
        - exloss_lambda > 0 (Tier 2B / ExtremeCast Gong 2024) replaces the
          symmetric pixel weight with an asymmetric one that penalises *under-
          predictions* of large |target_dhm| more than over-predictions. Under
          v-prediction parametrisation, "x_pred under-predicts magnitude" iff
          (v_pred - v_target) * sign(target_dhm) > 0 (since x_pred and v_pred
          are negatively related at fixed t). When both are set, exloss_lambda
          takes precedence.
        - min_snr_gamma > 0 applies Hang-2023 min-SNR-γ per-timestep weighting
          (γ ≈ 5 is standard for v-prediction).
        """
        valid_f = valid.float()
        err = v_pred - v_target
        sq = err.pow(2)
        if self.exloss_lambda > 0:
            # Asymmetric scaling on under-predictions of magnitude.
            under = ((err * torch.sign(target_dhm)) > 0).float()
            w_pix = 1.0 + self.exloss_lambda * target_dhm.abs() * under
            # Renormalise so average pixel weight on valid pixels is 1, keeping
            # loss scale comparable to the unweighted case.
            w_sum = (w_pix * valid_f).sum()
            v_sum = valid_f.sum().clamp(min=1.0)
            w_pix = w_pix * (v_sum / w_sum.clamp(min=1e-8))
        elif self.pixel_weight_alpha > 0:
            w_pix = target_dhm.abs().pow(self.pixel_weight_alpha) + self.pixel_weight_eps
            w_sum = (w_pix * valid_f).sum()
            v_sum = valid_f.sum().clamp(min=1.0)
            w_pix = w_pix * (v_sum / w_sum.clamp(min=1e-8))
        else:
            w_pix = torch.ones_like(sq)
        # SNR weight (1 if disabled). Per-batch-element [B, 1, 1, 1].
        snr_w = self._snr_weights(t)
        if snr_w is not None:
            sq = sq * snr_w
        denom = valid_f.sum().clamp(min=1.0)
        return (sq * w_pix * valid_f).sum() / denom

    def _tv_loss(self, x0_pred, target_dhm, valid):
        """Total Variation loss on predicted Δhm — addresses sample noisiness.

        Each diffusion sample has uniform ~0.028 per-pixel noise around the
        median, which makes individual samples look granular even where the
        truth is smooth (most pixels = 0). Penalising ‖∇x_pred‖ teaches the
        U-Net to produce spatially smooth outputs at all noise levels.

        To preserve the model's ability to produce *large positives* on the
        rare event tiles, we mask the TV penalty out wherever |target| exceeds
        `tv_loss_target_floor`: smoothness is only required in low-magnitude
        regions where the truth is genuinely flat.
        """
        if self.tv_loss_weight <= 0:
            return torch.zeros((), device=x0_pred.device)
        # |target| ≤ floor → smoothness applies; |target| > floor → free.
        floor = self.tv_loss_target_floor
        smooth_mask = (target_dhm.abs() <= floor).float()
        valid_f = valid.float()
        # Edge-mask: only penalise gradients where BOTH endpoints are smooth-and-valid.
        # mask for vertical edges (dx) — between rows i and i+1
        m_dx = smooth_mask[..., :-1, :] * smooth_mask[..., 1:, :] * \
               valid_f[..., :-1, :] * valid_f[..., 1:, :]
        m_dy = smooth_mask[..., :, :-1] * smooth_mask[..., :, 1:] * \
               valid_f[..., :, :-1] * valid_f[..., :, 1:]
        dx = (x0_pred[..., 1:, :] - x0_pred[..., :-1, :]).abs()
        dy = (x0_pred[..., :, 1:] - x0_pred[..., :, :-1]).abs()
        denom = (m_dx.sum() + m_dy.sum()).clamp(min=1.0)
        return ((dx * m_dx).sum() + (dy * m_dy).sum()) / denom

    def _per_tile_hist_loss(self, x0_pred, target_dhm, valid):
        """Per-tile histogram-intersection loss — directly mirrors `xinter` in
        `scripts/evaluate_diffusion.py`.

        For each evaluation bin [edge_i, edge_{i+1}), each pixel gets a soft
        membership via sigmoid differences; per-tile histograms are obtained
        by avg_pool2d at multiple scales; pred and target tile-histograms are
        normalised so they sum to 1 per tile; loss = 1 − Σ_bin min(p, q),
        averaged over tiles and scales. Targets the user-stated 'tile-level
        histogram agreement' criterion directly.
        """
        if self.hist_loss_weight <= 0 or len(self.hist_bin_edges) < 2:
            return torch.zeros((), device=x0_pred.device)
        edges = self.hist_bin_edges.to(x0_pred.device)
        T = self.hist_temperature
        valid_f = valid.float()
        # Per-pixel soft bin membership: [B, n_bins, H, W]
        # bin_i membership = sigmoid((x − edge_i)/T) − sigmoid((x − edge_{i+1})/T)
        sig_pred = torch.sigmoid((x0_pred - edges.view(1, -1, 1, 1)) / T)
        sig_targ = torch.sigmoid((target_dhm - edges.view(1, -1, 1, 1)) / T)
        # bin counts: difference of consecutive sigmoid responses
        h_pred = (sig_pred[:, :-1] - sig_pred[:, 1:]) * valid_f  # [B, n_bins, H, W]
        h_targ = (sig_targ[:, :-1] - sig_targ[:, 1:]) * valid_f
        terms = []
        for s in self.hist_scales:
            if x0_pred.shape[-1] < s or x0_pred.shape[-2] < s:
                continue
            # Per-tile sums (proportional to histogram counts)
            cp = F.avg_pool2d(h_pred, s, s)  # [B, n_bins, H/s, W/s]
            ct = F.avg_pool2d(h_targ, s, s)
            # Normalise per tile so the histogram sums to 1
            cp = cp / (cp.sum(dim=1, keepdim=True) + 1e-6)
            ct = ct / (ct.sum(dim=1, keepdim=True) + 1e-6)
            inter = torch.minimum(cp, ct).sum(dim=1)  # [B, H/s, W/s]
            terms.append((1.0 - inter).mean())
        if not terms:
            return torch.zeros((), device=x0_pred.device)
        return torch.stack(terms).mean()

    def _tile_mean_loss(self, x0_pred, target_dhm, valid):
        """Multi-scale tile-mean MSE: ‖avg_pool(pred) − avg_pool(target)‖².

        Targets the *amount* of change at the tile level (per-tile aggregate
        magnitude) — directly aligned with the user's stated success metric
        (pixel coverage + tile-level amount, not pixel-precise matching).
        Pooling makes this insensitive to which pixel within a tile carries
        the change.
        """
        if self.tile_mean_loss_weight <= 0 or not self.tile_mean_scales:
            return torch.zeros((), device=x0_pred.device)
        valid_f = valid.float()
        # Mask both fields, average only over valid pixels per tile.
        terms = []
        for s in self.tile_mean_scales:
            if x0_pred.shape[-1] < s or x0_pred.shape[-2] < s:
                continue
            p_sum = F.avg_pool2d(x0_pred * valid_f, s, s)
            t_sum = F.avg_pool2d(target_dhm * valid_f, s, s)
            v_sum = F.avg_pool2d(valid_f, s, s).clamp(min=1e-3)
            p_mean = p_sum / v_sum
            t_mean = t_sum / v_sum
            terms.append((p_mean - t_mean).pow(2).mean())
        if not terms:
            return torch.zeros((), device=x0_pred.device)
        return torch.stack(terms).mean()

    def _marginal_wasserstein_loss(self, x0_pred, target_dhm, valid):
        """Sliced 1D Wasserstein on the marginal pixel-value histogram (Tier 2C).

        Forces the predicted Δhm distribution (over all valid pixels in the
        batch) to match the target distribution. Per-pixel order-free — only
        the histogram shape is constrained — which directly addresses the
        observed mode-collapse to the conditional mean: even if the model
        cannot place high-magnitude pixels at the right *locations*, the
        Wasserstein loss demands they appear *somewhere*.
        """
        if self.wasserstein_loss_weight <= 0:
            return torch.zeros((), device=x0_pred.device)
        v = valid.bool() if valid.dtype != torch.bool else valid
        a = x0_pred[v]
        b = target_dhm[v]
        n = min(a.numel(), b.numel())
        if n < 64:
            return torch.zeros((), device=x0_pred.device)
        a_sorted, _ = torch.sort(a)
        b_sorted, _ = torch.sort(b)
        # Sample the same N quantiles from each via index linspace.
        N = min(self.wasserstein_n_quantiles, n)
        idx = torch.linspace(0, n - 1, N, device=a_sorted.device).long()
        a_q = a_sorted[idx]
        b_q = b_sorted[idx]
        return (a_q - b_q).abs().mean()

    def _x0_pred_from_v(self, noisy, v_pred, t):
        """Recover predicted clean x_0 from v_pred under v-prediction parametrisation."""
        alphas = self.train_scheduler.alphas_cumprod.to(t.device)[t]  # [B]
        alpha_sqrt = alphas.sqrt().view(-1, 1, 1, 1)
        sigma_sqrt = (1.0 - alphas).clamp(min=0).sqrt().view(-1, 1, 1, 1)
        return alpha_sqrt * noisy - sigma_sqrt * v_pred

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Convert model-space (normalised, possibly transformed) Δhm back to raw Δhm units.

        Inverts (1) z-score by self.dhm_mean/dhm_std and (2) the forward target
        transform applied in the dataloader (e.g. signed_log1p). Mirrors
        HumanFootprintChipDataset._apply_dhm_transform exactly.
        """
        x_t = x_norm * self.dhm_std + self.dhm_mean
        if self.dhm_transform == "signed_log1p":
            return torch.sign(x_t) * (torch.expm1(torch.abs(x_t)) * self.dhm_log_scale)
        return x_t

    def _pattern_loss(self, x0_pred, target_dhm, valid):
        """Multi-scale binary-pattern matching loss.

        Soft-binarises predicted/observed Δhm at each threshold (in raw units),
        then compares average-pooled binary maps at multiple scales. Pooling
        makes this insensitive to pixel-level alignment; the model has to put
        the right *amount* of high-change in roughly the right *places* within
        each tile.
        """
        if self.pattern_loss_weight <= 0 or not self.pattern_thresholds:
            return torch.zeros((), device=x0_pred.device)
        # Inverse-transform back to raw Δhm space so thresholds (0.05, 0.4) are
        # interpretable as actual change magnitudes.
        x0_raw = self.denormalize(x0_pred)
        target_raw = self.denormalize(target_dhm)
        valid_f = valid.float()
        terms = []
        for T in self.pattern_thresholds:
            soft = torch.sigmoid((x0_raw - T) / self.pattern_temperature) * valid_f
            hard = (target_raw > T).float() * valid_f
            for s in self.pattern_scales:
                # Skip scales that don't fit
                if soft.shape[-1] < s or soft.shape[-2] < s:
                    continue
                p = F.avg_pool2d(soft, s, s)
                o = F.avg_pool2d(hard, s, s)
                terms.append(F.mse_loss(p, o))
        if not terms:
            return torch.zeros((), device=x0_pred.device)
        return torch.stack(terms).mean()

    def _denoising_step(
        self,
        batch: Mapping[str, Any],
        log_prefix: str,
    ) -> torch.Tensor:
        cond = self._build_conditioning_from_batch(batch, drop_m=True)
        target = batch["target_dhm"]               # [B, 1, H, W]
        valid = batch["valid_mask"].unsqueeze(1)   # [B, 1, H, W]
        B = target.shape[0]
        # CFG: occasionally drop the *whole* spatial conditioning so the model
        # also learns the unconditional score (independent of magnitude
        # dropout, which lives in _build_conditioning_from_batch).
        if self.training and self.cfg_dropout_prob > 0:
            keep = (torch.rand(B, device=cond.device) > self.cfg_dropout_prob).float()
            cond = cond * keep.view(-1, 1, 1, 1)

        # CorrDiff residual decomposition (Tier 3A): if a deterministic mean
        # head is enabled, train it with MSE on (μ, target) and have the
        # diffusion learn the residual r = target − μ.detach() instead of the
        # full target. Frees the diffusion's stochastic capacity from
        # re-modelling the conditional mean.
        mean_loss_term = None
        if self.use_mean_head and self.mean_head is not None:
            mu = self.mean_head(cond)                # [B, 1, H, W]
            valid_f = valid.float()
            sq = (mu - target).pow(2)
            if self.mean_head_pixel_weight_alpha > 0:
                # Up-weight rare high-magnitude pixels so the mean head learns
                # to predict their actual values rather than being averaged out
                # by the dominant near-zero bulk. Re-normalised so average pixel
                # weight = 1, preserving loss scale.
                w = target.abs().pow(self.mean_head_pixel_weight_alpha) \
                    + self.mean_head_pixel_weight_eps
                w_sum = (w * valid_f).sum()
                v_sum = valid_f.sum().clamp(min=1.0)
                w = w * (v_sum / w_sum.clamp(min=1e-8))
                mean_loss_term = (sq * w * valid_f).sum() / v_sum
            else:
                denom = valid_f.sum().clamp(min=1.0)
                mean_loss_term = (sq * valid_f).sum() / denom
            x_0 = target - mu.detach()
        else:
            mu = None
            x_0 = target

        t = torch.randint(
            0, self.num_train_timesteps, (B,), device=x_0.device, dtype=torch.long
        )
        noise = torch.randn_like(x_0)
        noisy = self.train_scheduler.add_noise(x_0, noise, t)
        v_target = self.train_scheduler.get_velocity(x_0, noise, t)
        v_pred = self.unet(noisy, cond, t)

        v_loss = self._v_loss(v_pred, v_target, valid, x_0, t)
        total_loss = v_loss
        self.log(f"{log_prefix}/v_loss", v_loss, on_step=True, on_epoch=True)
        if mean_loss_term is not None:
            total_loss = total_loss + self.mean_loss_weight * mean_loss_term
            self.log(f"{log_prefix}/mean_loss", mean_loss_term, on_step=True, on_epoch=True)

        need_x0 = (self.pattern_loss_weight > 0
                   or self.wasserstein_loss_weight > 0
                   or self.tile_mean_loss_weight > 0
                   or self.hist_loss_weight > 0
                   or self.tv_loss_weight > 0)
        x0_pred = self._x0_pred_from_v(noisy, v_pred, t) if need_x0 else None
        # Pattern / Wasserstein losses must operate in *target space* (not
        # residual space) so their thresholds and histograms correspond to
        # actual Δhm magnitudes. With the mean head on, x_0 is the residual,
        # so we add μ.detach() back to recover the full prediction & target.
        if x0_pred is not None and mu is not None:
            x0_pred_full = x0_pred + mu.detach()
            target_for_loss = target
        else:
            x0_pred_full = x0_pred
            target_for_loss = x_0

        if self.pattern_loss_weight > 0:
            p_loss = self._pattern_loss(x0_pred_full, target_for_loss, valid)
            total_loss = total_loss + self.pattern_loss_weight * p_loss
            self.log(f"{log_prefix}/pattern_loss", p_loss, on_step=True, on_epoch=True)

        if self.wasserstein_loss_weight > 0:
            w_loss = self._marginal_wasserstein_loss(x0_pred_full, target_for_loss, valid)
            total_loss = total_loss + self.wasserstein_loss_weight * w_loss
            self.log(f"{log_prefix}/wasserstein_loss", w_loss, on_step=True, on_epoch=True)

        if self.tile_mean_loss_weight > 0:
            tm_loss = self._tile_mean_loss(x0_pred_full, target_for_loss, valid)
            total_loss = total_loss + self.tile_mean_loss_weight * tm_loss
            self.log(f"{log_prefix}/tile_mean_loss", tm_loss, on_step=True, on_epoch=True)

        if self.hist_loss_weight > 0:
            h_loss = self._per_tile_hist_loss(x0_pred_full, target_for_loss, valid)
            total_loss = total_loss + self.hist_loss_weight * h_loss
            self.log(f"{log_prefix}/hist_loss", h_loss, on_step=True, on_epoch=True)

        if self.tv_loss_weight > 0:
            tv = self._tv_loss(x0_pred_full, target_for_loss, valid)
            total_loss = total_loss + self.tv_loss_weight * tv
            self.log(f"{log_prefix}/tv_loss", tv, on_step=True, on_epoch=True)

        self.log(f"{log_prefix}/loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        return total_loss

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
        guidance_scale: float = 1.0,
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
            do_cfg = guidance_scale != 1.0
            zero_cond = torch.zeros_like(cond_tiled) if do_cfg else None
            x = torch.randn(
                n_samples * B, 1, H, W, device=conditioning.device,
                dtype=conditioning.dtype, generator=generator,
            )
            for t in scheduler.timesteps:
                t_batch = t.expand(x.shape[0]).to(x.device)
                if do_cfg:
                    v_cond = self.unet(x, cond_tiled, t_batch)
                    v_uncond = self.unet(x, zero_cond, t_batch)
                    v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_pred = self.unet(x, cond_tiled, t_batch)
                x = scheduler.step(v_pred, t, x).prev_sample
            samples = x.view(n_samples, B, 1, H, W)
            # CorrDiff: outputs are residuals; add μ to recover full prediction
            if self.use_mean_head and self.mean_head is not None:
                with torch.no_grad():
                    mu = self.mean_head(conditioning)             # [B, 1, H, W]
                samples = samples + mu.unsqueeze(0)               # broadcast over N
            return samples
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
