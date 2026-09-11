import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
from .spatiotemporal_predictor import SpatioTemporalPredictor
from .losses import LaplacianPyramidLoss
from .crps_loss import _masked_mean, crps_spline, crps_zspace, nll_spline
from .histogram_loss import HistogramLoss
from .quantile_spline import knot_preset, rebuild, splines_from_output
from .pinball_loss import PinballLoss
from .change_weights import (  # noqa: F401
    N_CONTEXT_CHANNELS,
    class_balanced_weights,
    past_change_from_inputs,
    quantile_context,
    quantile_context_from_distance,
)
import wandb
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class SpatioTemporalLightningModule(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 16,
        lr: float = 1e-3,
        ssim_weight: float = 2.0,
        laplacian_weight: float = 1.0,
        num_static_channels: int = 1,
        num_dynamic_channels: int = 1,
        num_layers: int = 1,
        kernel_size: int = 3,
        use_location_encoder: bool = True,
        locenc_backbone=("sphericalharmonics", "siren"),
        locenc_hparams=None,
        locenc_out_channels: int = 8,
        locenc_legendre_polys: int = 10,
        histogram_weight: float = 0.67,
        histogram_lambda_w2: float = 0.1,
        histogram_warmup_epochs: int = 20,
        quantile_class_weighting: str = 'none',
        context_channels: int = 0,
        quantile_weight_change_bins: bool = True,
        freeze_trunk: bool = False,
        central_residual: bool = False,
        monotone_quantile_width: bool = False,
        quantile_dhat_context: bool = False,
        width_parameterisation: str = 'softplus',
        convlstm_dilations=None,
        horizon_loss_weights=None,
        loss_on_change: bool = False,
        pinball_scale_norm: bool = False,
        lr_schedule: str = 'none',
        lr_warmup_frac: float = 0.05,
        lr_min_frac: float = 0.01,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        weight_avg_last: int = 0,
        abort_on_nonfinite: bool = True,
        head_hidden_layers: int = 1,
        width_head_mode: str = 'per_horizon',
        central_target_transform: str = 'none',
        quantile_loss: str = 'pinball',
        histogram_soft: bool = False,
        head_family: str = 'triple',
        dist_loss: str = 'crps',
        crps_nodes: int = 6,
        crps_tail_lam: float = 0.0,
        crps_tail_u0: float = 0.95,
        crps_tail_p: float = 2.0,
        mu_mse_weight: float = 1.0,
        spline_learn_slopes: bool = True,
        spline_cumulative_width: bool = True,
        spline_mean_nodes: int = 8,
        spline_checkpoint: bool = True,
        chip_weight_correct: bool = True,
        isolate_shape_grad: bool = False,
        spline_knots: str = 'default14',
        crps_z_weight: float = 0.0,
        crps_z_scale: float = 0.001,
        spline_gap_floor: bool = False,
        crps_tail_lam_lo: float = 0.0,
        crps_tail_u0_lo: float = 0.05,
        shape_head_hidden_layers: int = 1,
        shape_head_width: int = 0,
        context_radii=None,
        hm_context_stats=(),
        hm_context_radii=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Enable manual optimization to separately handle pinball loss gradients
        self.automatic_optimization = False
        
        # Build LocationEncoder hparams if not provided
        if locenc_hparams is None and use_location_encoder:
            locenc_hparams = {
                'legendre_polys': locenc_legendre_polys,
                'dim_hidden': 64,
                'num_layers': 2,
                'optimizer': {'lr': 1e-4, 'wd': 1e-3},
                'num_classes': locenc_out_channels
            }
        
        self.model = SpatioTemporalPredictor(
            hidden_dim=hidden_dim,
            num_static_channels=num_static_channels,
            num_dynamic_channels=num_dynamic_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            use_location_encoder=use_location_encoder,
            locenc_backbone=locenc_backbone,
            locenc_hparams=locenc_hparams,
            locenc_out_channels=locenc_out_channels,
            context_channels=context_channels,
            central_residual=central_residual,
            monotone_quantile_width=monotone_quantile_width,
            quantile_dhat_context=quantile_dhat_context,
            width_parameterisation=width_parameterisation,
            convlstm_dilations=convlstm_dilations,
            head_hidden_layers=head_hidden_layers,
            width_head_mode=width_head_mode,
            central_target_transform=central_target_transform,
            head_family=head_family,
            spline_learn_slopes=spline_learn_slopes,
            spline_cumulative_width=spline_cumulative_width,
            spline_mean_nodes=spline_mean_nodes,
            spline_checkpoint=spline_checkpoint,
            spline_gap_floor=spline_gap_floor,
            spline_u_knots=knot_preset(spline_knots),
            shape_head_hidden_layers=shape_head_hidden_layers,
            shape_head_width=shape_head_width,
        )
        # Context geometry. Defaults reproduce round 1's eight channels exactly, so the
        # stability gate still has a comparable baseline.
        from .change_weights import CONTEXT_RADII, HM_CONTEXT_RADII
        self.context_radii = tuple(context_radii) if context_radii else CONTEXT_RADII
        self.hm_context_stats = tuple(hm_context_stats or ())
        self.hm_context_radii = tuple(hm_context_radii) if hm_context_radii else HM_CONTEXT_RADII
        # The distributional head: one quantile function per pixel, trained end to end, with
        # no post-hoc width calibration or marginal reshaping behind it.
        self.head_family = str(head_family)
        if dist_loss not in ('crps', 'nll'):
            raise ValueError(f"unknown dist_loss {dist_loss!r}")
        self.dist_objective = str(dist_loss)
        self.crps_nodes = int(crps_nodes)
        self.crps_tail_lam = float(crps_tail_lam)
        self.crps_tail_u0 = float(crps_tail_u0)
        self.crps_tail_p = float(crps_tail_p)
        self.crps_tail_lam_lo = float(crps_tail_lam_lo)
        # E3. CRPS in raw HM units is dominated by the tail: the persistence core
        # spans ~0.0036 HM against a range above 1.2, so placing the core badly
        # costs almost nothing -- and 68% of land is in that core. Adding a term
        # scored in z = asinh(change / s) makes a 0.0005 error near zero weigh
        # about what a 0.05 error in the tail does. Summed, not substituted, so
        # the tail is not abandoned to fix the body.
        self.crps_z_weight = float(crps_z_weight)
        self.crps_z_scale = float(crps_z_scale)
        self.crps_tail_u0_lo = float(crps_tail_u0_lo)
        # Weight on MSE(E[Q], y). CRPS shapes the whole distribution but presses on its mean
        # only indirectly, and the published central forecast *is* that mean, so this term is
        # the hedge against losing central skill. Setting it to 0 is a deliberate ablation.
        self.mu_mse_weight = float(mu_mse_weight)
        self.spline_checkpoint = bool(spline_checkpoint)
        # Importance correction for stratified chip sampling. Reweighting *samples* moves the
        # target distribution, unlike reweighting quantile levels, so without this the model
        # is fitting a utility-weighted world rather than the real one.
        self.chip_weight_correct = bool(chip_weight_correct)
        self.isolate_shape_grad = bool(isolate_shape_grad)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.mae_fn = nn.L1Loss(reduction='mean')
        # Quantile losses
        self.pinball_lower = PinballLoss(quantile=0.025, reduction='mean')
        self.pinball_upper = PinballLoss(quantile=0.975, reduction='mean')
        # 'none' reproduces the pooled loss exactly. 'distance' balances the
        # distance-to-past-change bands, which is what stops the upper head from claiming
        # a possible large gain in country where change never happens.
        self.quantile_class_weighting = quantile_class_weighting
        self.quantile_weight_change_bins = bool(quantile_weight_change_bins)
        self.freeze_trunk = bool(freeze_trunk)
        self._quantile_weights = None
        self.context_channels = int(context_channels)
        # Every distributional head family. Defined once: three separate
        # `head_family == 'spline'` comparisons is exactly how 'pwl' silently took
        # the triple-head two-pass path and arrived at backward with no graph.
        self.is_distributional = head_family in ('spline', 'pwl', 'isqf')
        self.central_residual = bool(central_residual)
        self.monotone_quantile_width = bool(monotone_quantile_width)
        self.quantile_dhat_context = bool(quantile_dhat_context)
        # Training exposure is *not* uniform across horizons. end_year is sampled from
        # (2000, 2005, 2010, 2015) and targets past 2020 are NaN, so h=5 gets a target from
        # all four windows, h=10 from three, h=15 from two and h=20 from one — a 4:3:2:1
        # gradient budget, with the largest-error horizon getting the least. Validation
        # uses fixed years and is balanced, so nothing in val_* shows this.
        # None reproduces the uniform average exactly.
        if horizon_loss_weights is None:
            self.horizon_loss_weights = None
        else:
            w = [float(x) for x in horizon_loss_weights]
            if len(w) != 4:
                raise ValueError(f"horizon_loss_weights needs 4 entries, got {len(w)}")
            # Renormalised to mean 1, so the loss scale — and the learning rate tuned
            # against it — is unchanged; only the balance between horizons moves.
            s = sum(w) / len(w)
            self.horizon_loss_weights = [x / s for x in w]
        # SSIM and the Laplacian pyramid are computed on absolute HM, which is what the
        # central head used to predict. Under --central_residual the prediction is
        # HM_t0 + a small change, so both terms are dominated by the copy. This computes
        # them on the change field instead.
        self.loss_on_change = bool(loss_on_change)
        # Divide each pixel's pinball loss by that pixel's own (detached) interval width,
        # so far-field calibration contributes gradient at all. NOT inverse-frequency class
        # weighting, which is recorded in CLAUDE.md as backwards for this problem: this
        # equalises loss magnitude per pixel, not per class.
        self.pinball_scale_norm = bool(pinball_scale_norm)
        self.lr_schedule = str(lr_schedule)
        self.lr_warmup_frac = float(lr_warmup_frac)
        self.lr_min_frac = float(lr_min_frac)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        # Average the weights of the last N epochs instead of picking one of them.
        # Measured motivation: on fold 1 the epoch ModelCheckpoint selected was 140, 85 and
        # 60 across three seeds of the same configuration, while the best 10% of epochs sat
        # within 3% of the minimum on the monitored metric — so the argmin is close to
        # arbitrary among ~15 near-tied candidates, and those candidates produce materially
        # different central fields. 0 (the default) keeps today's single-epoch selection.
        self.weight_avg_last = int(weight_avg_last)
        self._wa_sum = None
        self._wa_count = 0
        # Refuse to keep training a diverged model, and refuse to publish one. See
        # on_train_epoch_end for why silence here corrupted a whole round of results.
        self.abort_on_nonfinite = bool(abort_on_nonfinite)
        # 'pinball' (default) fits the 2.5 and 97.5 percentiles independently. 'nll' instead
        # treats (lower, central, upper) as a two-piece normal and fits it by log score, so
        # the widths become scale parameters of a density rather than two quantiles — a
        # different estimand, and one dominated by the body rather than the tails.
        #
        # NOT offered: the Winkler interval score. It equals 2/alpha times the sum of the
        # two pinball losses exactly (checked algebraically and in
        # tests/test_round2_flags.py), so at alpha = 0.05 it is the pinball objective with a
        # 40x learning rate and cannot move the optimum.
        if quantile_loss not in ('pinball', 'nll'):
            raise ValueError(f"unknown quantile_loss {quantile_loss!r}")
        self.quantile_loss = str(quantile_loss)
        self.histogram_soft = bool(histogram_soft)
        self.lr = lr
        self.ssim_weight = ssim_weight
        self.laplacian_weight = laplacian_weight
        # 3-level Laplacian pyramid by default
        self.lap_loss = LaplacianPyramidLoss(levels=3, kernel_size=5, sigma=1.0, include_lowpass=True)
        # Normalization stats to be set externally
        self.hm_mean = None
        self.hm_std = None
        # Histogram loss for pixel-level change distributions
        self.histogram_weight = histogram_weight
        self.histogram_warmup_epochs = histogram_warmup_epochs
        self.histogram_lambda_w2 = histogram_lambda_w2  # Store for reference (not used in new implementation)
        if self.histogram_weight > 0:
            # Define histogram bins: 8 bins from decrease to extreme increase
            # Bin 1: decrease (<-0.005), Bin 2: no change (-0.005 to 0.005), 
            # Bin 3: tiny increase (0.005-0.02), Bin 4: small (0.02-0.1),
            # Bin 5: moderate (0.1-0.2), Bin 6: large (0.2-0.4),
            # Bin 7: very large (0.4-0.6), Bin 8: extreme (>0.6)
            histogram_bins = torch.tensor([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
            self.histogram_loss_fn = HistogramLoss(histogram_bins)  # Bin weights will be set later
            self.register_buffer('histogram_bins', histogram_bins)
            self.histogram_bins_initialized = False

    # Removed AR-specific helpers and panels for single-step setup
    
    def _compute_horizon_losses(self, pred_lower, pred_central, pred_upper, target_h, last_input, mask_h, horizon_name="", spline=None, sample_weight=None):
        """
        Compute all loss components for a single horizon with independent predictions.
        
        Loss Assignment (SEPARATED GRADIENT FLOW):
        - Central prediction: MSE + SSIM + Laplacian + Histogram (multi-objective optimization)
          → Gradients backprop through: ConvLSTM backbone + central_heads
        - Lower/Upper quantiles: Pinball loss ONLY (pure uncertainty estimation)
          → Gradients backprop through: lower_heads/upper_heads ONLY (NOT backbone)
        
        The manual optimization in training_step ensures:
        - Pinball loss never affects ConvLSTM or central_heads parameters
        - Central losses never affect quantile head parameters
        - Backbone learns from accuracy/spatial quality objectives only
        - Quantile heads learn from calibration objectives only
        
        Args:
            pred_lower: [B, 1, H, W] lower quantile (2.5%) predictions
            pred_central: [B, 1, H, W] central prediction (optimized for accuracy + spatial patterns)
            pred_upper: [B, 1, H, W] upper quantile (97.5%) predictions
            target_h: [B, 1, H, W] targets for this horizon
            last_input: [B, 1, H, W] last input timestep
            mask_h: [B, 1, H, W] validity mask
            horizon_name: str, for logging (e.g., "5yr")
            
        Returns:
            dict with keys: mse, mae, ssim, lap, hist, pinball_lower, pinball_upper, total
        """
        # Deltas (use central for MSE and other spatial losses)
        delta_central = pred_central - last_input
        delta_true = target_h - last_input
        valid_delta_central = delta_central[mask_h]
        valid_delta_true = delta_true[mask_h]
        
        if valid_delta_central.numel() == 0:
            return {
                'mse': torch.tensor(0.0, device=pred_central.device),
                'mae': torch.tensor(0.0, device=pred_central.device),
                'ssim': torch.tensor(0.0, device=pred_central.device),
                'lap': torch.tensor(0.0, device=pred_central.device),
                'hist': torch.tensor(0.0, device=pred_central.device),
                'pinball_lower': torch.tensor(0.0, device=pred_central.device),
                'pinball_upper': torch.tensor(0.0, device=pred_central.device),
                'dist': torch.tensor(0.0, device=pred_central.device),
                'central': torch.tensor(0.0, device=pred_central.device),
                'total': torch.tensor(0.0, device=pred_central.device)
            }
        
        # MSE on deltas (CENTRAL ONLY - independent from quantiles)
        if sample_weight is None:
            mse = self.loss_fn(valid_delta_central, valid_delta_true)
        else:
            # Weighted before the reduction, not after: the predictions are NaN outside the
            # mask and NaN * 0 is still NaN.
            sq = torch.where(mask_h, (delta_central - delta_true) ** 2,
                             torch.zeros_like(delta_true))
            w = sample_weight * mask_h.to(sq.dtype)
            mse = (sq * w).sum() / w.sum().clamp_min(1e-12)
        
        # MAE on absolute values (CENTRAL ONLY)
        mae = self.mae_fn(pred_central[mask_h], target_h[mask_h])
        
        # Pinball losses for quantiles (INDEPENDENT - only affect quantile heads)
        qw = self._quantile_weights
        if qw is not None and qw.shape != pred_lower.shape:
            qw = None
        w_lo = w_up = qw
        if self.pinball_scale_norm:
            # 1 / (that pixel's own half-width), detached. The pinball gradient with respect
            # to a bound has constant magnitude, so in absolute HM units the far field —
            # whose widths are ~18x smaller — contributes almost nothing to the loss and its
            # calibration is never learned. Dividing by the width makes the objective a
            # *relative* one. Detached, so it is a weight and creates no incentive to shrink.
            floor = 1e-3
            s_lo = (pred_central - pred_lower).detach().abs().clamp(min=floor)
            s_up = (pred_upper - pred_central).detach().abs().clamp(min=floor)
            w_lo = (1.0 / s_lo) if qw is None else qw / s_lo
            w_up = (1.0 / s_up) if qw is None else qw / s_up
        dist_loss = torch.tensor(0.0, device=pred_central.device)
        if spline is not None:
            # NaN is the validity mask in this codebase, and NaN * 0 is still NaN: a dense
            # computation has to be masked *before* the arithmetic, not after. Substituting
            # the anchor at invalid pixels keeps every intermediate finite; the mask then
            # removes those pixels from the mean. Masking afterwards instead cost this
            # project a 35-minute run that came back all-NaN.
            y = torch.where(mask_h, target_h, pred_central.detach()).squeeze(1)
            y = torch.nan_to_num(y, nan=0.0)
            # A float weight rather than a bool: _masked_mean computes (x*m).sum()/m.sum(),
            # which is already the importance-corrected estimator when m carries the weight.
            m = (mask_h if sample_weight is None
                 else mask_h.to(target_h.dtype) * sample_weight).squeeze(1)
            hm0 = last_input.squeeze(1)
            if self.head_family in ('pwl', 'isqf'):
                # Closed form: no quadrature, no nodes, nothing to checkpoint. This is the
                # whole point of the C0 families -- see src/models/quantile_pwl.py.
                dist_loss = _masked_mean(spline.crps(y), m)
                if self.crps_z_weight > 0:
                    dist_loss = dist_loss + self.crps_z_weight * _masked_mean(
                        spline.crps_z(y, hm0, self.crps_z_scale), m)
            else:
                def _dist(anchor, scale, v_knots, derivs, yy):
                    sp = rebuild(anchor, scale, v_knots, derivs, spline.u_knots, spline.clamp)
                    if self.dist_objective == 'nll':
                        return nll_spline(sp, yy, mask=m)
                    out = crps_spline(sp, yy, mask=m, n_nodes=self.crps_nodes,
                                      tail_lam=self.crps_tail_lam,
                                      tail_u0=self.crps_tail_u0, tail_p=self.crps_tail_p,
                                      tail_lam_lo=self.crps_tail_lam_lo,
                                      tail_u0_lo=self.crps_tail_u0_lo)
                    if self.crps_z_weight > 0:
                        out = out + self.crps_z_weight * crps_zspace(
                            sp, yy, hm0, self.crps_z_scale, mask=m, n_nodes=self.crps_nodes)
                    return out

                args_ = (spline.anchor, spline.scale, spline.v_knots, spline.derivs, y)
                if self.spline_checkpoint and torch.is_grad_enabled():
                    dist_loss = torch.utils.checkpoint.checkpoint(_dist, *args_,
                                                                  use_reentrant=False)
                else:
                    dist_loss = _dist(*args_)
            pinball_lower = pinball_upper = torch.tensor(0.0, device=pred_central.device)
        elif self.quantile_loss == 'nll':
            pinball_lower, pinball_upper = self._two_piece_nll(
                pred_lower, pred_central, pred_upper, target_h, mask_h)
        else:
            pinball_lower = self.pinball_lower(pred_lower, target_h, mask=mask_h, weights=w_lo)
            pinball_upper = self.pinball_upper(pred_upper, target_h, mask=mask_h, weights=w_up)

        # SSIM (CENTRAL ONLY - independent from quantiles). On absolute HM by default; on
        # the change field when loss_on_change, since under --central_residual the absolute
        # prediction is HM_t0 plus a small change and both terms mostly score the copy.
        if self.loss_on_change:
            pred_sanitized = delta_central.clone()
            target_sanitized = delta_true.clone()
        else:
            pred_sanitized = pred_central.clone()
            target_sanitized = target_h.clone()
        pred_sanitized[~mask_h] = 0.0
        target_sanitized[~mask_h] = 0.0
        ssim_val = ssim(pred_sanitized, target_sanitized, data_range=1.0)
        ssim_loss = 1.0 - ssim_val
        
        # Laplacian loss (CENTRAL ONLY - independent from quantiles)
        lap_loss = self.lap_loss(pred_sanitized, target_sanitized, mask=mask_h)
        
        # Histogram loss (CENTRAL ONLY - independent from quantiles)
        hist_loss = torch.tensor(0.0, device=pred_central.device)
        if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
            delta_true_2d = delta_true.squeeze(1)
            delta_central_2d = delta_central.squeeze(1)
            mask_2d = mask_h.squeeze(1)
            # Extract horizon index from horizon_name (e.g., "5yr" -> 0)
            horizon_map = {'5yr': 0, '10yr': 1, '15yr': 2, '20yr': 3}
            h_idx = horizon_map.get(horizon_name, 0)
            hist_loss, _, _ = self.histogram_loss_fn(delta_true_2d, delta_central_2d,
                                                     mask=mask_2d, horizon_idx=h_idx,
                                                     soft=self.histogram_soft)
        
        # Total loss for this horizon
        # INDEPENDENT GRADIENTS: central gets MSE+SSIM+Lap+Hist, quantiles get pinball only
        if spline is not None:
            # `pred_central` is already E[Q], so MSE, SSIM and the Laplacian pyramid act on
            # the distributional mean without any extra plumbing -- and on absolute HM, which
            # is the side of that choice the measurement supports (E3a moved the spatial
            # terms onto the change field and lost h=5/h=10 RMSE, because the change field is
            # near-zero almost everywhere and SSIM's local normalisation collapses there).
            central = (self.mu_mse_weight * mse
                       + self.ssim_weight * ssim_loss
                       + self.laplacian_weight * lap_loss)
            total = central + dist_loss
        else:
            central = mse + self.ssim_weight * ssim_loss + self.laplacian_weight * lap_loss
            total = central + pinball_lower + pinball_upper
        if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
            # Kept in `total` for backward compatibility of val_total_loss, and deliberately
            # kept OUT of `central`: compute_histogram bins with boolean comparisons into a
            # plain zeros buffer, so this term carries no gradient at all and has never
            # trained anything. See tests/test_histogram_loss_gradient.py.
            total = total + self.histogram_weight * hist_loss

        return {
            'mse': mse,
            'mae': mae,
            'ssim': ssim_loss,
            'lap': lap_loss,
            'hist': hist_loss,
            'pinball_lower': pinball_lower,
            'pinball_upper': pinball_upper,
            'dist': dist_loss,
            'central': central,
            'total': total
        }


    # 1.96: the published bounds are the 2.5/97.5 percentiles, so half-width / Z975 is the
    # scale of the normal that would place them there.
    Z975 = 1.959963985

    def _two_piece_nll(self, pred_lower, pred_central, pred_upper, target_h, mask_h):
        """Negative log likelihood of the observation under a two-piece normal.

        The published triple already *is* a two-piece normal's median and bounds, so fitting
        it by log score is a strictly different objective on the same parameterisation
        rather than a different product. The centre is detached: the central head keeps its
        RMSE-optimal target, which docs/background/next_phase_marginals.md section 6.2 measured as
        costing ~10% of skill to move.

        Returned split in two so the existing per-side logging keeps working; the halves are
        the below-centre and above-centre contributions.
        """
        c = pred_central.detach()
        s_lo = ((c - pred_lower) / self.Z975).clamp(min=1e-4)
        s_up = ((pred_upper - c) / self.Z975).clamp(min=1e-4)
        e = target_h - c
        below = e < 0
        s = torch.where(below, s_lo, s_up)
        # Two-piece normal density: 2/(s_lo+s_up) * phi(e/s). The normaliser couples the two
        # sides, which is what makes this a density rather than two independent quantiles.
        nll = 0.5 * (e / s) ** 2 + torch.log(s_lo + s_up)
        m = mask_h & torch.isfinite(nll)
        if m.sum() == 0:
            z = torch.zeros((), device=pred_central.device)
            return z, z
        lo_m = m & below
        up_m = m & ~below
        half = lambda sel: nll[sel].mean() if sel.sum() > 0 else torch.zeros((), device=nll.device)
        return half(lo_m), half(up_m)

    def _horizon_mean(self, horizon_losses, key):
        """Average one loss component across horizons, optionally weighted.

        ``horizon_loss_weights`` is renormalised to mean 1 in __init__, so with weights off
        this is exactly ``.mean()`` and with them on the total loss scale is preserved.
        """
        vals = torch.stack([h[key] for h in horizon_losses])
        if self.horizon_loss_weights is None:
            return vals.mean()
        w = torch.as_tensor(self.horizon_loss_weights, device=vals.device, dtype=vals.dtype)
        return (vals * w).mean()

    def _compute_quantile_weights(self, input_dynamic, target_h, mask):
        """Per-pixel weights that balance the distance-to-past-change bands.

        Returns None when weighting is off, which reproduces the pooled loss exactly.
        """
        if self.quantile_class_weighting in (None, 'none'):
            return None
        past = past_change_from_inputs(input_dynamic, hm_std=float(getattr(self, 'hm_std', 1.0)))
        change = None
        if self.quantile_weight_change_bins and target_h is not None:
            last = input_dynamic[:, -1, 0:1]
            change = (target_h - last) * float(getattr(self, 'hm_std', 1.0))
        return class_balanced_weights(past, mask, target_change=change)

    def _build_context(self, input_dynamic, change_context=None, hm_context=None):
        """The neighbourhood covariate the trunk consumes: past-change occupancy and
        distance to past change, plus the neighbourhood HM summaries.

        Prefers the precomputed full-raster context (band 1 past change, band 2 distance);
        falls back to the chip-local dilation only when none is supplied, which is correct
        just for chips much larger than the biggest radius.

        There is one consumer and one tensor. It goes into the trunk beside elevation and
        climate; no head receives it.
        """
        if self.context_channels <= 0:
            return None
        hm_now = input_dynamic[:, -1, 0:1]
        if change_context is not None:
            past = change_context[:, 0:1]
            dist = change_context[:, 1:2]
            return quantile_context_from_distance(
                dist, past, hm_now=hm_now, radii=self.context_radii,
                hm_context=hm_context, hm_stats=self.hm_context_stats,
                hm_radii=self.hm_context_radii)
        hm_std = float(getattr(self, 'hm_std', 1.0))
        past = past_change_from_inputs(input_dynamic, hm_std=hm_std)
        return quantile_context(past, hm_now=hm_now)

    def _splines(self, preds_all):
        """Per-horizon quantile functions from the trailing block of the model output.

        Returns ``None`` for the triple head, so every call site is a plain ``spline=...``
        keyword and there is no second code path to keep in step.
        """
        if not self.is_distributional:
            return [None] * 4
        return self.model._decode(preds_all, self.model.spline_clamp(), n_triple=12)

    def forward(self, input_dynamic, input_static, lonlat=None, change_context=None,
                hm_context=None):
        # Ensure input_dynamic is [B, T, 1, H, W]
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        ctx = self._build_context(input_dynamic, change_context, hm_context)
        return self.model(input_dynamic, input_static, lonlat=lonlat, context=ctx)

    def training_step(self, batch, batch_idx):
        # Get optimizer (manual optimization)
        opt = self.optimizers()
        
        input_dynamic = batch['input_dynamic']
        input_static = batch['input_static']
        # Multi-horizon targets
        targets = {
            '5yr': batch['target_5yr'].unsqueeze(1),
            '10yr': batch['target_10yr'].unsqueeze(1),
            '15yr': batch['target_15yr'].unsqueeze(1),
            '20yr': batch['target_20yr'].unsqueeze(1)
        }
        lonlat = batch.get('lonlat', None)
        if lonlat is not None:
            lonlat = lonlat.to(input_dynamic.device)
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        
        # Compute validity mask from RAW inputs
        dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
        static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
        
        # Replace NaNs in inputs with 0 for model forward
        # Note: 0 in normalized space = mean in original space (mean imputation)
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        
        # Forward pass: get predictions for all 4 horizons × 3 quantiles [B, 12, H, W]
        preds_all = self(input_dynamic, input_static, lonlat=lonlat,
                         change_context=batch.get('change_context'),
                         hm_context=batch.get('hm_context'))
        
        # Use last timestep HM channel (0) as baseline
        last_input = input_dynamic[:, -1, 0:1, :, :]  # [B, 1, H, W]
        
        splines = self._splines(preds_all)
        cw = batch.get('chip_weight')
        sample_w = (cw.to(input_dynamic.dtype).view(-1, 1, 1, 1)
                    if (cw is not None and self.chip_weight_correct) else None)

        # Compute losses for each horizon
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_losses = []
        
        for h_idx, h_name in enumerate(horizon_names):
            # Extract 3 independent predictions for this horizon
            # Channel ordering: [lower_5yr, central_5yr, upper_5yr, lower_10yr, ...]
            pred_lower = preds_all[:, 3*h_idx:3*h_idx+1, :, :]  # [B, 1, H, W]
            pred_central = preds_all[:, 3*h_idx+1:3*h_idx+2, :, :]  # [B, 1, H, W]
            pred_upper = preds_all[:, 3*h_idx+2:3*h_idx+3, :, :]  # [B, 1, H, W]
            target_h = targets[h_name]  # [B, 1, H, W]
            
            # Compute mask for this horizon
            target_valid = torch.isfinite(target_h)
            mask_h = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2) & torch.isfinite(last_input)
            
            # Set predictions to NaN where inputs were invalid
            pred_lower = pred_lower.clone()
            pred_central = pred_central.clone()
            pred_upper = pred_upper.clone()
            pred_lower[~mask_h] = float('nan')
            pred_central[~mask_h] = float('nan')
            pred_upper[~mask_h] = float('nan')
            
            # Class weights for the quantile heads (None reproduces the pooled loss)
            self._quantile_weights = self._compute_quantile_weights(input_dynamic, target_h, mask_h)

            # Compute all losses for this horizon
            losses_h = self._compute_horizon_losses(pred_lower, pred_central, pred_upper, target_h, last_input, mask_h, h_name, spline=splines[h_idx], sample_weight=sample_w)
            horizon_losses.append(losses_h)
        
        # Log per-horizon metrics
        horizon_suffixes = ['5yr', '10yr', '15yr', '20yr']
        for h_name, losses_h in zip(horizon_suffixes, horizon_losses):
            self.log(f'train_mae_{h_name}', losses_h['mae'])
            self.log(f'train_ssim_loss_{h_name}', losses_h['ssim'])
            self.log(f'train_lap_loss_{h_name}', losses_h['lap'])
            self.log(f'train_hist_loss_{h_name}', losses_h['hist'])
            self.log(f'train_pinball_lower_{h_name}', losses_h['pinball_lower'])
            self.log(f'train_pinball_upper_{h_name}', losses_h['pinball_upper'])
        
        # Average losses across horizons
        avg_mse = self._horizon_mean(horizon_losses, 'mse')
        avg_mae = self._horizon_mean(horizon_losses, 'mae')
        avg_ssim = self._horizon_mean(horizon_losses, 'ssim')
        avg_lap = self._horizon_mean(horizon_losses, 'lap')
        avg_hist = self._horizon_mean(horizon_losses, 'hist')
        avg_pinball_lower = self._horizon_mean(horizon_losses, 'pinball_lower')
        avg_pinball_upper = self._horizon_mean(horizon_losses, 'pinball_upper')
        avg_dist = self._horizon_mean(horizon_losses, 'dist')

        # Compute total losses separately for central vs quantile heads
        # Central loss: MSE + SSIM + Laplacian + Histogram (affects backbone + central heads)
        central_loss = avg_mse + self.ssim_weight * avg_ssim + self.laplacian_weight * avg_lap
        if self.is_distributional:
            central_loss = (self.mu_mse_weight * avg_mse
                            + self.ssim_weight * avg_ssim
                            + self.laplacian_weight * avg_lap)
        if self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs:
            central_loss = central_loss + self.histogram_weight * avg_hist
        
        self._quantile_weights = None

        # Pinball loss: Only affects quantile heads (lower_heads + upper_heads)
        pinball_loss = avg_pinball_lower + avg_pinball_upper

        if self.is_distributional:
            # One model, one objective. The two-pass gradient isolation exists because the
            # pinball loss was a *side* constraint on a product the central head owned; here
            # CRPS is the objective and the trunk is supposed to hear it. `isolate_shape_grad`
            # keeps the old behaviour available as an ablation, because the risk that the
            # distributional gradient degrades the central field is real and untested.
            opt.zero_grad()
            if self.isolate_shape_grad:
                self.manual_backward(central_loss, retain_graph=True)
                shape_params = set(self.model.width_heads.parameters()) | \
                               set(self.model.shape_heads.parameters())
                saved = {n: q.grad.clone() for n, q in self.named_parameters()
                         if q.grad is not None and q not in shape_params}
                self.manual_backward(avg_dist)
                for n, q in self.named_parameters():
                    if n in saved:
                        q.grad = saved[n]
            else:
                self.manual_backward(central_loss + avg_dist)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            opt.step()
            self._quantile_weights = None
            avg_total = central_loss + avg_dist
            self.log('train_loss', avg_mse, prog_bar=True)
            self.log('train_mae_total', avg_mae, prog_bar=True)
            self.log('train_ssim_loss_total', avg_ssim)
            self.log('train_lap_loss_total', avg_lap)
            self.log(f'train_{self.dist_objective}_total', avg_dist, prog_bar=True)
            self.log('train_total_loss', avg_total, prog_bar=True)
            if batch_idx == 0 and self.current_epoch == 0:
                # The architecture fingerprint. "No extra flags" is argparse defaults, not a
                # configuration, and a run launched on the wrong one reads exactly like a real
                # finding -- it cost this project a whole k=5 run once. Under head_family
                # 'spline' the trunk MUST carry a distributional gradient (isolate_shape_grad
                # off) or MUST NOT (on); either way it is visible here in one line.
                def gn(mod):
                    return sum(q.grad.norm().item() ** 2
                               for q in mod.parameters() if q.grad is not None) ** 0.5
                print(f"\n[SPLINE HEAD] epoch 0 batch 0, objective={self.dist_objective}, "
                      f"isolate_shape_grad={self.isolate_shape_grad}")
                print(f"  ConvLSTM grad norm:      {gn(self.model.convlstm):.6f}")
                print(f"  central heads grad norm: {gn(self.model.central_heads):.6f}")
                print(f"  width heads grad norm:   {gn(self.model.width_heads):.6f}")
                print(f"  shape heads grad norm:   {gn(self.model.shape_heads):.6f}")
                print(f"  {self.dist_objective}={avg_dist.item():.6f}  "
                      f"mse={avg_mse.item():.6f}  mae(persistence-scale)={avg_mae.item():.6f}")
                print(f"  support clamp (normalized units): {self.model.spline_clamp()}\n")
            return avg_total

        # MANUAL BACKWARD PASS:
        # Step 1: Backprop central loss through all parameters
        opt.zero_grad()
        if self.freeze_trunk:
            # Head-only retraining: the central objective is not optimised at all, so the
            # trunk and the central heads keep the frozen checkpoint's weights exactly and
            # the published central forecast is unchanged by construction. Only the
            # quantile heads move.
            central_loss = central_loss.detach()
        else:
            self.manual_backward(central_loss, retain_graph=True)

        # Step 2: Backprop pinball loss ONLY through quantile head parameters
        # First, zero out gradients for non-quantile parameters
        quantile_params = set()
        for param in self.model.lower_heads.parameters():
            quantile_params.add(param)
        for param in self.model.upper_heads.parameters():
            quantile_params.add(param)
        
        # Store central loss gradients for non-quantile params
        saved_grads = {}
        for name, param in self.named_parameters():
            if param.grad is not None and param not in quantile_params:
                saved_grads[name] = param.grad.clone()
        
        # Backprop pinball loss (no need to retain graph on second backward)
        self.manual_backward(pinball_loss)
        if self.freeze_trunk:
            # Nothing outside the quantile heads may carry a gradient in this mode.
            for param in self.parameters():
                if param not in quantile_params:
                    param.grad = None
        
        # Restore gradients for non-quantile params (so pinball doesn't affect them)
        for name, param in self.named_parameters():
            if name in saved_grads:
                param.grad = saved_grads[name]

        # Clipping happens after the restore, so it sees the gradients that will actually
        # be applied rather than the pinball pass's intermediate state.
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        # Optimizer step
        opt.step()
        
        # Total loss for logging (not used for backprop)
        avg_total = central_loss + pinball_loss
        
        # Log averaged metrics (total)
        self.log('train_loss', avg_mse, prog_bar=True)
        self.log('train_mae_total', avg_mae, prog_bar=True)
        self.log('train_ssim_loss_total', avg_ssim)
        self.log('train_lap_loss_total', avg_lap)
        self.log('train_hist_loss_total', avg_hist)
        self.log('train_pinball_lower_total', avg_pinball_lower, prog_bar=True)
        self.log('train_pinball_upper_total', avg_pinball_upper, prog_bar=True)
        self.log('train_total_loss', avg_total, prog_bar=True)
        
        # Debug print on first batch of warmup epoch
        if batch_idx == 0 and self.current_epoch == self.histogram_warmup_epochs and self.histogram_weight > 0:
            print(f"\n[HISTOGRAM ACTIVATED] Epoch {self.current_epoch}: avg_hist_loss={avg_hist.item():.6f}, weighted={self.histogram_weight * avg_hist.item():.6f}\n")
        
        # Debug gradient isolation on first epoch (optional)
        if batch_idx == 0 and self.current_epoch == 0:
            # Verify gradient isolation: ConvLSTM should only have central loss grads
            convlstm_grad_norm = sum(p.grad.norm().item()**2 for p in self.model.convlstm.parameters() if p.grad is not None)**0.5
            lower_head_grad_norm = sum(p.grad.norm().item()**2 for p in self.model.lower_heads.parameters() if p.grad is not None)**0.5
            upper_head_grad_norm = sum(p.grad.norm().item()**2 for p in self.model.upper_heads.parameters() if p.grad is not None)**0.5
            central_head_grad_norm = sum(p.grad.norm().item()**2 for p in self.model.central_heads.parameters() if p.grad is not None)**0.5
            print(f"\n[GRADIENT ISOLATION CHECK] Epoch {self.current_epoch}, Batch {batch_idx}:")
            print(f"  ConvLSTM grad norm: {convlstm_grad_norm:.6f} (from central loss only)")
            print(f"  Central heads grad norm: {central_head_grad_norm:.6f} (from central loss only)")
            print(f"  Lower heads grad norm: {lower_head_grad_norm:.6f} (from pinball loss only)")
            print(f"  Upper heads grad norm: {upper_head_grad_norm:.6f} (from pinball loss only)")
            print(f"  Central loss: {central_loss.item():.6f}, Pinball loss: {pinball_loss.item():.6f}\n")
        
        return avg_total

    def validation_step(self, batch, batch_idx):
        input_dynamic = batch['input_dynamic']
        input_static = batch['input_static']
        # Multi-horizon targets
        targets = {
            '5yr': batch['target_5yr'].unsqueeze(1),
            '10yr': batch['target_10yr'].unsqueeze(1),
            '15yr': batch['target_15yr'].unsqueeze(1),
            '20yr': batch['target_20yr'].unsqueeze(1)
        }
        lonlat = batch.get('lonlat', None)
        if lonlat is not None:
            lonlat = lonlat.to(input_dynamic.device)
        if input_dynamic.dim() == 4:
            input_dynamic = input_dynamic.unsqueeze(2)
        
        # Compute validity mask from RAW inputs
        dynamic_valid = torch.isfinite(input_dynamic).all(dim=(1, 2), keepdim=True)
        static_valid = torch.isfinite(input_static).all(dim=1, keepdim=True).unsqueeze(1)
        
        # Replace NaNs in inputs with 0 for model forward
        # Note: 0 in normalized space = mean in original space (mean imputation)
        input_dynamic = torch.nan_to_num(input_dynamic, nan=0.0)
        input_static = torch.nan_to_num(input_static, nan=0.0)
        
        # Forward pass: get predictions for all 4 horizons × 3 quantiles [B, 12, H, W]
        preds_all = self(input_dynamic, input_static, lonlat=lonlat,
                         change_context=batch.get('change_context'),
                         hm_context=batch.get('hm_context'))
        
        # Use last timestep HM channel (0) as baseline
        last_input = input_dynamic[:, -1, 0:1, :, :]  # [B, 1, H, W]
        
        splines = self._splines(preds_all)
        # Validation is never importance-weighted: val_crps has to stay comparable across
        # runs whatever the training sampler did.
        sample_w = None

        # Compute losses for each horizon
        horizon_names = ['5yr', '10yr', '15yr', '20yr']
        horizon_losses = []
        # Coverage tracking for quantiles
        coverage_stats = []
        
        for h_idx, h_name in enumerate(horizon_names):
            # Extract 3 quantile predictions for this horizon
            pred_lower = preds_all[:, 3*h_idx:3*h_idx+1, :, :]  # [B, 1, H, W]
            pred_central = preds_all[:, 3*h_idx+1:3*h_idx+2, :, :]  # [B, 1, H, W]
            pred_upper = preds_all[:, 3*h_idx+2:3*h_idx+3, :, :]  # [B, 1, H, W]
            target_h = targets[h_name]  # [B, 1, H, W]
            
            # Compute mask for this horizon
            target_valid = torch.isfinite(target_h)
            mask_h = target_valid & dynamic_valid.squeeze(2) & static_valid.squeeze(2) & torch.isfinite(last_input)
            
            # Set predictions to NaN where inputs were invalid
            pred_lower = pred_lower.clone()
            pred_central = pred_central.clone()
            pred_upper = pred_upper.clone()
            pred_lower[~mask_h] = float('nan')
            pred_central[~mask_h] = float('nan')
            pred_upper[~mask_h] = float('nan')
            
            # Compute coverage: fraction of targets within [lower, upper] interval
            if mask_h.sum() > 0:
                within_interval = ((target_h >= pred_lower) & (target_h <= pred_upper) & mask_h).sum().float()
                coverage = within_interval / mask_h.sum().float() * 100.0  # Percentage
                coverage_stats.append((h_name, coverage))
            
            # Validation stays *unweighted* whatever the training objective is, so
            # val_pinball_* and val_coverage_* stay comparable across runs and against
            # the production checkpoint.
            self._quantile_weights = None

            # Compute all losses for this horizon
            losses_h = self._compute_horizon_losses(pred_lower, pred_central, pred_upper, target_h, last_input, mask_h, h_name, spline=splines[h_idx], sample_weight=sample_w)
            horizon_losses.append(losses_h)
        
        # Log per-horizon metrics
        horizon_suffixes = ['5yr', '10yr', '15yr', '20yr']
        for h_name, losses_h in zip(horizon_suffixes, horizon_losses):
            self.log(f'val_mae_{h_name}', losses_h['mae'], on_step=False, on_epoch=True)
            self.log(f'val_ssim_loss_{h_name}', losses_h['ssim'], on_step=False, on_epoch=True)
            self.log(f'val_lap_loss_{h_name}', losses_h['lap'], on_step=False, on_epoch=True)
            self.log(f'val_hist_loss_{h_name}', losses_h['hist'], on_step=False, on_epoch=True)
            self.log(f'val_pinball_lower_{h_name}', losses_h['pinball_lower'], on_step=False, on_epoch=True)
            self.log(f'val_pinball_upper_{h_name}', losses_h['pinball_upper'], on_step=False, on_epoch=True)
        
        # Log coverage metrics
        for h_name, coverage in coverage_stats:
            self.log(f'val_coverage_{h_name}', coverage, on_step=False, on_epoch=True)
        
        # Average losses across horizons
        avg_mse = self._horizon_mean(horizon_losses, 'mse')
        avg_mae = self._horizon_mean(horizon_losses, 'mae')
        avg_ssim = self._horizon_mean(horizon_losses, 'ssim')
        avg_lap = self._horizon_mean(horizon_losses, 'lap')
        avg_hist = self._horizon_mean(horizon_losses, 'hist')
        avg_pinball_lower = self._horizon_mean(horizon_losses, 'pinball_lower')
        avg_pinball_upper = self._horizon_mean(horizon_losses, 'pinball_upper')
        avg_central = self._horizon_mean(horizon_losses, 'central')
        avg_dist = self._horizon_mean(horizon_losses, 'dist')
        avg_total = self._horizon_mean(horizon_losses, 'total')
        avg_coverage = torch.stack([c for _, c in coverage_stats]).mean() if coverage_stats else torch.tensor(0.0)
        
        # Log averaged metrics (total)
        self.log('val_loss', avg_mse, on_step=False, on_epoch=True)
        self.log('val_mae_total', avg_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ssim_loss_total', avg_ssim, on_step=False, on_epoch=True)
        self.log('val_lap_loss_total', avg_lap, on_step=False, on_epoch=True)
        self.log('val_hist_loss_total', avg_hist, on_step=False, on_epoch=True)
        self.log('val_pinball_lower_total', avg_pinball_lower, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pinball_upper_total', avg_pinball_upper, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_coverage_total', avg_coverage, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_total_loss', avg_total, on_step=False, on_epoch=True, prog_bar=True)
        # Exactly the objective the trunk and the central heads are trained on: no pinball,
        # no histogram. ModelCheckpoint monitors val_total_loss by default, which includes
        # both — so a quantile-only change selects a different epoch and therefore a
        # different central field, and a central-only A/B carries a confound without this.
        self.log('val_central_loss', avg_central, on_step=False, on_epoch=True)
        if self.is_distributional:
            # The objective itself, isolated from the auxiliary MSE and the spatial terms.
            # This is what --checkpoint_monitor val_crps selects on, and it is the only
            # metric that scores the *whole* predictive distribution rather than three
            # points of it.
            self.log('val_crps', avg_dist, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'val_{self.dist_objective}_total', avg_dist,
                     on_step=False, on_epoch=True)
        
        # Print validation metrics (only for 20yr horizon for brevity)
        if batch_idx == 0:
            losses_20yr = horizon_losses[-1]  # Last horizon is 20yr
            hist_active = self.histogram_weight > 0 and self.current_epoch >= self.histogram_warmup_epochs
            hist_str = f", Hist: {losses_20yr['hist'].item():.5f}" if hist_active else " [hist off]"
            coverage_20yr = coverage_stats[-1][1] if coverage_stats else 0.0
            print(f"\n{'='*70}")
            print(f"[VAL] Epoch {self.current_epoch} - 20yr Horizon Metrics:")
            print(f"  MSE: {losses_20yr['mse'].item():.5f}, MAE: {losses_20yr['mae'].item():.5f}, SSIM: {1.0 - losses_20yr['ssim'].item():.5f}, Lap: {losses_20yr['lap'].item():.5f}{hist_str}")
            print(f"  Pinball Lower: {losses_20yr['pinball_lower'].item():.5f}, Pinball Upper: {losses_20yr['pinball_upper'].item():.5f}")
            print(f"  Coverage: {coverage_20yr:.1f}% (target: 95%)")
            print(f"[VAL] Average across all horizons:")
            print(f"  Total Loss: {avg_total.item():.5f}, Coverage: {avg_coverage.item():.1f}%")
            print(f"  Pinball Lower: {avg_pinball_lower.item():.5f}, Pinball Upper: {avg_pinball_upper.item():.5f}")
            print(f"{'='*70}\n")
        
        return avg_total
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_train_epoch_end(self):
        """Fail loudly on a diverged model, then accumulate the tail epochs for averaging.

        A NaN in the weights is permanent and silent: `ModelCheckpoint` never selects a NaN
        epoch because NaN compares false, so the run quietly falls back to its last healthy
        checkpoint, finishes all 150 epochs, and **scores normally**. Six of round 1's sixteen
        runs died this way; one of them died at epoch 3 and was published as a variant result
        against 150-epoch baselines, and the spread that comparison produced is what motivated
        round 2. Whatever the cause, a run that trained for 3 epochs must not be able to
        masquerade as one that trained for 150.
        """
        if self.abort_on_nonfinite:
            bad = [n for n, p in self.named_parameters() if not torch.isfinite(p).all()]
            if bad:
                raise RuntimeError(
                    f"NON-FINITE WEIGHTS at end of epoch {self.current_epoch}: "
                    f"{len(bad)} of {len(list(self.parameters()))} tensors, e.g. {bad[:3]}. "
                    f"Training diverged and every later epoch is worthless; the monitored "
                    f"checkpoint would have hidden this by falling back to an earlier epoch. "
                    f"Re-run, or pass --abort_on_nonfinite False to keep the old silent "
                    f"behaviour."
                )
        if self.weight_avg_last <= 0:
            return
        total = int(self.trainer.max_epochs or 0)
        if self.current_epoch < total - self.weight_avg_last:
            return
        sd = self.state_dict()
        if self._wa_sum is None:
            # Only floating-point tensors are averaged; anything else (counters, integer
            # buffers) is taken from the final epoch, where averaging would be meaningless.
            self._wa_sum = {k: v.detach().double().clone()
                            for k, v in sd.items() if v.is_floating_point()}
            self._wa_count = 1
        else:
            for k, acc in self._wa_sum.items():
                acc.add_(sd[k].detach().double())
            self._wa_count += 1

    def on_train_end(self):
        if self.weight_avg_last <= 0 or not self._wa_count:
            return
        # One NaN epoch inside the averaging window poisons the mean of every tensor, and the
        # averaged weights are what prediction then uses -- so this path must not be the way a
        # diverged run reaches the product.
        bad_acc = [k for k, v in self._wa_sum.items() if not torch.isfinite(v).all()]
        if bad_acc and self.abort_on_nonfinite:
            raise RuntimeError(
                f"weight averaging accumulated non-finite values in {len(bad_acc)} tensors "
                f"(e.g. {bad_acc[:3]}); at least one of the last {self._wa_count} epochs was "
                f"diverged, so the average is NaN and prediction would run on it.")
        sd = self.state_dict()
        for k, acc in self._wa_sum.items():
            sd[k].copy_((acc / self._wa_count).to(sd[k].dtype))
        print(f"[weight averaging] wrote the mean of the last {self._wa_count} epochs "
              f"into {len(self._wa_sum)} tensors")

    def on_train_epoch_start(self):
        """Cosine schedule with linear warmup, stepped by hand.

        Under manual optimization Lightning does not drive a returned lr_scheduler, so the
        learning rate is written into the param groups directly. 'none' (the default) never
        touches them, which is today's behaviour exactly.
        """
        if self.lr_schedule != 'cosine':
            return
        import math
        total = max(int(self.trainer.max_epochs or 1), 1)
        warm = max(int(round(self.lr_warmup_frac * total)), 1)
        e = int(self.current_epoch)
        if e < warm:
            scale = (e + 1) / warm
        else:
            prog = (e - warm) / max(total - warm, 1)
            scale = self.lr_min_frac + (1.0 - self.lr_min_frac) * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        lr = self.lr * scale
        for opt in self.trainer.optimizers:
            for g in opt.param_groups:
                g['lr'] = lr
        if e == 0:
            # A printed signature, because `lr` only goes to W&B and a run whose schedule
            # silently did not engage is indistinguishable from one where it did nothing.
            print(f"[lr schedule] cosine: {warm} warmup epochs of {total}, "
                  f"lr {self.lr:.2e} -> {self.lr * self.lr_min_frac:.2e}")
        self.log('lr', lr, on_step=False, on_epoch=True)
