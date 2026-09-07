import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..locationencoder import LocationEncoder
from .convlstm import ConvLSTM
from torch.utils.checkpoint import checkpoint
from .quantile_pwl import ISQFQuantile, PWLQuantile, n_pwl_params
from .quantile_spline import (
    U_KNOTS_DEFAULT,
    n_spline_params,
    normal_height_bias,
    rebuild,
    splines_from_output,
)

class SpatioTemporalPredictor(nn.Module):
    """
    Multi-horizon spatio-temporal predictor using ConvLSTM with independent prediction heads.
    
    Architecture:
    - input: [B, T, C_d, H, W] dynamic + [B, C_s, H, W] static
    - ConvLSTM processes temporal sequence → shared representation
    - 12 independent prediction heads:
        * 4 central heads (one per horizon): Optimized for accuracy + spatial patterns
        * 4 lower quantile heads (2.5%): Optimized for lower bound estimation
        * 4 upper quantile heads (97.5%): Optimized for upper bound estimation
    - output: [B, 12, H, W] (predicted HM at 4 horizons × 3 predictions)
      
      Output channel ordering:
        - Channels [0, 3, 6, 9]: Lower 2.5% quantile (q=0.025)
        - Channels [1, 4, 7, 10]: Central prediction (optimized for MSE+SSIM+Lap+Hist)
        - Channels [2, 5, 8, 11]: Upper 97.5% quantile (q=0.975)
      Horizons: 5yr, 10yr, 15yr, 20yr
    
    Key Design - Gradient Flow:
    - Central loss (MSE+SSIM+Lap+Hist) → backprops through: ConvLSTM + central_heads
    - Pinball loss → backprops through: lower_heads/upper_heads ONLY (NOT ConvLSTM)
    - Central heads never receive pinball loss gradients
    - Quantile heads never receive central loss gradients (MSE/SSIM/Lap/Hist)
    - ConvLSTM backbone is ONLY updated by central loss, never by pinball loss
    - This ensures quantile heads focus purely on uncertainty estimation
    - Quantile heads are smaller (hidden_dim/2) for efficiency
    """
    def __init__(self, hidden_dim=16, kernel_size=3, num_layers=1, num_static_channels=1, num_dynamic_channels=1,
                 use_location_encoder: bool = True,
                 locenc_backbone=("sphericalharmonics", "siren"),
                 locenc_hparams=None,
                 locenc_out_channels: int = 8,
                 quantile_context_channels: int = 0,
                 central_context_channels: int = 0,
                 trunk_context_channels: int = 0,
                 central_residual: bool = False,
                 monotone_quantile_width: bool = False,
                 quantile_dhat_context: bool = False,
                 width_parameterisation: str = 'softplus',
                 convlstm_dilations=None,
                 head_hidden_layers: int = 1,
                 width_head_mode: str = 'per_horizon',
                 central_target_transform: str = 'none',
                 central_transform_scale: float = 0.01,
                 initial_width_normalized: float = 0.065,
                 head_family: str = 'triple',
                 spline_u_knots=None,
                 spline_learn_slopes: bool = True,
                 spline_cumulative_width: bool = True,
                 spline_mean_nodes: int = 8,
                 spline_checkpoint: bool = True,
                 shape_head_hidden_layers: int = 1,
                 shape_head_width: int = 0,
                 spline_gap_floor: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_static_channels = int(num_static_channels)
        self.num_dynamic_channels = int(num_dynamic_channels)
        self.use_location_encoder = bool(use_location_encoder)
        self.locenc_out_channels = int(locenc_out_channels)
        if self.use_location_encoder and self.locenc_out_channels > 0:
            # Default hparams if not provided
            if locenc_hparams is None:
                locenc_hparams = dict(legendre_polys=10, dim_hidden=64, num_layers=2,
                                      optimizer=dict(lr=1e-4, wd=1e-3), num_classes=self.locenc_out_channels)
            else:
                locenc_hparams = dict(locenc_hparams)
                locenc_hparams['num_classes'] = self.locenc_out_channels
            self.location_encoder = LocationEncoder(locenc_backbone[0], locenc_backbone[1], locenc_hparams)
        else:
            self.location_encoder = None
        # Per-layer dilation widens the trunk's ~10 px receptive radius — the binding
        # structural limit for both head families — at zero parameter cost. None (the
        # default) means dilation 1 everywhere, i.e. the original trunk.
        if convlstm_dilations is None:
            self.convlstm_dilations = [1] * int(num_layers)
        elif isinstance(convlstm_dilations, int):
            self.convlstm_dilations = [int(convlstm_dilations)] * int(num_layers)
        else:
            self.convlstm_dilations = [int(d) for d in convlstm_dilations]
            if len(self.convlstm_dilations) != int(num_layers):
                raise ValueError(
                    f"convlstm_dilations has {len(self.convlstm_dilations)} entries for "
                    f"{num_layers} layers")
        # Neighbourhood context handed to the *trunk*, repeated across timesteps exactly like
        # elevation and climate. Until this phase the context reached the heads only, so the
        # ConvLSTM never saw "can change happen here at all" and could not combine it with its
        # own spatial features -- it could only have the answer applied to its output. Nothing
        # ever forced that choice; the precompute-on-the-full-raster requirement (radii >= 30 px
        # saturate inside a 128 px chip) is about where the covariate is *derived*, not where it
        # is *consumed*.
        self.trunk_context_channels = int(trunk_context_channels)
        self.convlstm = ConvLSTM(
            input_dim=(self.num_dynamic_channels + self.num_static_channels
                       + self.trunk_context_channels
                       + (self.locenc_out_channels if (self.use_location_encoder and self.locenc_out_channels > 0) else 0)),  # C_d dynamic + C_s static + C_ctx + C_loc
            hidden_dim=hidden_dim,
            kernel_size=(kernel_size, kernel_size),
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            dilation=self.convlstm_dilations,
        )
        # Multi-horizon prediction with independent heads for central and quantile predictions
        # Central heads: Optimized for accuracy + spatial patterns (MSE, SSIM, Laplacian, Histogram)
        # Quantile heads: Optimized purely for uncertainty estimation (Pinball loss only)
        self.num_horizons = 4
        # Extra channels handed to the *quantile* heads only. The trunk's receptive field
        # is ~10 px, so it cannot see whether past change exists 30-100 px away — which is
        # exactly the covariate that decides whether change is possible at all. Feeding it
        # to the quantile heads supplies information no amount of retraining could recover
        # from the trunk features, and leaves the central head's input untouched.
        self.quantile_context_channels = int(quantile_context_channels)
        # The central head can be handed the same context. The trunk's ~10 px radius cannot
        # see past change 30-100 px away either, and beyond 100 px from past change not one
        # of 493,240 measured pixels moved by more than 0.01 in twenty years — so this is
        # the covariate that tells the central head where the answer is exactly zero.
        self.central_context_channels = int(central_context_channels)
        # Predict change on top of HM_t0 instead of the absolute level. Measured motivation:
        # on the 53-70% of pixels whose observed 5-20yr change is below 0.001, the absolute
        # parameterisation still emits change of sd ~0.0075 HM, because reproducing HM_t0
        # through the trunk is not free. With the skip it is free, and "nothing happens" —
        # the answer for most of the map — costs the model nothing to say.
        self.central_residual = bool(central_residual)
        # Quantile heads emit *widths* around the central forecast that accumulate over
        # horizons, so spread is non-decreasing in lead time by construction (T4.2) and
        # lower <= central <= upper is structural rather than a post-hoc clip (T1.6).
        self.monotone_quantile_width = bool(monotone_quantile_width)
        # The width the residual needs varies with the *predicted* change (measured factor
        # 0.75 for dhat in (0.01,0.05] against 0.95 for (0.05,0.15] inside one distance
        # band), and the quantile heads have no channel carrying it: the interval is built
        # around a detached central forecast that is never an input. These two extra
        # channels are derived from tensors the head already has upstream — nothing new is
        # read from disk. Scaled to O(1) because sd(dhat) is ~0.09 in normalized units.
        self.quantile_dhat_context = bool(quantile_dhat_context)
        self.dhat_context_scale = 10.0
        n_dhat = 2 if self.quantile_dhat_context else 0
        # 'softplus': step = softplus(raw), today's behaviour.
        # 'exp':      step = w0 * exp(clamp(raw, -6, 6)) with the output conv zero-init, so
        #             the head predicts a log-multiplier of a per-horizon reference width
        #             and starts at exactly the same place softplus does.
        if width_parameterisation not in ('softplus', 'exp'):
            raise ValueError(f"unknown width_parameterisation {width_parameterisation!r}")
        self.width_parameterisation = str(width_parameterisation)
        self.initial_width_normalized = float(initial_width_normalized)
        # Depth of every prediction head. 1 (the default) is today's
        # Conv3x3 -> ReLU -> Conv1x1; higher values insert more 3x3+ReLU stages, which also
        # widens the head's own receptive field by 1 px per stage.
        self.head_hidden_layers = max(int(head_hidden_layers), 1)
        # How the four horizons' half-widths are produced.
        #   'per_horizon' — one module per horizon, today's behaviour.
        #   'joint'       — one module emitting all four increments from shared features, so
        #                   the growth profile in lead time is learned coherently rather
        #                   than by four modules that never see each other.
        #   'power'       — w(h) = w0 * (h/5)**gamma, with w0 and gamma both per pixel. Two
        #                   parameters instead of four, monotone by construction for
        #                   gamma >= 0, and it parameterises *exactly* the quantity measured
        #                   to be wrong: the far field's width grows 3.7-8.9x too fast with
        #                   lead time (k_up(20)/k_up(5) = 0.27 and 0.11 against 0.82-0.92
        #                   in the near field).
        #   'power_plus'  — a power law plus a *monotone additive* correction per horizon,
        #                   w(h) = w0*(h/5)**gamma + cumsum(softplus(extra))_h. The power law
        #                   supplies the growth shape 'power' gets right and the correction
        #                   restores the per-horizon level freedom 'joint' gets right, while
        #                   staying non-decreasing because both terms are. Initialised with
        #                   the correction negligible, so it starts exactly at 'power'.
        if width_head_mode not in ('per_horizon', 'joint', 'power', 'power_plus'):
            raise ValueError(f"unknown width_head_mode {width_head_mode!r}")
        self.width_head_mode = str(width_head_mode)
        # 'asinh' gives the central head a variance-stabilised output space: the change is
        # scale * sinh(raw), so a raw output near zero is linear and small while large
        # values are reachable without the head needing large weights. Zero-init still
        # starts at exact persistence.
        if central_target_transform not in ('none', 'asinh'):
            raise ValueError(f"unknown central_target_transform {central_target_transform!r}")
        self.central_target_transform = str(central_target_transform)
        self.central_transform_scale = float(central_transform_scale)

        # 'triple' emits (lower, central, upper) and nothing else -- the frozen product.
        # 'spline' additionally emits a full per-pixel quantile function, and derives the
        # triple *from* it, so every existing consumer of the first 12 channels keeps working
        # while the post-hoc width calibration and marginal reshaping become unnecessary.
        # 'triple' emits the frozen product's (lower, central, upper); 'spline' the C1
        # rational-quadratic quantile function; 'pwl' the same construction with linear pieces
        # and a closed-form CRPS (E2); 'isqf' Park et al. (2022) bounded (E1).
        if head_family not in ('triple', 'spline', 'pwl', 'isqf'):
            raise ValueError(f"unknown head_family {head_family!r}")
        self.head_family = str(head_family)
        self.spline_learn_slopes = bool(spline_learn_slopes)
        self.spline_cumulative_width = bool(spline_cumulative_width)
        self.spline_mean_nodes = int(spline_mean_nodes)
        self.spline_checkpoint = bool(spline_checkpoint)
        # E4: floor every quantile gap so no segment can imply a density above the
        # one the observation noise supports. Makes a picket fence structurally
        # impossible rather than merely discouraged.
        self.spline_gap_floor = bool(spline_gap_floor)
        knots = U_KNOTS_DEFAULT if spline_u_knots is None else tuple(spline_u_knots)
        self.register_buffer('spline_u_knots', torch.tensor(knots, dtype=torch.float32),
                             persistent=True)
        # HM is an index on [0, 1], but the model works in normalized space, so the support
        # constraint has to be carried in the same units. These are buffers rather than the
        # plain attributes `hm_mean`/`hm_std` that the rest of the codebase uses because they
        # must survive a checkpoint round trip: a spline restored without its support bounds
        # would silently emit quantiles outside the physical range.
        self.register_buffer('hm_norm', torch.tensor([0.0, 1.0]), persistent=True)
        self.register_buffer('hm_norm_set', torch.tensor(False), persistent=True)

        def _head(in_ch, mid_ch, out_ch, hidden_layers=None):
            n_hidden = self.head_hidden_layers if hidden_layers is None else int(hidden_layers)
            layers = [nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True),
                      nn.ReLU(inplace=True)]
            for _ in range(n_hidden - 1):
                layers += [nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=True),
                           nn.ReLU(inplace=True)]
            layers.append(nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=True))
            return nn.Sequential(*layers)

        self._head = _head

        # Central prediction heads (one per horizon)
        # These produce the "best estimate" optimized for multiple objectives
        self.central_heads = nn.ModuleList([
            _head(hidden_dim + self.central_context_channels, hidden_dim, 1)
            for _ in range(self.num_horizons)
        ])
        if self.central_residual:
            # Zero the output convolution so the model starts at exact persistence. The
            # weight still receives gradient (its input activations are non-zero), so this
            # is a starting point, not a dead branch.
            for head in self.central_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

        # Quantile heads. 'per_horizon' keeps one module per horizon (today); 'joint' and
        # 'power' use a single module per side emitting all horizons at once, so the shape
        # of the width's growth in lead time is a learned function of shared features.
        q_in = hidden_dim + self.quantile_context_channels + n_dhat
        q_mid = hidden_dim // 2
        n_out = {'per_horizon': 1, 'joint': self.num_horizons, 'power': 2,
                 'power_plus': 2 + self.num_horizons}[self.width_head_mode]
        n_modules = self.num_horizons if self.width_head_mode == 'per_horizon' else 1
        self.lower_heads = nn.ModuleList([_head(q_in, q_mid, n_out) for _ in range(n_modules)])
        self.upper_heads = nn.ModuleList([_head(q_in, q_mid, n_out) for _ in range(n_modules)])

        if self.head_family in ('spline', 'pwl', 'isqf'):
            n_knots = self.spline_u_knots.numel()
            n_bins = n_knots - 1
            if self.head_family == 'spline':
                self.n_spline_params = n_spline_params(n_knots, self.spline_learn_slopes)
            else:
                self.n_spline_params = n_pwl_params(n_knots)
            # Every family emits [location, scale, shape...], so this subtraction means the
            # same thing for all three.
            n_shape = self.n_spline_params - 2
            # One width head and one shape head per horizon, at the same half-trunk width as
            # the quantile heads they replace.
            self.width_heads = nn.ModuleList([_head(q_in, q_mid, 1)
                                              for _ in range(self.num_horizons)])
            # The shape head carries the whole distribution's form: 27 outputs from q_mid
            # channels at the default. These two flags size it independently of the other
            # heads, because --head_hidden_layers goes through the shared factory and would
            # confound a shape-head experiment with a change to the central head.
            s_width = int(shape_head_width) or q_mid
            self.shape_heads = nn.ModuleList([
                _head(q_in, s_width, n_shape, hidden_layers=shape_head_hidden_layers)
                for _ in range(self.num_horizons)])
            # `scale` is the *full* 95% width, where initial_width_normalized is a half-width.
            b0 = math.log(math.expm1(max(2.0 * self.initial_width_normalized, 1e-4)))
            for head in self.width_heads:
                nn.init.constant_(head[-1].bias, b0)
            hb = normal_height_bias(self.spline_u_knots)
            for head in self.shape_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                head[-1].bias.data[:n_bins] = hb
                # Slopes start at softplus(0), which the Fritsch-Carlson alternative would
                # also give up to a constant; the heights carry the shape at init.
        if self.monotone_quantile_width:
            heads = list(self.lower_heads) + list(self.upper_heads)
            if self.width_head_mode in ('power', 'power_plus'):
                # Channel 0 is w0, channel 1 is the growth exponent. Start at w(5) = w0 and
                # gamma = 1, i.e. width linear in lead time — exactly what four equal
                # cumulative increments give, so 'power' starts where 'per_horizon' does.
                b_w0 = math.log(math.expm1(max(self.initial_width_normalized, 1e-4)))
                b_g = math.log(math.expm1(1.0))
                b_extra = math.log(math.expm1(1e-4))   # negligible additive correction
                for head in heads:
                    nn.init.zeros_(head[-1].weight)
                    head[-1].bias.data[0] = b_w0
                    head[-1].bias.data[1] = b_g
                    if self.width_head_mode == 'power_plus':
                        head[-1].bias.data[2:] = b_extra
            elif self.width_parameterisation == 'exp':
                # step = w0 * exp(raw); zero the output conv so every head starts at exactly
                # w0, the same place the softplus bias below starts it.
                for head in heads:
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)
            else:
                # Each head's raw output passes through softplus and is accumulated, so bias
                # the output conv to make the *first* increment a sensible interval rather than
                # softplus(0) = 0.69 normalized units (~0.11 HM), which starts absurdly wide.
                b0 = math.log(math.expm1(max(self.initial_width_normalized, 1e-4)))
                for head in heads:
                    nn.init.constant_(head[-1].bias, b0)

    def _width_step(self, raw):
        """Raw head output -> a strictly positive half-width increment.

        The pinball gradient with respect to the head's raw output is (a constant) times
        d(step)/d(raw). Under softplus that derivative is sigmoid(raw), which is ~0.0085 at
        the far field's width and ~0.14 at the near field's — so the far field's width
        parameter learns ~18x slower than the near field's, on exactly the pixels the
        measured width factors say are furthest from right. The multiplicative form makes
        d(step)/d(raw) = step, which composes with a scale-normalised pinball into a
        scale-free effective learning rate.
        """
        if self.width_parameterisation == 'exp':
            return self.initial_width_normalized * torch.exp(raw.clamp(-6.0, 6.0))
        return F.softplus(raw)

    def set_norm_stats(self, hm_mean, hm_std):
        """Record the HM normalisation so the spline's support can be expressed in it."""
        self.hm_norm.data = torch.tensor([float(hm_mean), float(hm_std)],
                                         device=self.hm_norm.device)
        self.hm_norm_set.data = torch.tensor(True, device=self.hm_norm_set.device)

    def spline_clamp(self):
        """HM's physical range [0, 1], expressed in the model's normalized units.

        Refuses to guess. An unclamped spline trains and predicts perfectly happily while
        emitting negative HM in its lower tail, and nothing downstream would object -- the
        rasters would simply carry impossible values in the 2.5% quantile of remote pixels.
        """
        if not bool(self.hm_norm_set):
            raise RuntimeError(
                "head_family='spline' needs the HM normalisation: call "
                "model.set_norm_stats(hm_mean, hm_std) before the first forward pass.")
        mean, std = self.hm_norm[0], self.hm_norm[1]
        return float((0.0 - mean) / std), float((1.0 - mean) / std)

    def forward(self, input_dynamic, input_static, lonlat=None, quantile_context=None):
        # input_dynamic: [B, T, C_d, H, W]
        # input_static: [B, C_s, H, W]
        # lonlat: [B, H, W, 2]
        B, T, C, H, W = input_dynamic.shape
        # Optionally compute learnable location features
        if self.use_location_encoder and (self.location_encoder is not None) and (lonlat is not None):
            # Vectorized: process entire batch at once
            ll_flat = lonlat.reshape(B * H * W, 2)  # [B*H*W, 2]
            feats = self.location_encoder(ll_flat)  # [B*H*W, C_loc]
            loc_feats = feats.view(B, H, W, self.locenc_out_channels).permute(0, 3, 1, 2).contiguous()  # [B, C_loc, H, W]
            input_static = torch.cat([input_static, loc_feats], dim=1)
        if self.trunk_context_channels > 0:
            if quantile_context is None:
                raise RuntimeError(
                    f"this model expects {self.trunk_context_channels} trunk context channels "
                    f"but none was supplied. Pass change_context (and hm_context, if "
                    f"configured) through the batch; a zeroed covariate is not a safe default.")
            if quantile_context.shape[1] != self.trunk_context_channels:
                raise RuntimeError(
                    f"context has {quantile_context.shape[1]} channels, this model expects "
                    f"{self.trunk_context_channels} in the trunk. Check --context_radii / "
                    f"--hm_context_stats against the checkpoint they were trained with.")
            input_static = torch.cat(
                [input_static, quantile_context.to(input_static.dtype)], dim=1)
        # Repeat all static channels for each timestep and concat
        static_rep = input_static.unsqueeze(1).repeat(1, T, 1, 1, 1)  # [B, T, C_s, H, W]
        x = torch.cat([input_dynamic, static_rep], dim=2)  # [B, T, C_d+C_s, H, W]
        # ConvLSTM expects [B, T, C, H, W]
        output, _ = self.convlstm(x)
        # output[0]: [B, T, hidden_dim, H, W] (last layer)
        last_hidden = output[0][:, -1]  # [B, hidden_dim, H, W] (last timestep)
        
        # Generate independent predictions for each horizon
        # Each horizon has 3 separate heads: lower, central, upper
        preds = []

        def _with_context(n_channels):
            if n_channels <= 0:
                return last_hidden
            # Refuse rather than substitute zeros. The old behaviour trained happily on a
            # zeroed covariate whenever the context raster was not wired through, which reads
            # downstream as a real result; and now that the channel count varies with the
            # flags, an off-by-one band selection is a live way to get the wrong covariate
            # rather than a hypothetical one.
            if quantile_context is None:
                raise RuntimeError(
                    f"this model expects {n_channels} context channels but none was supplied. "
                    f"Pass change_context (and hm_context, if configured) through the batch; "
                    f"a zeroed covariate is not a safe default.")
            if quantile_context.shape[1] != n_channels:
                raise RuntimeError(
                    f"context has {quantile_context.shape[1]} channels, this model expects "
                    f"{n_channels}. Check --context_radii / --hm_context_stats against the "
                    f"checkpoint they were trained with.")
            return torch.cat([last_hidden, quantile_context.to(last_hidden.dtype)], dim=1)

        q_input = _with_context(self.quantile_context_channels)
        c_input = _with_context(self.central_context_channels)

        # HM at the last input timestep, in the same normalized space as the targets, so a
        # zero head output is exactly "no change".
        hm_t0 = input_dynamic[:, -1, 0:1]

        # Central forecasts first: the quantile heads may need the predicted change, and the
        # joint and power width heads need it for every horizon at once.
        centrals = []
        for h_idx in range(self.num_horizons):
            pc = self.central_heads[h_idx](c_input)               # [B, 1, H, W]
            if self.central_target_transform == 'asinh':
                # A variance-stabilised output space for the change: near zero the map is
                # linear with slope `scale`, and large changes are reachable without the
                # head carrying large weights. sinh(0) = 0, so a zero-initialised output
                # convolution still starts at exact persistence.
                pc = self.central_transform_scale * torch.sinh(pc.clamp(-8.0, 8.0))
            if self.central_residual:
                pc = hm_t0 + pc
            centrals.append(pc)

        def _q_input_for(central):
            """Head input, optionally carrying the detached predicted change.

            Detached and built from hm_t0, which is a plain input, so no gradient path to
            the trunk or the central heads is created; the isolation is unchanged.
            """
            if not self.quantile_dhat_context:
                return q_input
            dhat = (central.detach() - hm_t0) * self.dhat_context_scale
            return torch.cat([q_input, dhat, dhat.abs()], dim=1)

        if self.head_family in ('spline', 'pwl', 'isqf'):
            return self._forward_spline(centrals, _q_input_for)

        # Half-widths per horizon, non-decreasing in lead time by construction.
        w_lo, w_up = [], []
        if not self.monotone_quantile_width:
            pass
        elif self.width_head_mode == 'per_horizon':
            cum_lo = cum_up = None
            for h_idx in range(self.num_horizons):
                qi = _q_input_for(centrals[h_idx])
                step_lo = self._width_step(self.lower_heads[h_idx](qi))
                step_up = self._width_step(self.upper_heads[h_idx](qi))
                cum_lo = step_lo if cum_lo is None else cum_lo + step_lo
                cum_up = step_up if cum_up is None else cum_up + step_up
                w_lo.append(cum_lo)
                w_up.append(cum_up)
        else:
            # One module per side sees shared features and emits every horizon, so the shape
            # of the growth in lead time is learned coherently. Conditioned on the
            # longest-lead predicted change, which is the most informative of the four.
            qi = _q_input_for(centrals[-1])
            raw_lo = self.lower_heads[0](qi)
            raw_up = self.upper_heads[0](qi)
            if self.width_head_mode == 'joint':
                cum_lo = torch.cumsum(self._width_step(raw_lo), dim=1)
                cum_up = torch.cumsum(self._width_step(raw_up), dim=1)
                w_lo = [cum_lo[:, i:i + 1] for i in range(self.num_horizons)]
                w_up = [cum_up[:, i:i + 1] for i in range(self.num_horizons)]
            else:  # 'power' / 'power_plus': w(h) = w0 * (h/5)**gamma [+ monotone correction]
                w0_lo, g_lo = F.softplus(raw_lo[:, 0:1]), F.softplus(raw_lo[:, 1:2])
                w0_up, g_up = F.softplus(raw_up[:, 0:1]), F.softplus(raw_up[:, 1:2])
                add_lo = add_up = None
                if self.width_head_mode == 'power_plus':
                    add_lo = torch.cumsum(F.softplus(raw_lo[:, 2:]), dim=1)
                    add_up = torch.cumsum(F.softplus(raw_up[:, 2:]), dim=1)
                for h_idx in range(self.num_horizons):
                    lr = math.log(float(h_idx + 1))    # (5,10,15,20) / 5
                    wl = w0_lo * torch.exp((g_lo * lr).clamp(max=8.0))
                    wu = w0_up * torch.exp((g_up * lr).clamp(max=8.0))
                    if add_lo is not None:
                        wl = wl + add_lo[:, h_idx:h_idx + 1]
                        wu = wu + add_up[:, h_idx:h_idx + 1]
                    w_lo.append(wl)
                    w_up.append(wu)

        for h_idx in range(self.num_horizons):
            pred_central = centrals[h_idx]
            if self.monotone_quantile_width:
                # Anchor the interval to the central forecast, detached so the pinball loss
                # still cannot reach the trunk or the central heads — the same gradient
                # isolation as before, expressed structurally instead of by zeroing grads.
                anchor = pred_central.detach()
                pred_lower = anchor - w_lo[h_idx]
                pred_upper = anchor + w_up[h_idx]
            else:
                qi = _q_input_for(pred_central)
                pred_lower = self.lower_heads[h_idx](qi)           # [B, 1, H, W]
                pred_upper = self.upper_heads[h_idx](qi)           # [B, 1, H, W]

            # Append in order: lower, central, upper for this horizon
            preds.extend([pred_lower, pred_central, pred_upper])

        # Stack predictions: [B, 12, H, W] (4 horizons × 3 predictions)
        # Channel ordering: [lower_5yr, central_5yr, upper_5yr, lower_10yr, central_10yr, upper_10yr, ...]
        pred = torch.cat(preds, dim=1)
        return pred

    def _forward_spline(self, centrals, q_input_for):
        """Emit the triple *and* the quantile function behind it.

        Output is ``[B, 12 + 4 * P, H, W]``. The first 12 channels keep the historical
        ``(lower, central, upper) x 4`` order, so every existing reader -- the prediction
        writer, the stitcher, the scorers, the T1/T2 scorecard rows -- is untouched. They are
        now *derived* from the spline rather than predicted alongside it, which is what makes
        the ensemble a representation of the published maps instead of a second product.

        Channel 1 of each horizon is ``E[Q]``, not the median. The mean is the RMSE-optimal
        point estimate and this residual is strongly right-skewed, so the two differ; that
        also means the published central is no longer the midpoint of the published interval,
        and on a heavily skewed pixel it can even fall outside it. That is honest rather than
        convenient, and the scorer reports how often it happens.
        """
        clamp = self.spline_clamp()
        blocks = []
        cum = None
        for h_idx in range(self.num_horizons):
            qi = q_input_for(centrals[h_idx])
            # Non-negative increments accumulated across horizons: the 95% width cannot
            # shrink with lead time (T4.2), by construction rather than by a later pass.
            step = F.softplus(self.width_heads[h_idx](qi))
            # spline_cumulative_width=False makes each horizon's scale independent, so the
            # 95% width is free to SHRINK with lead time. That is the one monotonicity this
            # model imposes rather than inherits: Q(u) increasing in u is structural to the
            # spline (softmax bin heights, positive derivatives) and is untouched here.
            cum = step if (cum is None or not self.spline_cumulative_width) else cum + step
            blocks.append(torch.cat([centrals[h_idx], cum, self.shape_heads[h_idx](qi)], dim=1))
        block = torch.cat(blocks, dim=1)

        splines = self._decode(block, clamp)
        n_nodes = self.spline_mean_nodes
        u_knots = self.spline_u_knots

        def _mean(anchor, scale, v_knots, derivs):
            return rebuild(anchor, scale, v_knots, derivs, u_knots, clamp).mean(n_nodes=n_nodes)

        preds = []
        for sp in splines:
            lower, _, upper = sp.triple()
            if self.head_family != 'spline':
                # Q is linear per segment, so E[Q] is the trapezoid sum -- exact, cheap, and
                # with no quadrature graph to checkpoint.
                central = sp.mean()
            elif self.spline_checkpoint and torch.is_grad_enabled():
                central = checkpoint(_mean, sp.anchor, sp.scale, sp.v_knots, sp.derivs,
                                     use_reentrant=False)
            else:
                central = sp.mean(n_nodes=n_nodes)
            preds.extend([lower.unsqueeze(1), central.unsqueeze(1), upper.unsqueeze(1)])
        return torch.cat(preds + [block], dim=1)

    def _decode(self, block, clamp, n_triple=0):
        """``[B, n_h * P, H, W]`` -> one quantile function per horizon, whichever family.

        One decoder for the loss, the prediction writer and the scorer alike; this project
        has been bitten repeatedly by the same quantity being defined in several places.
        """
        if self.head_family == 'spline':
            return splines_from_output(block, self.num_horizons, self.spline_u_knots,
                                       learn_slopes=self.spline_learn_slopes,
                                       clamp=clamp, n_triple=n_triple)
        cls = ISQFQuantile if self.head_family == 'isqf' else PWLQuantile
        p = self.n_spline_params
        blk = block[:, n_triple:].movedim(1, -1)
        if blk.shape[-1] != self.num_horizons * p:
            raise ValueError(f"expected {self.num_horizons * p} head channels after "
                             f"{n_triple}, got {blk.shape[-1]}")
        kw = {}
        if self.head_family == 'pwl':
            kw = dict(gap_floor=self.spline_gap_floor, hm_std=float(self.hm_norm[1]))
        return [cls.from_channels(blk[..., h * p:(h + 1) * p], self.spline_u_knots,
                                  scale_pre=blk[..., h * p + 1], clamp=clamp, **kw)
                for h in range(self.num_horizons)]
