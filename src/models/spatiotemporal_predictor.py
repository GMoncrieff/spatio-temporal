import torch
import torch.nn as nn
import torch.nn.functional as F
from ..locationencoder import LocationEncoder
from .convlstm import ConvLSTM

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
                 central_residual: bool = False,
                 monotone_quantile_width: bool = False,
                 quantile_dhat_context: bool = False,
                 width_parameterisation: str = 'softplus',
                 convlstm_dilations=None,
                 initial_width_normalized: float = 0.065):
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
        self.convlstm = ConvLSTM(
            input_dim=self.num_dynamic_channels + self.num_static_channels + (self.locenc_out_channels if (self.use_location_encoder and self.locenc_out_channels > 0) else 0),  # C_d dynamic + C_s static + C_loc
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

        # Central prediction heads (one per horizon)
        # These produce the "best estimate" optimized for multiple objectives
        self.central_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim + self.central_context_channels, hidden_dim, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
            )
            for _ in range(self.num_horizons)
        ])
        if self.central_residual:
            # Zero the output convolution so the model starts at exact persistence. The
            # weight still receives gradient (its input activations are non-zero), so this
            # is a starting point, not a dead branch.
            for head in self.central_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

        # Lower quantile heads (2.5%, one per horizon)
        # Smaller networks since quantile estimation is simpler than full prediction
        self.lower_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim + self.quantile_context_channels + n_dhat, hidden_dim // 2, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 1, kernel_size=1, bias=True),
            )
            for _ in range(self.num_horizons)
        ])
        
        # Upper quantile heads (97.5%, one per horizon)
        self.upper_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim + self.quantile_context_channels + n_dhat, hidden_dim // 2, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 1, kernel_size=1, bias=True),
            )
            for _ in range(self.num_horizons)
        ])
        if self.monotone_quantile_width:
            if self.width_parameterisation == 'exp':
                # step = w0 * exp(raw); zero the output conv so every head starts at exactly
                # w0, the same place the softplus bias below starts it.
                for head in list(self.lower_heads) + list(self.upper_heads):
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)
            else:
                # Each head's raw output passes through softplus and is accumulated, so bias
                # the output conv to make the *first* increment a sensible interval rather than
                # softplus(0) = 0.69 normalized units (~0.11 HM), which starts absurdly wide.
                import math
                b0 = math.log(math.expm1(max(self.initial_width_normalized, 1e-4)))
                for head in list(self.lower_heads) + list(self.upper_heads):
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
            if quantile_context is None:
                ctx = last_hidden.new_zeros(B, n_channels, H, W)
            else:
                ctx = quantile_context.to(last_hidden.dtype)
            return torch.cat([last_hidden, ctx], dim=1)

        q_input = _with_context(self.quantile_context_channels)
        c_input = _with_context(self.central_context_channels)

        # HM at the last input timestep, in the same normalized space as the targets, so a
        # zero head output is exactly "no change".
        hm_t0 = input_dynamic[:, -1, 0:1]

        # Cumulative half-widths, so spread cannot shrink with lead time.
        w_lo_cum = None
        w_up_cum = None

        for h_idx in range(self.num_horizons):
            pred_central = self.central_heads[h_idx](c_input)     # [B, 1, H, W]
            if self.central_residual:
                pred_central = hm_t0 + pred_central

            # The predicted change, detached, as a head input. Detached and derived from
            # hm_t0 (a plain input), so no gradient path to the trunk or the central heads
            # is created — the isolation is unchanged.
            if self.quantile_dhat_context:
                dhat = (pred_central.detach() - hm_t0) * self.dhat_context_scale
                q_input_h = torch.cat([q_input, dhat, dhat.abs()], dim=1)
            else:
                q_input_h = q_input

            if self.monotone_quantile_width:
                # Anchor the interval to the central forecast, detached so the pinball loss
                # still cannot reach the trunk or the central heads — the same gradient
                # isolation as before, expressed structurally instead of by zeroing grads.
                anchor = pred_central.detach()
                step_lo = self._width_step(self.lower_heads[h_idx](q_input_h))
                step_up = self._width_step(self.upper_heads[h_idx](q_input_h))
                w_lo_cum = step_lo if w_lo_cum is None else w_lo_cum + step_lo
                w_up_cum = step_up if w_up_cum is None else w_up_cum + step_up
                pred_lower = anchor - w_lo_cum
                pred_upper = anchor + w_up_cum
            else:
                pred_lower = self.lower_heads[h_idx](q_input_h)     # [B, 1, H, W]
                pred_upper = self.upper_heads[h_idx](q_input_h)     # [B, 1, H, W]

            # Append in order: lower, central, upper for this horizon
            preds.extend([pred_lower, pred_central, pred_upper])

        # Stack predictions: [B, 12, H, W] (4 horizons × 3 predictions)
        # Channel ordering: [lower_5yr, central_5yr, upper_5yr, lower_10yr, central_10yr, upper_10yr, ...]
        pred = torch.cat(preds, dim=1)
        return pred
