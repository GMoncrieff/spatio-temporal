import os
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset
import torch
from pyproj import Transformer

# Paths (updated to hm_medium dataset)
HM_DIR = os.path.join("data", "raw", "hm_global")
STATIC_DIR = HM_DIR

# List of years for which we have human footprint data
years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
hm_files = [os.path.join(HM_DIR, f"HM_{year}_AA_1000.tiff") for year in years]
# Time-varying HM covariates
HM_VARS = ["AG", "BU", "EX", "FR", "HI", "NS", "PO", "TI", "gdp", "population"]
component_files = {
    y: [os.path.join(HM_DIR, f"HM_{y}_{v}_1000.tiff") for v in HM_VARS]
    for y in years
}
static_files = [
    os.path.join(STATIC_DIR, "hm_static_ele_1000.tiff"),
   # os.path.join(STATIC_DIR, "hm_static_ele_asp_cosin_1000.tiff"),
   # os.path.join(STATIC_DIR, "hm_static_ele_asp_sin_1000.tiff"),
   # os.path.join(STATIC_DIR, "hm_static_ele_slope_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_tas_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_tasmin_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_pr_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_dpi_dsi_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_iucn_nostrict_1000.tiff"),
    os.path.join(STATIC_DIR, "hm_static_iucn_strict_1000.tiff"),
]

import numpy as np
import rasterio

class HumanFootprintChipDataset(torch.utils.data.Dataset):
    """
    Yields chips for ConvLSTM training (multi-horizon forecasting):
      - input_dynamic: [3, C_dyn, H, W] from years (1990, 1995, 2000)
      - input_static: [C_static, H, W]
      - targets: Dict with keys 'target_5yr', 'target_10yr', 'target_15yr', 'target_20yr'
                 Each target is [H, W] for years 2005, 2010, 2015, 2020 respectively
    """
    def __init__(
        self,
        hm_files,
        component_files,
        static_files,
        chip_size=512,
        timesteps=3,
        stride=256,
        mode="random",
        chips_per_epoch=100,
        fixed_input_years=(1990, 1995, 2000),
        fixed_target_years=(2005, 2010, 2015, 2020),  # Multi-horizon targets
        use_temporal_sampling=True,  # NEW: Enable temporal sampling for training
        end_year_options=(2000, 2005, 2010, 2015),  # NEW: Valid end years for temporal sampling
        stat_samples=2048,
        stat_sample_size=512,
        random_seed=42,
        min_valid_ratio=0.8,
        use_validity_filter=True,
        enforce_input_hm_valid=True,
        include_components=True,
        static_channels=None,
        split_mask_file=None,
        split_value=None,
        exclude_split_values=None,
        norm_stats=None,
        context_pattern=None,
        chip_weights=None,
        chip_sampling="uniform",
        chip_weight_alpha=4.0,
        hm_context_pattern=None,
        hm_context_stats=(),
        hm_context_radii=(3, 30, 100),
    ):
        self.hm_files = hm_files
        self.component_files = component_files
        self.static_files = static_files
        self.chip_size = chip_size
        self.timesteps = timesteps  # kept for compatibility; must be 3
        if len(fixed_input_years) != 3:
            raise ValueError("Multi-horizon setup expects exactly 3 input timesteps (1990, 1995, 2000)")
        self.stride = stride
        self.mode = mode
        self.chips_per_epoch = chips_per_epoch
        # Temporal sampling setup
        self.use_temporal_sampling = use_temporal_sampling and mode == "random"  # Only for training
        self.end_year_options = list(end_year_options)
        self.fixed_input_years = tuple(fixed_input_years)
        self.fixed_target_years = tuple(fixed_target_years)  # (2005, 2010, 2015, 2020)
        # Get indices for all target years
        self.target_t_indices = [years.index(y) for y in fixed_target_years]  # [3, 4, 5, 6]
        # Year to index mapping
        self.year_to_idx = {y: i for i, y in enumerate(years)}
        # Use lazy, windowed IO to avoid loading entire rasters into memory
        self.include_components = bool(include_components)
        self._hm_files = list(hm_files)
        self._static_files = list(static_files if static_channels is None else static_files[:int(static_channels)])
        self._comp_files = {y: list(component_files.get(y, [])) for y in years} if self.include_components else {y: [] for y in years}
        # Past-change context rasters, one per input window (band 1 = past change,
        # band 2 = distance to the nearest past change). Computed on the full raster, so
        # a 100 px radius means 100 km of geography rather than the edge of a 128 px chip.
        self.context_pattern = context_pattern
        self._ctx_srcs = None
        # Stratified chip sampling. Defaults reproduce uniform sampling exactly.
        self.chip_weights = chip_weights
        self.chip_sampling = str(chip_sampling)
        self.chip_weight_alpha = float(chip_weight_alpha)
        # Neighbourhood-HM context: mean and max HM within each radius, precomputed on the
        # full raster by scripts/prepare_hm_context.py. Bands are selected by (stat, radius)
        # from the raster's own tags, so a mismatch between what the model asks for and what
        # the raster holds fails loudly instead of reading the wrong band.
        self.hm_context_pattern = hm_context_pattern
        self.hm_context_stats = tuple(hm_context_stats or ())
        self.hm_context_radii = tuple(hm_context_radii)
        self._hmctx_srcs = None
        self._hm_band_idx = None

        # Split mask for train/val/test separation
        self.split_mask_file = split_mask_file
        self.split_value = split_value  # 1=train, 2=val, 3=test, 4=calib
        # Optional exclusion list (e.g. fold-CV: train on "everything except fold f").
        # A chip is kept only if NO pixel in it belongs to an excluded value.
        if exclude_split_values is None:
            self.exclude_split_values = None
        else:
            self.exclude_split_values = [int(v) for v in np.atleast_1d(exclude_split_values)]
        
        # Lazily opened rasterio datasets (per worker)
        self._hm_srcs = None
        self._static_srcs = None
        self._comp_srcs = None
        # Read raster shape from the first HM file
        with rasterio.open(self._hm_files[0]) as src0:
            self.H, self.W = src0.height, src0.width
        self.T = len(self._hm_files)
        # Estimate normalization stats from random windows (stat_samples), unless a
        # precomputed set is supplied. The stats are plain attributes (not persisted in
        # the .ckpt), so every inference entrypoint has to reproduce them; passing them in
        # from a JSON sidecar avoids repeatedly paying the raster-sampling cost.
        self.norm_stats_source = "computed"
        if norm_stats is not None:
            self._load_norm_stats(norm_stats)
        else:
            self._estimate_norm_stats(random_seed, stat_samples)

        # Map fixed years to indices in the stacked timeline
        year_to_idx = {y: i for i, y in enumerate(years)}
        # Expose available years for downstream labeling
        self.years = years
        try:
            self.input_t_idxs = [year_to_idx[y] for y in self.fixed_input_years]
            # Target indices already computed above as self.target_t_indices
        except KeyError as e:
            raise ValueError(f"Requested year {e} not found in available years {years}")
        # Validate ordering and availability
        if len(self.input_t_idxs) != 3:
            raise ValueError("fixed_input_years must have exactly 3 entries: e.g., (1990, 1995, 2000)")
        # Precompute valid positions (use last target year for positioning)
        self.valid_time_idxs = [self.target_t_indices[-1]]
        self._init_split_positions(chip_size, stride)

        self.C_comp = len(self._comp_files[years[0]]) if self.include_components else 0
        self.C_dyn = 1 + self.C_comp
        self.C_static = len(self._static_files)

    def norm_stats_dict(self):
        """Serializable snapshot of the normalization statistics (JSON sidecar payload)."""
        return {
            "hm_mean": float(self.hm_mean),
            "hm_std": float(self.hm_std),
            "static_means": [float(v) for v in self.static_means],
            "static_stds": [float(v) for v in self.static_stds],
            "comp_means": {k: float(v) for k, v in self.comp_means.items()},
            "comp_stds": {k: float(v) for k, v in self.comp_stds.items()},
            "static_files": [os.path.basename(f) for f in self._static_files],
            "include_components": bool(self.include_components),
        }

    def _load_norm_stats(self, norm_stats):
        """Adopt precomputed normalization statistics instead of resampling rasters."""
        expected_static = [os.path.basename(f) for f in self._static_files]
        if "static_files" in norm_stats and list(norm_stats["static_files"]) != expected_static:
            raise ValueError(
                "norm_stats were computed for a different static-channel set: "
                f"{norm_stats['static_files']} vs {expected_static}"
            )
        self.hm_mean = float(norm_stats["hm_mean"])
        self.hm_std = float(norm_stats["hm_std"])
        self.static_means = [float(v) for v in norm_stats["static_means"]]
        self.static_stds = [float(v) for v in norm_stats["static_stds"]]
        self.comp_means = {k: float(v) for k, v in norm_stats.get("comp_means", {}).items()}
        self.comp_stds = {k: float(v) for k, v in norm_stats.get("comp_stds", {}).items()}
        if self.include_components and not self.comp_means:
            raise ValueError("norm_stats missing comp_means/comp_stds but include_components=True")
        self.elev_mean = self.static_means[0] if self.static_means else 0.0
        self.elev_std = self.static_stds[0] if self.static_stds else 1.0
        self.norm_stats_source = "cached"
        print(f"Using cached normalization stats (hm_mean={self.hm_mean:.6f}, hm_std={self.hm_std:.6f})")

    def _estimate_norm_stats(self, random_seed, stat_samples):
        rng = np.random.default_rng(random_seed)
        hm_samples = []
        total_samps = max(64, int(stat_samples))
        per_year = max(1, int(np.ceil(total_samps / len(self._hm_files))))
        for yidx, f in enumerate(self._hm_files):
            with rasterio.open(f) as src:
                Hs, Ws = src.height, src.width
                for _ in range(per_year):
                    if Hs < self.chip_size or Ws < self.chip_size:
                        i = 0; j = 0
                    else:
                        i = int(rng.integers(0, Hs - self.chip_size + 1))
                        j = int(rng.integers(0, Ws - self.chip_size + 1))
                    window = rasterio.windows.Window(j, i, self.chip_size, self.chip_size)
                    arr = src.read(1, window=window, masked=True).filled(np.nan)
                    # Data is already in [0, 1] range
                    hm_samples.append(arr)
        if len(hm_samples) == 0:
            raise RuntimeError("Failed to sample windows for HM stats; check input rasters")
        hm_stack_samp = np.stack(hm_samples, axis=0)
        self.hm_mean = np.nanmean(hm_stack_samp)
        self.hm_std = np.nanstd(hm_stack_samp) + 1e-8
        # Static stats (per variable) - different variables have different scales
        self.static_means = []
        self.static_stds = []
        if len(self._static_files) > 0:
            print("Computing per-variable normalization statistics for static layers...")
            total_static_samps = max(32, int(stat_samples))
            for static_idx, f in enumerate(self._static_files):
                static_samples = []
                with rasterio.open(f) as src:
                    Hs, Ws = src.height, src.width
                    per_static = max(1, int(np.ceil(total_static_samps / max(1, len(self._static_files)))))
                    for _ in range(per_static):
                        if Hs < self.chip_size or Ws < self.chip_size:
                            i = 0; j = 0
                        else:
                            i = int(rng.integers(0, Hs - self.chip_size + 1))
                            j = int(rng.integers(0, Ws - self.chip_size + 1))
                        window = rasterio.windows.Window(j, i, self.chip_size, self.chip_size)
                        static_samples.append(src.read(1, window=window, masked=True).filled(np.nan))
                if len(static_samples) > 0:
                    static_stack = np.stack(static_samples, axis=0)
                    self.static_means.append(np.nanmean(static_stack))
                    self.static_stds.append(np.nanstd(static_stack) + 1e-8)
                else:
                    self.static_means.append(0.0)
                    self.static_stds.append(1.0)
            print("Static normalization stats:")
            for idx, f in enumerate(self._static_files):
                var_name = f.split('/')[-1].replace('hm_static_', '').replace('_1000.tiff', '')
                print(f"  {var_name}: mean={self.static_means[idx]:.6e}, std={self.static_stds[idx]:.6e}")
        # Keep elev_mean/std for backward compatibility (use first static variable)
        self.elev_mean = self.static_means[0] if self.static_means else 0.0
        self.elev_std = self.static_stds[0] if self.static_stds else 1.0
        
        # Component stats (per variable) - critical for GDP/population which have different scales
        self.comp_means = {}
        self.comp_stds = {}
        if self.include_components:
            print("Computing per-variable normalization statistics for components...")
            for var_idx, var_name in enumerate(HM_VARS):
                var_samples = []
                for year in years:
                    comp_file = self._comp_files[year][var_idx]
                    with rasterio.open(comp_file) as src:
                        Hs, Ws = src.height, src.width
                        per_var = max(1, int(np.ceil(stat_samples / len(years))))
                        for _ in range(per_var):
                            if Hs < self.chip_size or Ws < self.chip_size:
                                i = 0; j = 0
                            else:
                                i = int(rng.integers(0, Hs - self.chip_size + 1))
                                j = int(rng.integers(0, Ws - self.chip_size + 1))
                            window = rasterio.windows.Window(j, i, self.chip_size, self.chip_size)
                            arr = src.read(1, window=window, masked=True).filled(np.nan)
                            var_samples.append(arr)
                if len(var_samples) > 0:
                    var_stack = np.stack(var_samples, axis=0)
                    self.comp_means[var_name] = np.nanmean(var_stack)
                    self.comp_stds[var_name] = np.nanstd(var_stack) + 1e-8
                else:
                    self.comp_means[var_name] = 0.0
                    self.comp_stds[var_name] = 1.0
            print("Component normalization stats:")
            for var_name in HM_VARS:
                print(f"  {var_name}: mean={self.comp_means[var_name]:.6e}, std={self.comp_stds[var_name]:.6e}")
        
    def _hm_context_bands(self, src):
        """Band indices for the requested (stat, radius) pairs, cached per worker."""
        if self._hm_band_idx is None:
            from prepare_hm_context import band_indices
            self._hm_band_idx = band_indices(src.tags(), self.hm_context_stats,
                                             self.hm_context_radii)
        return self._hm_band_idx

    def _init_chip_sampling(self):
        """Sampling probabilities over the already-fold-filtered chip positions.

        Positions come from ``valid_split_positions``, which the fold mask has already
        filtered, so stratifying adds **no new leak surface**: it changes how often a
        permitted chip is drawn, never which chips are permitted. The weights themselves are
        a function of position computed from input-window covariates only.

        ``_pos_correction`` is the importance weight that takes the estimator back to the
        population: uniform probability over stratified probability. Applied, the model still
        targets the natural distribution and the stratification only improves the optimiser's
        exposure to rare chips. Left off, the model targets a utility-weighted distribution
        instead -- a different estimand, and the run is labelled as such.
        """
        self._pos_cdf = None
        self._pos_correction = None
        if getattr(self, "chip_sampling", "uniform") != "stratified" or self.mode == "grid":
            return
        if self.valid_split_positions is None:
            raise ValueError("--chip_sampling stratified needs a split or fold mask")
        if not self.chip_weights:
            raise ValueError("--chip_sampling stratified needs --chip_weights")
        tab = np.load(self.chip_weights)
        w_tab, cs = tab["w"], int(tab["chip_size"])
        if cs != self.chip_size:
            raise ValueError(f"chip weights built at {cs} px, dataset uses {self.chip_size}")
        ii = np.clip(np.array([i for i, _ in self.valid_split_positions]) // cs,
                     0, w_tab.shape[0] - 1)
        jj = np.clip(np.array([j for _, j in self.valid_split_positions]) // cs,
                     0, w_tab.shape[1] - 1)
        w = w_tab[ii, jj].astype(np.float64)
        w = w / max(w.mean(), 1e-12)
        p = 1.0 + float(self.chip_weight_alpha) * w
        p = p / p.sum()
        self._pos_cdf = np.cumsum(p)
        self._pos_correction = ((1.0 / p.size) / p).astype(np.float32)
        print(f"  Stratified chip sampling: alpha={self.chip_weight_alpha}, "
              f"{p.size} positions, sampling ratio max/min "
              f"{p.max() / p.min():.1f}, correction range "
              f"[{self._pos_correction.min():.3f}, {self._pos_correction.max():.3f}]")

    def _init_split_positions(self, chip_size, stride):
        # Precompute valid positions for split if using split mask
        # Positions are filtered in BOTH random and grid mode. Grid mode previously ignored
        # the split mask entirely, which silently evaluated val/test over the whole globe.
        self._use_split_filter = self.split_mask_file is not None and (
            self.split_value is not None or self.exclude_split_values is not None
        )
        self.valid_split_positions = None
        if self._use_split_filter:
            print(
                f"Pre-computing valid positions for split_value={self.split_value}"
                f", exclude={self.exclude_split_values}..."
            )
            with rasterio.open(self.split_mask_file) as split_src:
                split_data = split_src.read(1)
            # Random mode samples on a chip_size lattice (matches split generation);
            # grid mode enumerates at the requested stride.
            step = chip_size if self.mode != "grid" else stride
            valid_positions = self._filter_split_positions(split_data, chip_size, step)
            del split_data
            self.valid_split_positions = valid_positions
            print(f"  Found {len(valid_positions)} valid chip positions (step={step})")
            if len(valid_positions) == 0:
                raise ValueError(
                    f"No valid positions found for split_value={self.split_value} / "
                    f"exclude_split_values={self.exclude_split_values}. Check split mask."
                )

        self._init_chip_sampling()

        # Precompute all chip positions if not random
        if self.mode == "grid":
            self.chip_positions = []
            if self._use_split_filter:
                for t in self.valid_time_idxs:
                    for i, j in self.valid_split_positions:
                        self.chip_positions.append((t, i, j))
            else:
                for t in self.valid_time_idxs:
                    for i in range(0, self.H - chip_size + 1, stride):
                        for j in range(0, self.W - chip_size + 1, stride):
                            self.chip_positions.append((t, i, j))
        else:
            self.chip_positions = None

    def _filter_split_positions(self, split_data, chip_size, step, block=128):
        """Enumerate (i, j) chip origins whose chip satisfies the split constraints.

        Include rule (``split_value``):    at least one pixel equals ``split_value``.
        Exclude rule (``exclude_split_values``): no pixel is in the excluded set, and at
        least one pixel is non-zero (i.e. the chip carries usable data).

        A coarse ``block``-resolution summary is used only as a conservative prefilter;
        every surviving candidate is still checked exactly against the full-resolution
        mask, so the result is identical to the naive double loop.
        """
        H, W = split_data.shape
        exclude = self.exclude_split_values

        include_mask = (split_data == self.split_value) if self.split_value is not None else None
        exclude_mask = np.isin(split_data, exclude) if exclude is not None else None
        nonzero_mask = (split_data > 0) if exclude is not None else None

        def block_any(arr):
            hb = (H + block - 1) // block
            wb = (W + block - 1) // block
            pad_h, pad_w = hb * block - H, wb * block - W
            if pad_h or pad_w:
                arr = np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=False)
            return arr.reshape(hb, block, wb, block).any(axis=(1, 3))

        inc_blocks = block_any(include_mask) if include_mask is not None else None
        exc_blocks = block_any(exclude_mask) if exclude_mask is not None else None
        nz_blocks = block_any(nonzero_mask) if nonzero_mask is not None else None

        positions = []
        for i in range(0, H - chip_size + 1, step):
            b_i0, b_i1 = i // block, (i + chip_size - 1) // block + 1
            for j in range(0, W - chip_size + 1, step):
                b_j0, b_j1 = j // block, (j + chip_size - 1) // block + 1
                # Cheap conservative rejections first.
                if inc_blocks is not None and not inc_blocks[b_i0:b_i1, b_j0:b_j1].any():
                    continue
                if nz_blocks is not None and not nz_blocks[b_i0:b_i1, b_j0:b_j1].any():
                    continue
                chip = split_data[i:i + chip_size, j:j + chip_size]
                if include_mask is not None and not (chip == self.split_value).any():
                    continue
                if exclude is not None:
                    if exc_blocks[b_i0:b_i1, b_j0:b_j1].any() and np.isin(chip, exclude).any():
                        continue
                    if not (chip > 0).any():
                        continue
                positions.append((i, j))
        return positions

    def _ensure_open(self):
        # Open datasets lazily per worker process
        if self._hm_srcs is None:
            self._hm_srcs = [rasterio.open(f) for f in self._hm_files]
        if self._static_srcs is None:
            self._static_srcs = [rasterio.open(f) for f in self._static_files]
        if self._comp_srcs is None:
            self._comp_srcs = {y: [rasterio.open(f) for f in self._comp_files[y]] for y in years}
        if self.context_pattern is not None and self._ctx_srcs is None:
            self._ctx_srcs = {}
            for y in years:
                p = self.context_pattern.format(year=y)
                if os.path.exists(p):
                    self._ctx_srcs[y] = rasterio.open(p)
        if (self.hm_context_stats and self.hm_context_pattern is not None
                and self._hmctx_srcs is None):
            self._hmctx_srcs = {}
            for y in years:
                p = self.hm_context_pattern.format(year=y)
                if os.path.exists(p):
                    self._hmctx_srcs[y] = rasterio.open(p)
            if not self._hmctx_srcs:
                raise FileNotFoundError(
                    f"--hm_context_stats was requested but no raster matched "
                    f"{self.hm_context_pattern}; run scripts/prepare_hm_context.py first")

    def __len__(self):
        if self.mode == "grid":
            return len(self.chip_positions)
        else:
            return self.chips_per_epoch

    def __getitem__(self, idx):
        # Try up to 10 times to get a non-all-NaN chip
        for _ in range(10):
            # === TEMPORAL SAMPLING ===
            if self.use_temporal_sampling:
                # Randomly sample end_year for this sample
                end_year = np.random.choice(self.end_year_options)
                input_years = [end_year - 10, end_year - 5, end_year]
                target_years = [end_year + 5, end_year + 10, end_year + 15, end_year + 20]
            else:
                # Fixed years (validation/testing)
                end_year = self.fixed_input_years[-1]  # 2000
                input_years = list(self.fixed_input_years)
                target_years = list(self.fixed_target_years)
            
            # Convert years to indices
            input_t_idxs = [self.year_to_idx[y] for y in input_years]
            target_t_idxs = [self.year_to_idx.get(y, None) for y in target_years]  # None for missing years
            
            chip_weight = 1.0
            if self.mode == "grid":
                t, i, j = self.chip_positions[idx]
            else:
                # Use last available target year for grid positioning
                t = self.target_t_indices[-1]  # Use 2020 for grid positioning
                
                # If using splits, sample from pre-computed valid positions
                if self.valid_split_positions is not None:
                    # Randomly select a valid chip position
                    if self._pos_cdf is not None:
                        pos_idx = int(min(np.searchsorted(self._pos_cdf, np.random.random()),
                                          len(self.valid_split_positions) - 1))
                        chip_weight = float(self._pos_correction[pos_idx])
                    else:
                        pos_idx = np.random.randint(0, len(self.valid_split_positions))
                    i, j = self.valid_split_positions[pos_idx]
                    # Add small random offset within chip for diversity.
                    # Skipped when excluding folds: a jittered chip could reach into the
                    # held-out fold and leak it into training, which would make the
                    # hindcast residuals no longer out-of-sample.
                    if self.exclude_split_values is None:
                        offset = min(32, self.chip_size // 4)
                        i = max(0, min(self.H - self.chip_size, i + np.random.randint(-offset, offset)))
                        j = max(0, min(self.W - self.chip_size, j + np.random.randint(-offset, offset)))
                else:
                    i = np.random.randint(0, self.H - self.chip_size + 1)
                    j = np.random.randint(0, self.W - self.chip_size + 1)
            # Build input from computed years using windowed reads
            self._ensure_open()
            # Dynamic inputs per timestep: base HM plus HM covariates (if enabled)
            dyn_list = []  # list of [C_dyn, H, W] for each timestep
            for t_idx, year in zip(input_t_idxs, input_years):
                channels = []
                # Base HM for this timestep
                arr_hm = self._hm_srcs[t_idx].read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
                # Data is already in [0, 1] range
                arr_hm = (arr_hm - self.hm_mean) / self.hm_std
                channels.append(arr_hm)
                # HM covariates for the same year
                if self.include_components and self._comp_srcs.get(year, []):
                    for var_idx, src in enumerate(self._comp_srcs[year]):
                        carr = src.read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
                        var_name = HM_VARS[var_idx]
                        # Replace NaN with 0 BEFORE normalization for all HM components
                        # Interpretation: missing data means "no pressure/activity"
                        carr = np.nan_to_num(carr, nan=0.0)
                        # Use per-variable normalization (critical for GDP/population)
                        carr = (carr - self.comp_means[var_name]) / self.comp_stds[var_name]
                        channels.append(carr)
                dyn_list.append(np.stack(channels, axis=0))  # [C_dyn, H, W]
            input_dynamic = np.stack(dyn_list, axis=0)  # [T, C_dyn, H, W]
            # Static layers
            static_list = []
            # Static file order: [ele, tas, tasmin, pr, dpi_dsi, iucn_nostrict, iucn_strict]
            # Replace NaN with 0 BEFORE normalization for specific variables
            nan_to_zero_static = {0, 4, 5, 6}  # ele, dpi_dsi, iucn_nostrict, iucn_strict
            for static_idx, src in enumerate(self._static_srcs):
                sarr = src.read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
                # Replace NaN with 0 for specific variables (before normalization)
                if static_idx in nan_to_zero_static:
                    sarr = np.nan_to_num(sarr, nan=0.0)
                # Use per-variable normalization
                sarr = (sarr - self.static_means[static_idx]) / self.static_stds[static_idx]
                static_list.append(sarr)
            input_static = np.stack(static_list, axis=0) if static_list else np.zeros((0, self.chip_size, self.chip_size), dtype=np.float32)
            # Build lon/lat grid [H, W, 2] from target raster transform, reproject to EPSG:4326 if needed
            ref = self._hm_srcs[self.target_t_indices[-1]]  # Use last target year (2020) as reference
            rows = np.arange(i, i + self.chip_size)
            cols = np.arange(j, j + self.chip_size)
            rr, cc = np.meshgrid(rows, cols, indexing='ij')
            xs, ys = rasterio.transform.xy(ref.transform, rr, cc)
            xs = np.array(xs); ys = np.array(ys)
            if ref.crs and ref.crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
                transformer = Transformer.from_crs(ref.crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(xs, ys)
            else:
                lon, lat = xs, ys
            lonlat = np.stack([lon, lat], axis=-1).astype(np.float32)  # [H, W, 2]
            
            # Multi-horizon targets (computed from end_year)
            targets = {}
            horizon_names = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
            all_valid = False
            
            for horizon_name, t_idx, target_year in zip(horizon_names, target_t_idxs, target_years):
                if t_idx is None or target_year > 2020:
                    # Missing year - fill with NaN (will be masked in loss)
                    target_h = np.full((self.chip_size, self.chip_size), np.nan, dtype=np.float32)
                else:
                    target_h = self._hm_srcs[t_idx].read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
                    # Data is already in [0, 1] range
                    target_h = (target_h - self.hm_mean) / self.hm_std
                targets[horizon_name] = torch.from_numpy(target_h).float()
                if not np.isnan(target_h).all():
                    all_valid = True
            
            context_arr = None
            if self.context_pattern is not None:
                src = (self._ctx_srcs or {}).get(input_years[-1])
                if src is not None:
                    win = rasterio.windows.Window(j, i, self.chip_size, self.chip_size)
                    context_arr = np.stack([
                        np.nan_to_num(src.read(1, window=win, masked=True).filled(np.nan), nan=0.0),
                        np.nan_to_num(src.read(2, window=win, masked=True).filled(np.nan), nan=1e4),
                    ], axis=0).astype(np.float32)

            hm_ctx_arr = None
            if self.hm_context_stats and self.hm_context_pattern is not None:
                src = (self._hmctx_srcs or {}).get(input_years[-1])
                if src is not None:
                    win = rasterio.windows.Window(j, i, self.chip_size, self.chip_size)
                    bands = self._hm_context_bands(src)
                    raw = src.read(bands, window=win).astype(np.float32)
                    # int16 x 1/32767 with -32768 as nodata. Nodata becomes 0: "no development
                    # nearby", which is what open water and the poles are.
                    hm_ctx_arr = np.where(raw == -32768, 0.0, raw * np.float32(1.0 / 32767.0))
            if hm_ctx_arr is None and self.hm_context_stats and self.hm_context_pattern:
                # Filled here rather than in either branch: the two sample dicts must carry
                # exactly the same keys or the default collate fails the moment one sample in
                # a batch takes the fallback path.
                n_hm = len(self.hm_context_stats) * len(self.hm_context_radii)
                hm_ctx_arr = np.zeros((n_hm, self.chip_size, self.chip_size), dtype=np.float32)

            if all_valid:
                sample = {
                    "input_dynamic": torch.from_numpy(input_dynamic).float(),
                    "input_static": torch.from_numpy(input_static).float(),
                    "lonlat": torch.from_numpy(lonlat).float(),
                    "timestep": t,
                    # NEW: Year metadata for visualization
                    "input_years": input_years,
                    "target_years": target_years,
                    "end_year": end_year,
                    "chip_weight": torch.tensor(chip_weight, dtype=torch.float32),
                }
                if context_arr is not None:
                    sample["change_context"] = torch.from_numpy(context_arr).float()
                if hm_ctx_arr is not None:
                    sample["hm_context"] = torch.from_numpy(hm_ctx_arr).float()
                sample.update(targets)  # Add all horizon targets
                return sample
        # If all attempts fail, return anyway (will be masked out in loss).
        # It must carry exactly the same keys as the success path or the default collate
        # fails as soon as one sample in a batch takes this branch.
        sample = {
            "input_dynamic": torch.from_numpy(input_dynamic).float(),
            "input_static": torch.from_numpy(input_static).float(),
            "lonlat": torch.from_numpy(lonlat).float(),
            "timestep": t,
            # NEW: Year metadata for visualization
            "input_years": input_years,
            "target_years": target_years,
            "end_year": end_year,
            "chip_weight": torch.tensor(chip_weight, dtype=torch.float32),
        }
        if self.context_pattern is not None:
            if context_arr is None:
                context_arr = np.zeros((2, self.chip_size, self.chip_size), dtype=np.float32)
                context_arr[1] = 1e4          # "no past change anywhere near"
            sample["change_context"] = torch.from_numpy(context_arr).float()
        if hm_ctx_arr is not None:
            sample["hm_context"] = torch.from_numpy(hm_ctx_arr).float()
        sample.update(targets)  # Add all horizon targets
        return sample


def get_dataloader(
    batch_size=1,
    chip_size=128,
    timesteps=3,
    stride=64,
    mode="random",
    chips_per_epoch=100,
    fixed_input_years=(1990, 1995, 2000),
    fixed_target_years=(2005, 2010, 2015, 2020),
    use_temporal_sampling=True,  # NEW: Enable temporal sampling
    end_year_options=(2000, 2005, 2010, 2015),  # NEW: End year options
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    min_valid_ratio=0.8,
    stat_samples=256,
    enforce_input_hm_valid=True,
    include_components=True,
    static_channels=None,
    split_mask_file=None,
    split_value=None,
    exclude_split_values=None,
    norm_stats=None,
    context_pattern=None,
    chip_weights=None,
    chip_sampling="uniform",
    chip_weight_alpha=4.0,
    hm_context_pattern=None,
    hm_context_stats=(),
    hm_context_radii=(3, 30, 100),
):
    """
    Create a DataLoader for the Human Footprint dataset.

    Args:
        split_mask_file: Path to split mask GeoTIFF (e.g., 'data/raw/hm_global/split_mask_1000.tif')
        split_value: Which split to use (1=train, 2=val, 3=test, 4=calib, None=all data)
        exclude_split_values: Values to exclude (e.g. [3] to train on everything except fold 3).
            A chip is dropped if any pixel in it belongs to an excluded value.
    """
    ds = HumanFootprintChipDataset(
        hm_files,
        component_files,
        static_files,
        chip_size=chip_size,
        timesteps=timesteps,
        stride=stride,
        mode=mode,
        chips_per_epoch=chips_per_epoch,
        fixed_input_years=fixed_input_years,
        fixed_target_years=fixed_target_years,
        use_temporal_sampling=use_temporal_sampling,
        end_year_options=end_year_options,
        min_valid_ratio=min_valid_ratio,
        stat_samples=stat_samples,
        enforce_input_hm_valid=enforce_input_hm_valid,
        include_components=include_components,
        static_channels=static_channels,
        split_mask_file=split_mask_file,
        split_value=split_value,
        exclude_split_values=exclude_split_values,
        chip_weights=chip_weights,
        chip_sampling=chip_sampling,
        chip_weight_alpha=chip_weight_alpha,
        hm_context_pattern=hm_context_pattern,
        hm_context_stats=hm_context_stats,
        hm_context_radii=hm_context_radii,
        norm_stats=norm_stats,
        context_pattern=context_pattern,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

if __name__ == "__main__":
    loader = get_dataloader(batch_size=2, chip_size=128, timesteps=3, chips_per_epoch=2)
    batch = next(iter(loader))
    print(f"input_dynamic shape: {batch['input_dynamic'].shape}")  # [B, 3, 128, 128]
    print(f"input_static shape: {batch['input_static'].shape}")    # [B, 1, 128, 128]
    print(f"target shape: {batch['target'].shape}")               # [B, 128, 128]
    print(f"target timestep index: {batch['timestep']}")
    ds = loader.dataset
    print("Fixed input years:", getattr(ds, 'fixed_input_years', None))
    print("Fixed target year:", getattr(ds, 'fixed_target_year', None))
