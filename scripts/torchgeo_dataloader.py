import os
import json
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset
import torch
from pyproj import Transformer

# Paths (updated to hm_medium dataset)
HM_DIR = os.path.join("data", "raw", "hm_global")
STATIC_DIR = HM_DIR
# Some files were originally staged under HM_DIR/smal/ (full-globe rasters,
# same grid). Resolve transparently from either directory.
_FALLBACK_DIRS = [HM_DIR, os.path.join(HM_DIR, "smal")]


def _resolve(name: str) -> str:
    for d in _FALLBACK_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    # Fall back to primary path; rasterio will surface a clear error.
    return os.path.join(HM_DIR, name)


# List of years for which we have human footprint data
years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
hm_files = [_resolve(f"HM_{year}_AA_1000.tiff") for year in years]
# Time-varying HM covariates
HM_VARS = ["AG", "BU", "EX", "FR", "HI", "NS", "PO", "TI", "gdp", "population"]
component_files = {
    y: [_resolve(f"HM_{y}_{v}_1000.tiff") for v in HM_VARS]
    for y in years
}
static_files = [
    _resolve("hm_static_ele_1000.tiff"),
    _resolve("hm_static_tas_1000.tiff"),
    _resolve("hm_static_tasmin_1000.tiff"),
    _resolve("hm_static_pr_1000.tiff"),
    _resolve("hm_static_dpi_dsi_1000.tiff"),
    _resolve("hm_static_iucn_nostrict_1000.tiff"),
    _resolve("hm_static_iucn_strict_1000.tiff"),
]

import numpy as np
import rasterio


def _distance_label(threshold: float) -> str:
    """Format a threshold like 0.10 → 'dist10', 0.40 → 'dist40'."""
    return f"dist{int(round(threshold * 100)):02d}"


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
        target_mode="absolute_4horizons",
        restrict_to_region=None,
        dhm_stats_cache=None,
        add_distance=None,
        distance_thresholds=(0.1, 0.4),
        weighted_sampling=False,
        weight_alpha=1.0,
        dhm_transform="none",
        dhm_log_scale=0.05,
    ):
        self.hm_files = hm_files
        self.component_files = component_files
        self.static_files = static_files
        self.chip_size = chip_size
        self.timesteps = timesteps  # kept for compatibility; must be 3
        self.stride = stride
        self.mode = mode
        self.chips_per_epoch = chips_per_epoch
        if target_mode not in ("absolute_4horizons", "delta_20yr"):
            raise ValueError(
                f"target_mode must be 'absolute_4horizons' or 'delta_20yr', got {target_mode!r}"
            )
        self.target_mode = target_mode
        self.restrict_to_region = restrict_to_region
        if dhm_transform not in ("none", "signed_log1p"):
            raise ValueError(
                f"dhm_transform must be 'none' or 'signed_log1p', got {dhm_transform!r}"
            )
        self.dhm_transform = str(dhm_transform)
        self.dhm_log_scale = float(dhm_log_scale)
        self.dhm_stats_cache = dhm_stats_cache
        self.weighted_sampling = bool(weighted_sampling) and target_mode == "delta_20yr"
        self.weight_alpha = float(weight_alpha)
        self._chip_weights = None  # populated lazily after valid_split_positions is known

        # Distance-to-HM-threshold channels: defaults on for delta_20yr, off otherwise.
        # Distance rasters (e.g. HM_2000_dist10_1000.tiff, HM_2000_dist40_1000.tiff)
        # must be precomputed via scripts/preprocess_distance.py and live alongside
        # the HM_AA / component rasters.
        if add_distance is None:
            add_distance = (target_mode == "delta_20yr")
        self.add_distance = bool(add_distance)
        self.distance_thresholds = (
            tuple(float(t) for t in distance_thresholds) if self.add_distance else ()
        )
        self.n_distance_channels = len(self.distance_thresholds)

        if target_mode == "delta_20yr":
            if list(end_year_options) != [2000] or use_temporal_sampling \
               or tuple(fixed_input_years) != (1990, 1995, 2000) \
               or tuple(fixed_target_years) != (2020,):
                print("[delta_20yr] Forcing input_years=(1990,1995,2000), target=(2020,), end_year_options=[2000], use_temporal_sampling=False (only valid (t, t+20) config in available data)")
            end_year_options = [2000]
            use_temporal_sampling = False
            fixed_input_years = (1990, 1995, 2000)
            fixed_target_years = (2020,)
        if len(fixed_input_years) != 3:
            raise ValueError("Multi-horizon setup expects exactly 3 input timesteps (1990, 1995, 2000)")
        # Temporal sampling setup
        self.use_temporal_sampling = use_temporal_sampling and mode == "random"  # Only for training
        self.end_year_options = list(end_year_options)
        self.fixed_input_years = tuple(fixed_input_years)
        self.fixed_target_years = tuple(fixed_target_years)
        # Get indices for all target years
        self.target_t_indices = [years.index(y) for y in fixed_target_years]
        # Year to index mapping
        self.year_to_idx = {y: i for i, y in enumerate(years)}
        # Use lazy, windowed IO to avoid loading entire rasters into memory
        self.include_components = bool(include_components)
        self._hm_files = list(hm_files)
        self._static_files = list(static_files if static_channels is None else static_files[:int(static_channels)])
        # Effective component variable list (extended with precomputed
        # distance-to-HM-threshold vars when add_distance=True). Those distance
        # rasters get the same per-variable normalization treatment as the
        # original HM components.
        self.hm_vars = list(HM_VARS)
        if self.add_distance and self.distance_thresholds:
            self.hm_vars = self.hm_vars + [
                _distance_label(t) for t in self.distance_thresholds
            ]
        if self.include_components:
            self._comp_files = {
                y: [_resolve(f"HM_{y}_{v}_1000.tiff") for v in self.hm_vars]
                for y in years
            }
        else:
            self._comp_files = {y: [] for y in years}
        # Split mask for train/val/test separation
        self.split_mask_file = split_mask_file
        self.split_value = split_value  # 1=train, 2=val, 3=test, 4=calib
        
        # Lazily opened rasterio datasets (per worker)
        self._hm_srcs = None
        self._static_srcs = None
        self._comp_srcs = None
        # Read raster shape from the first HM file
        with rasterio.open(self._hm_files[0]) as src0:
            self.H, self.W = src0.height, src0.width
        self.T = len(self._hm_files)
        # Estimate normalization stats from random windows (stat_samples)
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
        
        # Component stats (per variable) - critical for GDP/population which have different scales.
        # Note: self.hm_vars may include precomputed distance vars (dist10, dist40) when
        # add_distance=True; those go through the same per-variable normalization pipeline.
        self.comp_means = {}
        self.comp_stds = {}
        if self.include_components:
            print("Computing per-variable normalization statistics for components...")
            for var_idx, var_name in enumerate(self.hm_vars):
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
            for var_name in self.hm_vars:
                print(f"  {var_name}: mean={self.comp_means[var_name]:.6e}, std={self.comp_stds[var_name]:.6e}")

        # Δhm stats (only when target_mode == "delta_20yr")
        self.dhm_mean = 0.0
        self.dhm_std = 1.0
        if self.target_mode == "delta_20yr":
            self._compute_dhm_stats(rng, stat_samples)


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
        
        # Precompute valid positions for split if using split mask
        self.valid_split_positions = None
        if self.split_mask_file is not None and self.split_value is not None:
            print(f"Pre-computing valid positions for split_value={self.split_value}...")
            with rasterio.open(self.split_mask_file) as split_src:
                split_data = split_src.read(1)
                valid_positions = []
                # Use chip_size for sampling (matches split generation)
                for i in range(0, self.H - chip_size + 1, chip_size):
                    for j in range(0, self.W - chip_size + 1, chip_size):
                        chip = split_data[i:i+chip_size, j:j+chip_size]
                        # Check if this chip belongs to the correct split
                        if (chip == self.split_value).any():  # Any pixel in split
                            valid_positions.append((i, j))
                self.valid_split_positions = valid_positions
                print(f"  Found {len(valid_positions)} valid chip positions for split {self.split_value}")
                if len(valid_positions) == 0:
                    raise ValueError(f"No valid positions found for split_value={self.split_value}. Check split mask.")

        # Optionally restrict valid positions to a GeoJSON region
        if self.restrict_to_region is not None:
            print(f"Restricting chip positions to region {self.restrict_to_region}...")
            region_mask = self._rasterize_region_to_mask(self.restrict_to_region)
            if self.valid_split_positions is not None:
                before = len(self.valid_split_positions)
                positions = [(i, j) for (i, j) in self.valid_split_positions
                             if region_mask[i:i+chip_size, j:j+chip_size].any()]
                self.valid_split_positions = positions
                print(f"  Region-restricted: {before} -> {len(positions)} positions")
            else:
                positions = []
                for i in range(0, self.H - chip_size + 1, chip_size):
                    for j in range(0, self.W - chip_size + 1, chip_size):
                        if region_mask[i:i+chip_size, j:j+chip_size].any():
                            positions.append((i, j))
                self.valid_split_positions = positions
                print(f"  Region: {len(positions)} positions")
            if len(self.valid_split_positions) == 0:
                raise ValueError(f"No valid positions inside region {self.restrict_to_region}")

        # Precompute all chip positions if not random
        if self.mode == "grid":
            self.chip_positions = []
            allowed = set(self.valid_split_positions) if self.valid_split_positions is not None else None
            for t in self.valid_time_idxs:
                for i in range(0, self.H - chip_size + 1, stride):
                    for j in range(0, self.W - chip_size + 1, stride):
                        if allowed is None or (i, j) in allowed:
                            self.chip_positions.append((t, i, j))
            if len(self.chip_positions) == 0 and allowed is not None:
                # Fall back: use the explicit allowed positions even if they are off-stride
                for t in self.valid_time_idxs:
                    for (i, j) in self.valid_split_positions:
                        self.chip_positions.append((t, i, j))
        else:
            self.chip_positions = None

        # C_comp already includes distance vars when add_distance is True (they're
        # added to self.hm_vars and self._comp_files above).
        self.C_comp = len(self._comp_files[years[0]]) if self.include_components else 0
        self.C_dyn = 1 + self.C_comp

        # Per-chip importance weights for weighted sampling (delta_20yr only).
        if self.weighted_sampling and self.valid_split_positions:
            self._chip_weights = self._compute_chip_weights()
        self.C_static = len(self._static_files)

    def _apply_dhm_transform(self, x):
        """Forward Δhm transform applied BEFORE z-score normalization.

        Heavy-tailed Δhm distributions get crushed into the noise budget by a
        plain z-score; signed_log1p compresses bulk and stretches the tail so
        the diffusion forward process can resolve extreme magnitudes.
        Inverse must mirror this exactly in the diffusion module's sample().
        """
        if self.dhm_transform == "signed_log1p":
            s = self.dhm_log_scale
            return np.sign(x) * np.log1p(np.abs(x) / s)
        return x

    def _dhm_cache_path(self):
        """Cache key includes transform name + scale so different transforms
        don't share statistics."""
        if not self.dhm_stats_cache:
            return None
        if self.dhm_transform == "none":
            return self.dhm_stats_cache
        base, ext = os.path.splitext(self.dhm_stats_cache)
        suffix = f"_{self.dhm_transform}_s{self.dhm_log_scale:g}"
        return f"{base}{suffix}{ext}"

    def _compute_dhm_stats(self, rng, stat_samples):
        """Compute or load Δhm = HM(2020) - HM(2000) normalization stats."""
        cache_path = self._dhm_cache_path()
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    stats = json.load(f)
                self.dhm_mean = float(stats["dhm_mean"])
                self.dhm_std = float(stats["dhm_std"])
                print(f"Loaded Δhm stats from {cache_path}: mean={self.dhm_mean:.6e} std={self.dhm_std:.6e}")
                return
            except Exception as e:
                print(f"Failed to load Δhm stats cache ({e}); recomputing")

        print("Computing Δhm normalization stats from random (HM_2020 - HM_2000) windows...")
        idx_2000 = self.year_to_idx[2000]
        idx_2020 = self.year_to_idx[2020]
        f_2000 = self._hm_files[idx_2000]
        f_2020 = self._hm_files[idx_2020]
        n_samples = max(64, int(stat_samples))
        samples = []
        with rasterio.open(f_2000) as s00, rasterio.open(f_2020) as s20:
            Hs, Ws = s00.height, s00.width
            for _ in range(n_samples):
                if Hs < self.chip_size or Ws < self.chip_size:
                    i = 0; j = 0
                else:
                    i = int(rng.integers(0, Hs - self.chip_size + 1))
                    j = int(rng.integers(0, Ws - self.chip_size + 1))
                window = rasterio.windows.Window(j, i, self.chip_size, self.chip_size)
                arr00 = s00.read(1, window=window, masked=True).filled(np.nan)
                arr20 = s20.read(1, window=window, masked=True).filled(np.nan)
                samples.append(arr20 - arr00)
        arr = np.stack(samples, axis=0)
        # Apply forward transform BEFORE computing mean/std so the z-score
        # below targets the post-transform distribution.
        arr_t = self._apply_dhm_transform(arr)
        self.dhm_mean = float(np.nanmean(arr_t))
        self.dhm_std = float(np.nanstd(arr_t) + 1e-8)
        print(f"Computed Δhm stats (transform={self.dhm_transform}, scale={self.dhm_log_scale}): "
              f"mean={self.dhm_mean:.6e} std={self.dhm_std:.6e}")
        cache_path = self._dhm_cache_path()
        if cache_path:
            try:
                cache_dir = os.path.dirname(cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({
                        "dhm_mean": self.dhm_mean,
                        "dhm_std": self.dhm_std,
                        "dhm_transform": self.dhm_transform,
                        "dhm_log_scale": self.dhm_log_scale,
                    }, f)
                print(f"Cached Δhm stats to {cache_path}")
            except Exception as e:
                print(f"Failed to write Δhm stats cache: {e}")

    def _compute_chip_weights(self):
        """Per-chip sampling weights ∝ (max |Δhm| in chip)^α.

        Sampling proportional to max |Δhm| concentrates training compute on the
        small minority of chips that contain rare-but-important high-change
        pixels — directly addresses poor coverage in upper Δhm bins. α controls
        sharpness: α=1 ⇒ proportional, α=0 ⇒ uniform.
        """
        idx_2000 = self.year_to_idx[2000]
        idx_2020 = self.year_to_idx[2020]
        n = len(self.valid_split_positions)
        weights = np.zeros(n, dtype=np.float64)
        print(f"Computing weighted-sampling chip weights over {n} valid positions "
              f"(α={self.weight_alpha})...")
        with rasterio.open(self._hm_files[idx_2000]) as s00, \
             rasterio.open(self._hm_files[idx_2020]) as s20:
            for k, (i, j) in enumerate(self.valid_split_positions):
                window = rasterio.windows.Window(j, i, self.chip_size, self.chip_size)
                arr00 = s00.read(1, window=window, masked=True).filled(np.nan)
                arr20 = s20.read(1, window=window, masked=True).filled(np.nan)
                dhm = np.abs(arr20 - arr00)
                if not np.isfinite(dhm).any():
                    weights[k] = 0.0
                else:
                    weights[k] = float(np.nanmax(dhm))
        eps = 1e-3  # so all-zero chips still get a tiny non-zero weight
        weights = (weights + eps) ** self.weight_alpha
        weights = weights / weights.sum()
        # Diagnostic: how concentrated is the sampling?
        sorted_w = np.sort(weights)[::-1]
        top10pct = sorted_w[: max(1, n // 10)].sum()
        print(f"  top 10% of chips will draw {100 * top10pct:.1f}% of sampling probability "
              f"(uniform = 10%)")
        return weights

    def _rasterize_region_to_mask(self, geojson_path):
        """Rasterize a GeoJSON region into a boolean mask in the dataset's reference grid."""
        from shapely.geometry import shape
        from shapely.ops import unary_union, transform as shp_transform
        from rasterio import features as rio_features

        with rasterio.open(self._hm_files[0]) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_h, ref_w = ref.height, ref.width

        with open(geojson_path) as f:
            gj = json.load(f)
        if gj.get("type") == "FeatureCollection":
            geoms = [shape(feat["geometry"]) for feat in gj["features"]]
        elif gj.get("type") == "Feature":
            geoms = [shape(gj["geometry"])]
        else:
            geoms = [shape(gj)]
        geom = unary_union(geoms)

        if ref_crs and ref_crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
            geom = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)

        mask = rio_features.rasterize(
            [(geom, 1)], out_shape=(ref_h, ref_w), transform=ref_transform,
            fill=0, dtype="uint8",
        )
        return mask.astype(bool)

    def _ensure_open(self):
        # Open datasets lazily per worker process
        if self._hm_srcs is None:
            self._hm_srcs = [rasterio.open(f) for f in self._hm_files]
        if self._static_srcs is None:
            self._static_srcs = [rasterio.open(f) for f in self._static_files]
        if self._comp_srcs is None:
            self._comp_srcs = {y: [rasterio.open(f) for f in self._comp_files[y]] for y in years}

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
            
            if self.mode == "grid":
                t, i, j = self.chip_positions[idx]
            else:
                # Use last available target year for grid positioning
                t = self.target_t_indices[-1]  # Use 2020 for grid positioning
                
                # If using splits, sample from pre-computed valid positions
                if self.valid_split_positions is not None:
                    # Sample by chip weight if available, else uniform
                    if self._chip_weights is not None:
                        pos_idx = int(np.random.choice(
                            len(self.valid_split_positions),
                            p=self._chip_weights,
                        ))
                    else:
                        pos_idx = np.random.randint(0, len(self.valid_split_positions))
                    i, j = self.valid_split_positions[pos_idx]
                    # Add small random offset within chip for diversity
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
                arr_hm = self._hm_srcs[t_idx].read(
                    1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size),
                    masked=True,
                ).filled(np.nan)
                # Data is already in [0, 1] range
                arr_hm = (arr_hm - self.hm_mean) / self.hm_std
                channels.append(arr_hm)
                # HM covariates (originals + precomputed distance vars when add_distance)
                if self.include_components and self._comp_srcs.get(year, []):
                    for var_idx, src in enumerate(self._comp_srcs[year]):
                        carr = src.read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
                        var_name = self.hm_vars[var_idx]
                        # Replace NaN with 0 BEFORE normalization for all components
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
            
            common = {
                "input_dynamic": torch.from_numpy(input_dynamic).float(),
                "input_static": torch.from_numpy(input_static).float(),
                "lonlat": torch.from_numpy(lonlat).float(),
                "timestep": t,
                "input_years": input_years,
                "target_years": list(target_years),
                "end_year": end_year,
            }

            if self.target_mode == "delta_20yr":
                idx_2000 = self.year_to_idx[2000]
                idx_2020 = self.year_to_idx[2020]
                arr_2000 = self._hm_srcs[idx_2000].read(
                    1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size),
                    masked=True,
                ).filled(np.nan)
                arr_2020 = self._hm_srcs[idx_2020].read(
                    1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size),
                    masked=True,
                ).filled(np.nan)
                dhm_raw = arr_2020 - arr_2000
                dhm_transformed = self._apply_dhm_transform(dhm_raw)
                target_dhm = (dhm_transformed - self.dhm_mean) / self.dhm_std

                target_finite = np.isfinite(target_dhm)
                dyn_finite = np.isfinite(input_dynamic).all(axis=(0, 1))
                stat_finite = (
                    np.isfinite(input_static).all(axis=0)
                    if input_static.size
                    else np.ones((self.chip_size, self.chip_size), dtype=bool)
                )
                valid_mask = target_finite & dyn_finite & stat_finite

                hm_t_normalized = input_dynamic[2, 0]

                if valid_mask.any():
                    sample = dict(common)
                    sample["target_dhm"] = torch.from_numpy(
                        np.nan_to_num(target_dhm, nan=0.0)[None, ...]
                    ).float()
                    sample["hm_t_normalized"] = torch.from_numpy(
                        np.nan_to_num(hm_t_normalized, nan=0.0)[None, ...]
                    ).float()
                    sample["valid_mask"] = torch.from_numpy(valid_mask)
                    sample["target_year"] = 2020
                    return sample
                continue

            # absolute_4horizons (legacy default): produce target_5yr..target_20yr
            targets = {}
            horizon_names = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
            all_valid = False
            for horizon_name, t_idx, target_year in zip(horizon_names, target_t_idxs, target_years):
                if t_idx is None or target_year > 2020:
                    target_h = np.full((self.chip_size, self.chip_size), np.nan, dtype=np.float32)
                else:
                    target_h = self._hm_srcs[t_idx].read(
                        1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size),
                        masked=True,
                    ).filled(np.nan)
                    target_h = (target_h - self.hm_mean) / self.hm_std
                targets[horizon_name] = torch.from_numpy(target_h).float()
                if not np.isnan(target_h).all():
                    all_valid = True

            if all_valid:
                sample = dict(common)
                sample.update(targets)
                return sample

        # All retries exhausted — return whatever we last built (will be masked downstream).
        sample = dict(common)
        if self.target_mode == "delta_20yr":
            sample["target_dhm"] = torch.from_numpy(
                np.nan_to_num(target_dhm, nan=0.0)[None, ...]
            ).float()
            sample["hm_t_normalized"] = torch.from_numpy(
                np.nan_to_num(hm_t_normalized, nan=0.0)[None, ...]
            ).float()
            sample["valid_mask"] = torch.from_numpy(valid_mask)
            sample["target_year"] = 2020
        else:
            sample.update(targets)
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
    target_mode="absolute_4horizons",
    restrict_to_region=None,
    dhm_stats_cache=None,
    add_distance=None,
    distance_thresholds=(0.1, 0.4),
    weighted_sampling=False,
    weight_alpha=1.0,
    dhm_transform="none",
    dhm_log_scale=0.05,
):
    """
    Create a DataLoader for the Human Footprint dataset.

    Args:
        split_mask_file: Path to split mask GeoTIFF (e.g., 'data/raw/hm_global/split_mask_1000.tif')
        split_value: Which split to use (1=train, 2=val, 3=test, 4=calib, None=all data)
        target_mode: "absolute_4horizons" (legacy) or "delta_20yr" (single Δhm target).
        restrict_to_region: Path to a GeoJSON file. If set, only chip positions whose window
            overlaps the rasterized region are kept.
        dhm_stats_cache: Optional path to cache (mean, std) of Δhm; ignored unless
            target_mode == "delta_20yr".
        add_distance: Whether to read precomputed distance-to-HM-threshold rasters
            as extra component channels (default: True for delta_20yr, False otherwise).
            Rasters must be precomputed via scripts/preprocess_distance.py.
        distance_thresholds: Per-channel HM thresholds. Default (0.1, 0.4).
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
        target_mode=target_mode,
        restrict_to_region=restrict_to_region,
        dhm_stats_cache=dhm_stats_cache,
        add_distance=add_distance,
        distance_thresholds=distance_thresholds,
        weighted_sampling=weighted_sampling,
        weight_alpha=weight_alpha,
        dhm_transform=dhm_transform,
        dhm_log_scale=dhm_log_scale,
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
