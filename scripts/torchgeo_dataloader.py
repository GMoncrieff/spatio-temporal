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
        
        # Compute CHANGE (delta) statistics from 20-YEAR HORIZON
        # Use 20-year statistics for all horizons (covers maximum expected changes)
        print("Computing delta statistics from 20-year horizon...")
        delta_samples = []
        # Open HM files temporarily for stats
        with rasterio.open(self._hm_files[0]) as src:
            Hs, Ws = src.height, src.width
        num_delta_samples = 500
        offset_20yr = 4  # 20-year horizon (4 timesteps)
        
        for _ in range(num_delta_samples):
            if Hs < self.chip_size or Ws < self.chip_size:
                i, j = 0, 0
            else:
                i = int(rng.integers(0, Hs - self.chip_size + 1))
                j = int(rng.integers(0, Ws - self.chip_size + 1))
            for t1 in range(len(self._hm_files) - offset_20yr):
                t2 = t1 + offset_20yr
                w = rasterio.windows.Window(j, i, self.chip_size, self.chip_size)
                with rasterio.open(self._hm_files[t1]) as src1:
                    h1 = src1.read(1, window=w, masked=True).filled(np.nan)
                with rasterio.open(self._hm_files[t2]) as src2:
                    h2 = src2.read(1, window=w, masked=True).filled(np.nan)
                d = h2 - h1
                vd = d[np.isfinite(d)]
                if len(vd) > 0:
                    delta_samples.extend(vd.flatten())
        delta_samples = np.array(delta_samples)
        self.delta_mean = np.mean(delta_samples)
        self.delta_std = np.std(delta_samples) + 1e-8
        print(f"HM: mean={self.hm_mean:.4f}, std={self.hm_std:.4f}")
        print(f"Delta (20yr): mean={self.delta_mean:.6f}, std={self.delta_std:.6f}")
        print("  (Applied uniformly to all horizons: 5yr, 10yr, 15yr, 20yr)")
        
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
        
        # Precompute all chip positions if not random
        if self.mode == "grid":
            self.chip_positions = []
            for t in self.valid_time_idxs:
                for i in range(0, self.H - chip_size + 1, stride):
                    for j in range(0, self.W - chip_size + 1, stride):
                        self.chip_positions.append((t, i, j))
        else:
            self.chip_positions = None

        self.C_comp = len(self._comp_files[years[0]]) if self.include_components else 0
        self.C_dyn = 1 + self.C_comp
        self.C_static = len(self._static_files)

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
                    # Randomly select a valid chip position
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
            
            # Get last input HM (RAW, for computing changes)
            last_input_idx = input_t_idxs[-1]
            last_input_raw = self._hm_srcs[last_input_idx].read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
            
            # Multi-horizon targets (computed from end_year)
            # TARGETS ARE NOW NORMALIZED CHANGES, NOT ABSOLUTE HM
            targets = {}
            horizon_names = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
            all_valid = False
            
            for horizon_name, t_idx, target_year in zip(horizon_names, target_t_idxs, target_years):
                if t_idx is None or target_year > 2020:
                    # Missing year - fill with NaN (will be masked in loss)
                    target_delta = np.full((self.chip_size, self.chip_size), np.nan, dtype=np.float32)
                else:
                    # Read target HM (RAW)
                    target_hm_raw = self._hm_srcs[t_idx].read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
                    # Compute CHANGE in original space
                    delta_raw = target_hm_raw - last_input_raw
                    # NO NORMALIZATION - use raw changes directly (TEST)
                    target_delta = delta_raw
                targets[horizon_name] = torch.from_numpy(target_delta).float()
                if not np.isnan(target_delta).all():
                    all_valid = True
            
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
                }
                sample.update(targets)  # Add all horizon targets
                return sample
        # If all attempts fail, return anyway (will be masked out in loss)
        sample = {
            "input_dynamic": torch.from_numpy(input_dynamic).float(),
            "input_static": torch.from_numpy(input_static).float(),
            "lonlat": torch.from_numpy(lonlat).float(),
            "timestep": t,
            # NEW: Year metadata for visualization
            "input_years": input_years,
            "target_years": target_years,
            "end_year": end_year,
        }
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
):
    """
    Create a DataLoader for the Human Footprint dataset.
    
    Args:
        split_mask_file: Path to split mask GeoTIFF (e.g., 'data/raw/hm_global/split_mask_1000.tif')
        split_value: Which split to use (1=train, 2=val, 3=test, 4=calib, None=all data)
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
