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
HM_VARS = ["AG", "BU", "EX", "FR", "HI", "NS", "PO", "TI"]
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
   # os.path.join(STATIC_DIR, "hm_static_dpi_dsi_1000.tiff"),
   # os.path.join(STATIC_DIR, "hm_static_iucn_nostrict_1000.tiff"),
   # os.path.join(STATIC_DIR, "hm_static_iucn_strict_1000.tiff"),
]

import numpy as np
import rasterio

class HumanFootprintChipDataset(torch.utils.data.Dataset):
    """
    Yields chips for ConvLSTM training (single-step, fixed-year forecasting):
      - input_dynamic: [3, H, W] from years (1990, 1995, 2000)
      - input_static: [1, H, W]
      - target: [H, W] for year 2020
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
        fixed_target_year=2020,
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
        if self.timesteps != 3:
            raise ValueError("Single-step setup expects exactly 3 input timesteps (1990, 1995, 2000)")
        self.stride = stride
        self.mode = mode
        self.chips_per_epoch = chips_per_epoch
        # Fixed-year setup
        self.fixed_input_years = tuple(fixed_input_years)
        self.fixed_target_year = int(fixed_target_year)

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
        # Static stats (sample from each static layer)
        static_samples = []
        total_static_samps = max(32, int(stat_samples))
        for f in self._static_files:
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
        static_stack_samp = np.stack(static_samples, axis=0) if static_samples else np.zeros((1,1,1))
        self.elev_mean = np.nanmean(static_stack_samp)
        self.elev_std = np.nanstd(static_stack_samp) + 1e-8
        # Map fixed years to indices in the stacked timeline
        year_to_idx = {y: i for i, y in enumerate(years)}
        # Expose available years for downstream labeling
        self.years = years
        try:
            self.input_t_idxs = [year_to_idx[y] for y in self.fixed_input_years]
            self.target_t_idx = year_to_idx[self.fixed_target_year]
        except KeyError as e:
            raise ValueError(f"Requested year {e} not found in available years {years}")
        # Validate ordering and availability
        if len(self.input_t_idxs) != 3:
            raise ValueError("fixed_input_years must have exactly 3 entries: e.g., (1990, 1995, 2000)")
        # Precompute valid positions (target year only)
        self.valid_time_idxs = [self.target_t_idx]
        
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
            if self.mode == "grid":
                t, i, j = self.chip_positions[idx]
            else:
                # Target year is fixed; randomly sample spatial chips
                t = self.target_t_idx
                
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
            # Build input from fixed earlier years using windowed reads
            self._ensure_open()
            # Dynamic inputs per timestep: base HM plus HM covariates (if enabled)
            dyn_list = []  # list of [C_dyn, H, W] for each timestep
            for t_idx, year in zip(self.input_t_idxs, self.fixed_input_years):
                channels = []
                # Base HM for this timestep
                arr_hm = self._hm_srcs[t_idx].read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
                # Data is already in [0, 1] range
                arr_hm = (arr_hm - self.hm_mean) / self.hm_std
                channels.append(arr_hm)
                # HM covariates for the same year
                if self.include_components and self._comp_srcs.get(year, []):
                    for src in self._comp_srcs[year]:
                        carr = src.read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
                        # Data is already in [0, 1] range
                        carr = (carr - self.hm_mean) / self.hm_std
                        channels.append(carr)
                dyn_list.append(np.stack(channels, axis=0))  # [C_dyn, H, W]
            input_dynamic = np.stack(dyn_list, axis=0)  # [T, C_dyn, H, W]
            # Static layers
            static_list = []
            for src in self._static_srcs:
                sarr = src.read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
                sarr = (sarr - self.elev_mean) / self.elev_std
                static_list.append(sarr)
            input_static = np.stack(static_list, axis=0) if static_list else np.zeros((0, self.chip_size, self.chip_size), dtype=np.float32)
            # Build lon/lat grid [H, W, 2] from target raster transform, reproject to EPSG:4326 if needed
            ref = self._hm_srcs[self.target_t_idx]
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
            # Target (next time)
            target = self._hm_srcs[self.target_t_idx].read(1, window=rasterio.windows.Window(j, i, self.chip_size, self.chip_size), masked=True).filled(np.nan)
            # Data is already in [0, 1] range
            target = (target - self.hm_mean) / self.hm_std
            if not np.isnan(target).all():
                sample = {
                    "input_dynamic": torch.from_numpy(input_dynamic).float(),
                    "input_static": torch.from_numpy(input_static).float(),
                    "lonlat": torch.from_numpy(lonlat).float(),
                    "target": torch.from_numpy(target).float(),
                    "timestep": t
                }
                return sample
        # If all attempts fail, return anyway (will be masked out in loss)
        sample = {
            "input_dynamic": torch.from_numpy(input_dynamic).float(),
            "input_static": torch.from_numpy(input_static).float(),
            "lonlat": torch.from_numpy(lonlat).float(),
            "target": torch.from_numpy(target).float(),
            "timestep": t
        }
        return sample


def get_dataloader(
    batch_size=1,
    chip_size=128,
    timesteps=3,
    stride=64,
    mode="random",
    chips_per_epoch=100,
    fixed_input_years=(1990, 1995, 2000),
    fixed_target_year=2020,
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
        fixed_target_year=fixed_target_year,
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
