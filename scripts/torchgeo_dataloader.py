import os
from torch.utils.data import DataLoader
import torch

# Paths (updated to hm_medium layout)
HM_DIR = os.path.join("data", "raw", "hm_medium")
STATIC_DIR = os.path.join("data", "raw", "hm_medium")

# List of years for which we have human footprint data
years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
# Human modification total per year
hm_files = [os.path.join(HM_DIR, f"HM_{year}_300.tif") for year in years]
# Human modification components per year (must match files available)
COMPONENT_CODES = ["AG", "BU", "EX", "FR", "HI", "NS", "PO", "TI"]
component_files = {
    year: [os.path.join(HM_DIR, f"HM_{year}_{code}_300.tif") for code in COMPONENT_CODES]
    for year in years
}
# Static covariates available in hm_medium
static_files = [
    os.path.join(STATIC_DIR, "SRTM_Elevation_300.tif"),
    os.path.join(STATIC_DIR, "SRTM_Slope_300.tif"),
    os.path.join(STATIC_DIR, "WCv1_BIO05_TmaxWarmest_C_300.tif"),
    os.path.join(STATIC_DIR, "WCv1_BIO06_TminColdest_C_300.tif"),
    os.path.join(STATIC_DIR, "WCv1_BIO12_AnnualPrecip_mm_300.tif"),
]

import numpy as np
import rasterio

class HumanFootprintChipDataset(torch.utils.data.Dataset):
    """
    Memory-safe dataset that lazily reads chip windows from GeoTIFFs.
    - input_dynamic: [T(=3), C_dyn(=1+len(components)), h, w]
    - input_static: [C_static, h, w]
    - target: [h, w] (HM at target year)
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
        stat_samples=256,
        stat_sample_size=512,
        random_seed=42,
        min_valid_ratio=0.8,
        use_validity_filter=True,
        enforce_input_hm_valid=True,
    ):
        self.hm_files = hm_files
        self.component_files = component_files
        self.static_files = static_files
        self.chip_size = chip_size
        self.timesteps = timesteps
        if self.timesteps != 3:
            raise ValueError("Single-step setup expects exactly 3 input timesteps (1990, 1995, 2000)")
        self.stride = stride
        self.mode = mode
        self.chips_per_epoch = chips_per_epoch
        self.fixed_input_years = tuple(fixed_input_years)
        self.fixed_target_year = int(fixed_target_year)
        self.rng = np.random.default_rng(random_seed)
        self.min_valid_ratio = float(min_valid_ratio)
        self.use_validity_filter = bool(use_validity_filter)
        self.enforce_input_hm_valid = bool(enforce_input_hm_valid)

        # Open raster handles (do not read data here)
        self.hm_ds = [rasterio.open(f) for f in hm_files]
        self.comp_ds = {y: [rasterio.open(f) for f in component_files[y]] for y in years}
        self.static_ds = [rasterio.open(f) for f in static_files]

        # Assume all rasters share the same shape
        self.H = self.hm_ds[0].height
        self.W = self.hm_ds[0].width
        self.C_comp = len(self.comp_ds[years[0]])
        self.C_dyn = 1 + self.C_comp
        self.C_static = len(self.static_ds)

        # Compute normalization statistics from random samples (memory-light)
        hm_vals = []
        comp_vals = [list() for _ in range(self.C_comp)]
        static_vals = [list() for _ in range(self.C_static)]
        sample_hw = min(stat_sample_size, self.chip_size)
        for _ in range(stat_samples):
            i = int(self.rng.integers(0, max(1, self.H - sample_hw + 1)))
            j = int(self.rng.integers(0, max(1, self.W - sample_hw + 1)))
            window = rasterio.windows.Window(j, i, sample_hw, sample_hw)
            # Sample HM totals across all years
            for ds in self.hm_ds:
                arr = ds.read(1, window=window, masked=True)
                hm_vals.append(arr.filled(np.nan))
            # Sample components (use same year set as hm years)
            for k in range(self.C_comp):
                # pick a random year per sample to reduce bias across years
                y = years[int(self.rng.integers(0, len(years)))]
                arr = self.comp_ds[y][k].read(1, window=window, masked=True)
                comp_vals[k].append(arr.filled(np.nan))
            # Sample statics
            for s, ds in enumerate(self.static_ds):
                arr = ds.read(1, window=window, masked=True)
                static_vals[s].append(arr.filled(np.nan))

        hm_stack = np.stack(hm_vals, axis=0)
        self.hm_mean = float(np.nanmean(hm_stack))
        self.hm_std = float(np.nanstd(hm_stack) + 1e-8)
        comp_means = []
        comp_stds = []
        for k in range(self.C_comp):
            cstack = np.stack(comp_vals[k], axis=0)
            comp_means.append(np.nanmean(cstack))
            comp_stds.append(np.nanstd(cstack) + 1e-8)
        self.comp_mean = np.array(comp_means, dtype=np.float32)
        self.comp_std = np.array(comp_stds, dtype=np.float32)
        static_means = []
        static_stds = []
        for s in range(self.C_static):
            sstack = np.stack(static_vals[s], axis=0)
            static_means.append(np.nanmean(sstack))
            static_stds.append(np.nanstd(sstack) + 1e-8)
        self.static_mean = np.array(static_means, dtype=np.float32)
        self.static_std = np.array(static_stds, dtype=np.float32)

        # Year indexing helpers
        year_to_idx = {y: i for i, y in enumerate(years)}
        self.years = years
        try:
            self.input_t_idxs = [year_to_idx[y] for y in self.fixed_input_years]
            self.target_t_idx = year_to_idx[self.fixed_target_year]
        except KeyError as e:
            raise ValueError(f"Requested year {e} not found in available years {years}")

        # Precompute valid chip positions based on target-year mask
        self.valid_positions = []
        target_idx = years.index(self.fixed_target_year)
        ds_tgt = self.hm_ds[target_idx]
        for i in range(0, self.H - chip_size + 1, stride):
            for j in range(0, self.W - chip_size + 1, stride):
                window = rasterio.windows.Window(j, i, chip_size, chip_size)
                tgt = ds_tgt.read(1, window=window, masked=True).filled(np.nan)
                if tgt.size == 0:
                    continue
                valid_ratio = 1.0 - float(np.isnan(tgt).mean())
                if valid_ratio >= self.min_valid_ratio:
                    self.valid_positions.append((i, j))

        # Grid mode uses all valid positions; random may sample from them if enabled
        if self.mode == "grid":
            self.chip_positions = list(self.valid_positions)
        else:
            self.chip_positions = None

        # Expose channels
        self.dynamic_channels = self.C_dyn
        self.static_channels = self.C_static

    def __len__(self):
        if self.mode == "grid":
            return len(self.chip_positions)
        else:
            return self.chips_per_epoch

    def __getitem__(self, idx):
        # Try up to 30 times to get a chip with sufficient valid pixels
        for _ in range(30):
            if self.mode == "grid":
                i, j = self.chip_positions[idx]
            else:
                if self.use_validity_filter and len(self.valid_positions) > 0:
                    i, j = self.valid_positions[int(self.rng.integers(0, len(self.valid_positions)))]
                else:
                    i = int(self.rng.integers(0, max(1, self.H - self.chip_size + 1)))
                    j = int(self.rng.integers(0, max(1, self.W - self.chip_size + 1)))
            window = rasterio.windows.Window(j, i, self.chip_size, self.chip_size)

            # Build dynamic input for the three fixed input years
            dyn_list = []
            # Track validity across input years: HM & all component channels
            dyn_valid_list = []
            for y in self.fixed_input_years:
                # HM channel
                hm_idx = years.index(y)
                hm_raw = self.hm_ds[hm_idx].read(1, window=window, masked=True).filled(np.nan)
                hm_valid = ~np.isnan(hm_raw)
                hm_chip = (hm_raw - self.hm_mean) / self.hm_std
                # Components for this year
                comps = []
                comp_valid_all = np.ones_like(hm_valid, dtype=bool)
                for k in range(self.C_comp):
                    c_raw = self.comp_ds[y][k].read(1, window=window, masked=True).filled(np.nan)
                    comp_valid_all &= ~np.isnan(c_raw)
                    c = (c_raw - self.comp_mean[k]) / self.comp_std[k]
                    comps.append(c)
                dyn = np.concatenate([hm_chip[None, ...]] + [c[None, ...] for c in comps], axis=0)
                dyn_list.append(dyn)
                dyn_valid = hm_valid & comp_valid_all
                dyn_valid_list.append(dyn_valid[None, ...])  # [1, H, W]
            input_dynamic = np.stack(dyn_list, axis=0)  # [T, C_dyn, h, w]
            dynamic_valid_mask = np.stack(dyn_valid_list, axis=0)  # [T, 1, h, w]

            # Static inputs
            static_list = []
            for s in range(self.C_static):
                st = self.static_ds[s].read(1, window=window, masked=True).filled(np.nan)
                st = (st - self.static_mean[s]) / self.static_std[s]
                static_list.append(st)
            input_static = np.stack(static_list, axis=0)  # [C_static, h, w]

            # Target is HM at target year (normalized with HM stats)
            target_idx = years.index(self.fixed_target_year)
            target_raw = self.hm_ds[target_idx].read(1, window=window, masked=True).filled(np.nan)
            target_valid_mask = ~np.isnan(target_raw)
            target = (target_raw - self.hm_mean) / self.hm_std

            # Enforce minimum valid ratio to avoid NaN-heavy chips
            if self.enforce_input_hm_valid:
                # combine target validity with dynamic validity at last timestep
                combined_valid = target_valid_mask & dynamic_valid_mask[-1, 0]
                valid_ratio = float(np.mean(combined_valid))
            else:
                valid_ratio = 1.0 - float(np.isnan(target_raw).mean())
            if valid_ratio >= self.min_valid_ratio:
                return {
                    "input_dynamic": torch.from_numpy(input_dynamic).float(),
                    "input_static": torch.from_numpy(input_static).float(),
                    "target": torch.from_numpy(target).float(),
                    "dynamic_valid_mask": torch.from_numpy(dynamic_valid_mask.astype(np.bool_)),
                    "target_valid_mask": torch.from_numpy(target_valid_mask[None, ...].astype(np.bool_)),
                    "timestep": target_idx,
                }

        # If all attempts fail, return last sampled (may be NaN-heavy)
        return {
            "input_dynamic": torch.from_numpy(input_dynamic).float(),
            "input_static": torch.from_numpy(input_static).float(),
            "target": torch.from_numpy(target).float(),
            "dynamic_valid_mask": torch.from_numpy(dynamic_valid_mask.astype(np.bool_)),
            "target_valid_mask": torch.from_numpy(target_valid_mask[None, ...].astype(np.bool_)),
            "timestep": target_idx,
        }


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
):
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
    print(f"input_dynamic shape: {batch['input_dynamic'].shape}")  # [B, T, C_dyn, h, w]
    print(f"input_static shape: {batch['input_static'].shape}")    # [B, C_static, h, w]
    print(f"target shape: {batch['target'].shape}")               # [B, h, w]
    print(f"target timestep index: {batch['timestep']}")
    ds = loader.dataset
    print("Fixed input years:", getattr(ds, 'fixed_input_years', None))
    print("Fixed target year:", getattr(ds, 'fixed_target_year', None))
    print("Dynamic channels:", getattr(ds, 'dynamic_channels', None))
    print("Static channels:", getattr(ds, 'static_channels', None))
