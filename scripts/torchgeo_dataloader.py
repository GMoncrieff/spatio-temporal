import os
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset
import torch

# Paths
HM_DIR = os.path.join("data", "raw", "hm")
STATIC_DIR = os.path.join("data", "raw", "static")

# List of years for which we have human footprint data
years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
hm_files = [os.path.join(HM_DIR, f"HM_{year}_1km.tif") for year in years]
static_files = [os.path.join(STATIC_DIR, "SRTM_Elevation_1km.tif")]

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
        static_files,
        chip_size=512,
        timesteps=3,
        stride=256,
        mode="random",
        chips_per_epoch=100,
        fixed_input_years=(1990, 1995, 2000),
        fixed_target_year=2020,
    ):
        self.hm_files = hm_files
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

        # Load all human footprint rasters into [T, H, W]
        self.hm_stack = []
        for f in hm_files:
            with rasterio.open(f) as src:
                arr = src.read(1)
                arr = arr * 10000  # Scale HM from [0,1] to [0,10000] for regression
                self.hm_stack.append(arr)
        self.hm_stack = np.stack(self.hm_stack, axis=0)  # [T, H, W]
        # Compute mean/std for normalization (ignore NaNs)
        self.hm_mean = np.nanmean(self.hm_stack)
        self.hm_std = np.nanstd(self.hm_stack)
        self.hm_stack = (self.hm_stack - self.hm_mean) / self.hm_std

        self.T, self.H, self.W = self.hm_stack.shape
        # Load static variable(s) [C, H, W]
        static_layers = []
        for f in static_files:
            with rasterio.open(f) as src:
                static_layers.append(src.read(1))
        self.static = np.stack(static_layers, axis=0)
        # Compute mean/std for elevation normalization (ignore NaNs)
        self.elev_mean = np.nanmean(self.static)
        self.elev_std = np.nanstd(self.static)
        self.static = (self.static - self.elev_mean) / self.elev_std
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
        # Precompute all chip positions if not random
        if self.mode == "grid":
            self.chip_positions = []
            for t in self.valid_time_idxs:
                for i in range(0, self.H - chip_size + 1, stride):
                    for j in range(0, self.W - chip_size + 1, stride):
                        self.chip_positions.append((t, i, j))
        else:
            self.chip_positions = None

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
                i = np.random.randint(0, self.H - self.chip_size + 1)
                j = np.random.randint(0, self.W - self.chip_size + 1)
            # Build input from fixed earlier years
            input_dynamic = self.hm_stack[self.input_t_idxs, i:i+self.chip_size, j:j+self.chip_size]
            input_static = self.static[:, i:i+self.chip_size, j:j+self.chip_size]
            target = self.hm_stack[self.target_t_idx, i:i+self.chip_size, j:j+self.chip_size]
            if not np.isnan(target).all():
                sample = {
                    "input_dynamic": torch.from_numpy(input_dynamic).float(),
                    "input_static": torch.from_numpy(input_static).float(),
                    "target": torch.from_numpy(target).float(),
                    "timestep": t
                }
                return sample
        # If all attempts fail, return anyway (will be masked out in loss)
        sample = {
            "input_dynamic": torch.from_numpy(input_dynamic).float(),
            "input_static": torch.from_numpy(input_static).float(),
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
):
    ds = HumanFootprintChipDataset(
        hm_files,
        static_files,
        chip_size=chip_size,
        timesteps=timesteps,
        stride=stride,
        mode=mode,
        chips_per_epoch=chips_per_epoch,
        fixed_input_years=fixed_input_years,
        fixed_target_year=fixed_target_year,
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
