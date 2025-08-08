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
    Yields chips for ConvLSTM training:
      - input_dynamic: [timesteps=3, 512, 512]
      - input_static: [1, 512, 512]
      - target: [512, 512] (next timestep)
    """
    def __init__(self, hm_files, static_files, chip_size=512, timesteps=3, stride=256, mode="random", chips_per_epoch=100, future_horizons=0):
        self.hm_files = hm_files
        self.static_files = static_files
        self.chip_size = chip_size
        self.timesteps = timesteps
        self.stride = stride
        self.mode = mode
        self.chips_per_epoch = chips_per_epoch
        self.future_horizons = future_horizons

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
        # Compute valid time indices
        self.valid_time_idxs = list(range(self.timesteps, self.T))
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
                t = np.random.choice(self.valid_time_idxs)
                i = np.random.randint(0, self.H - self.chip_size + 1)
                j = np.random.randint(0, self.W - self.chip_size + 1)
            input_dynamic = self.hm_stack[t - self.timesteps:t, i:i+self.chip_size, j:j+self.chip_size]
            input_static = self.static[:, i:i+self.chip_size, j:j+self.chip_size]
            target = self.hm_stack[t, i:i+self.chip_size, j:j+self.chip_size]
            if not np.isnan(target).all():
                sample = {
                    "input_dynamic": torch.from_numpy(input_dynamic).float(),
                    "input_static": torch.from_numpy(input_static).float(),
                    "target": torch.from_numpy(target).float(),
                    "timestep": t
                }
                # Optionally include future targets up to future_horizons
                if self.future_horizons > 0:
                    # Compute how many future steps are actually available from t
                    available = max(0, self.T - 1 - t)
                    max_future = min(self.future_horizons, available)
                    # Prepare fixed-size container [future_horizons, Hc, Wc] padded with NaNs
                    fut_arr = np.full((self.future_horizons, self.chip_size, self.chip_size), np.nan, dtype=self.hm_stack[0].dtype)
                    for h in range(max_future):
                        fut_arr[h] = self.hm_stack[t + (h + 1), i:i+self.chip_size, j:j+self.chip_size]
                    sample["future_targets"] = torch.from_numpy(fut_arr).float()
                    sample["future_horizons"] = max_future  # how many valid horizons are filled
                    sample["future_horizons_fixed"] = self.future_horizons  # requested K
                return sample
        # If all attempts fail, return anyway (will be masked out in loss)
        sample = {
            "input_dynamic": torch.from_numpy(input_dynamic).float(),
            "input_static": torch.from_numpy(input_static).float(),
            "target": torch.from_numpy(target).float(),
            "timestep": t
        }
        if self.future_horizons > 0:
            # Return fixed-size padded tensor of NaNs when we failed to find a non-NaN target
            fut_arr = np.full((self.future_horizons, self.chip_size, self.chip_size), np.nan, dtype=self.hm_stack[0].dtype)
            sample["future_targets"] = torch.from_numpy(fut_arr).float()
            sample["future_horizons"] = 0
            sample["future_horizons_fixed"] = self.future_horizons
        return sample


def get_dataloader(batch_size=1, chip_size=128, timesteps=3, stride=64, mode="random", chips_per_epoch=100, future_horizons=0):
    ds = HumanFootprintChipDataset(
        hm_files,
        static_files,
        chip_size=chip_size,
        timesteps=timesteps,
        stride=stride,
        mode=mode,
        chips_per_epoch=chips_per_epoch,
        future_horizons=future_horizons,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

if __name__ == "__main__":
    loader = get_dataloader(batch_size=2, chip_size=128, timesteps=3, chips_per_epoch=2)
    batch = next(iter(loader))
    print(f"input_dynamic shape: {batch['input_dynamic'].shape}")  # [B, 3, 128, 128]
    print(f"input_static shape: {batch['input_static'].shape}")    # [B, 1, 128, 128]
    print(f"target shape: {batch['target'].shape}")               # [B, 128, 128]
    print(f"timesteps: {batch['timestep']}")

    years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
    t = batch['timestep']
    print("Target year(s):", [years[ti] for ti in t])
    print("Input years:", [[years[ti-3], years[ti-2], years[ti-1]] for ti in t])
    for batch in loader:
        print(f"Year: {batch['year']}")
        print(f"Image shape (static): {batch['image'].shape}")
        print(f"Target shape (human footprint): {batch['target'].shape}")
        print(f"Target min/max: {batch['target'].min().item()} / {batch['target'].max().item()}")
