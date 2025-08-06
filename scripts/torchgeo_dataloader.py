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

class HumanFootprintDataset(RasterDataset):
    """
    Loads human footprint (dynamic) and static variable rasters for a given year.
    Returns a dict with 'image' (static vars), 'target' (human footprint), and 'year'.
    """
    def __init__(self, hm_file, static_files):
        super().__init__()
        self.hm_file = hm_file
        self.static_files = static_files

        # No need to register self.files for this custom dataset

    def __getitem__(self, index):
        # Ignore index, always return the full raster (single sample)
        import rasterio
        import numpy as np
        # Human footprint (target)
        with rasterio.open(self.hm_file) as src:
            target = src.read(1)
            profile = src.profile
        # Static variable(s)
        static_layers = []
        for f in self.static_files:
            with rasterio.open(f) as s:
                static_layers.append(s.read(1))
        static = np.stack(static_layers, axis=0)  # shape: [C, H, W]
        # Add batch dimension (simulate sample)
        return {
            'image': torch.from_numpy(static).float(),
            'target': torch.from_numpy(target).float().unsqueeze(0),
            'year': int(os.path.basename(self.hm_file).split('_')[1])
        }

    def __len__(self):
        return 1  # Each dataset is a single raster

def get_datasets():
    datasets = []
    for hm_file in hm_files:
        datasets.append(HumanFootprintDataset(hm_file, static_files))
    return datasets

def get_dataloader(year_idx=0, batch_size=1):
    ds = get_datasets()[year_idx]
    return DataLoader(ds, batch_size=batch_size)

if __name__ == "__main__":
    # Example: load 2010 data
    year_idx = 4  # 2010
    loader = get_dataloader(year_idx)
    for batch in loader:
        print(f"Year: {batch['year']}")
        print(f"Image shape (static): {batch['image'].shape}")
        print(f"Target shape (human footprint): {batch['target'].shape}")
        print(f"Target min/max: {batch['target'].min().item()} / {batch['target'].max().item()}")
