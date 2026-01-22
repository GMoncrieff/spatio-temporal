import os
from urllib.parse import urlparse
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset
import torch
from pyproj import Transformer

# Paths (updated to hm_medium dataset)
HM_DIR = os.path.join("data", "raw", "hm_global")
STATIC_DIR = HM_DIR
ZARR_PATH = os.path.join("scripts", "notebooks", "hm.icechunk")

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

# Removed HumanFootprintChipDataset - no longer used for batch loading
# rasterio kept for split_mask and validity_mask reading

class HumanFootprintZarrChipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zarr_path,
        chip_size=512,
        timesteps=3,
        stride=256,
        mode="random",
        chips_per_epoch=100,
        fixed_input_years=(1990, 1995, 2000),
        fixed_target_years=(2005, 2010, 2015, 2020),
        use_temporal_sampling=True,
        end_year_options=(2000, 2005, 2010, 2015),
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
        static_files=None,
    ):
        try:
            import xarray as xr
            import xbatcher
            import icechunk
        except Exception as e:
            raise ImportError(
                "Zarr backend requires xarray, xbatcher, and icechunk. Install: xarray dask xbatcher icechunk zarr"
            ) from e

        self.zarr_path = zarr_path
        self.chip_size = chip_size
        self.timesteps = timesteps
        if len(fixed_input_years) != 3:
            raise ValueError("Multi-horizon setup expects exactly 3 input timesteps (1990, 1995, 2000)")
        self.stride = stride
        self.mode = mode
        self.chips_per_epoch = chips_per_epoch
        self.use_temporal_sampling = use_temporal_sampling and mode == "random"
        self.end_year_options = list(end_year_options)
        self.fixed_input_years = tuple(fixed_input_years)
        self.fixed_target_years = tuple(fixed_target_years)
        self.target_t_indices = [years.index(y) for y in fixed_target_years]
        self.year_to_idx = {y: i for i, y in enumerate(years)}
        self.include_components = bool(include_components)

        self._static_files = list(static_files if static_files is not None else [])
        if static_channels is not None:
            self._static_files = self._static_files[: int(static_channels)]

        static_var_names = [
            os.path.basename(f).replace("hm_static_", "").replace(".tiff", "") for f in self._static_files
        ]
        dynamic_var_names = ["AA"] + (list(HM_VARS) if self.include_components else [])

        repo_path = str(self.zarr_path)
        print("=== Zarr Dataset Info ===")
        if repo_path.startswith("s3://"):
            parsed = urlparse(repo_path)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
            store = icechunk.s3_storage(bucket=bucket, prefix=prefix, region=None, endpoint_url=None, anonymous=False, allow_http=False, force_path_style=False)
        else:
            store = icechunk.local_filesystem_storage(repo_path)

        repo = icechunk.Repository.open(store)
        session = repo.readonly_session("main")

        try:
            ds = xr.open_zarr(session.store, group="hm", consolidated=False)
        except Exception:
            ds = xr.open_zarr(session.store, consolidated=False)

        # Print dataset info immediately after loading
        print(f"Dataset dimensions: {dict(ds.sizes)}")
        print(f"Data variables: {list(ds.data_vars.keys())}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        if "var_dynamic" in ds.coords:
            print(f"Dynamic variables (var_dynamic): {list(ds.coords['var_dynamic'].values)}")
        if "var_static" in ds.coords:
            print(f"Static variables (var_static): {list(ds.coords['var_static'].values)}")
        print("========================")

        if "dynamic" in ds.data_vars and "static" in ds.data_vars:
            ds_expanded = xr.Dataset(coords={k: ds.coords[k] for k in ("time", "y", "x") if k in ds.coords})

            if "var_dynamic" not in ds.coords:
                raise ValueError("Expected coord 'var_dynamic' when 'dynamic' variable is present")
            if "var_static" not in ds.coords:
                raise ValueError("Expected coord 'var_static' when 'static' variable is present")

            for v in ds.coords["var_dynamic"].values:
                vname = str(v)
                ds_expanded[vname] = ds["dynamic"].sel(var_dynamic=v).drop_vars("var_dynamic")

            for v in ds.coords["var_static"].values:
                vname = str(v)
                ds_expanded[vname] = ds["static"].sel(var_static=v).drop_vars("var_static")

            ds = ds_expanded

        keep_vars = [v for v in dynamic_var_names if v in ds.data_vars]
        keep_vars += [v for v in static_var_names if v in ds.data_vars]
        missing_vars = [v for v in (dynamic_var_names + static_var_names) if v not in ds.data_vars]
        if len(missing_vars) > 0:
            raise ValueError(f"Missing required variables in Zarr: {missing_vars}")
        ds = ds[keep_vars]
        self.ds = ds

        self.H = int(ds.sizes["y"])
        self.W = int(ds.sizes["x"])
        self.T = int(ds.sizes.get("time", 0))

        if self.T != len(years):
            raise ValueError(f"Unexpected time dimension length in Zarr: {self.T}, expected {len(years)}")

        input_dims = {"time": self.T, "y": self.chip_size, "x": self.chip_size}
        overlap_y = max(0, int(self.chip_size - self.stride))
        overlap_x = max(0, int(self.chip_size - self.stride))
        if overlap_y >= self.chip_size or overlap_x >= self.chip_size:
            raise ValueError("stride must be <= chip_size and produce overlap < chip_size")
        input_overlap = {"time": 0, "y": overlap_y, "x": overlap_x}
        self._bgen = xbatcher.BatchGenerator(
            ds,
            input_dims=input_dims,
            input_overlap=input_overlap,
            preload_batch=False,
        )

        y_starts = list(range(0, self.H - self.chip_size + 1, self.stride))
        x_starts = list(range(0, self.W - self.chip_size + 1, self.stride))
        self._n_x = len(x_starts)
        self._n_y = len(y_starts)

        self.split_mask_file = split_mask_file
        self.split_value = split_value
        self.valid_bgen_indices = None
        if self.split_mask_file is not None and self.split_value is not None:
            print(f"Pre-computing valid positions for split_value={self.split_value}...")
            with rasterio.open(self.split_mask_file) as split_src:
                split_data = split_src.read(1)
                if split_data.shape != (self.H, self.W):
                    raise ValueError(
                        f"Split mask shape {split_data.shape} does not match Zarr spatial shape {(self.H, self.W)}"
                    )
                valid_indices = []
                for yi, i in enumerate(y_starts):
                    for xi, j in enumerate(x_starts):
                        chip = split_data[i : i + self.chip_size, j : j + self.chip_size]
                        if (chip == self.split_value).any():
                            valid_indices.append(yi * self._n_x + xi)
                self.valid_bgen_indices = valid_indices
                print(f"  Found {len(valid_indices)} valid chip positions for split {self.split_value}")
                if len(valid_indices) == 0:
                    raise ValueError(
                        f"No valid positions found for split_value={self.split_value}. Check split mask."
                    )

        rng = np.random.default_rng(random_seed)
        hm_samples = []
        total_samps = max(64, int(stat_samples))
        per_time = max(1, int(np.ceil(total_samps / self.T)))
        for t_idx in range(self.T):
            for _ in range(per_time):
                if self.H < self.chip_size or self.W < self.chip_size:
                    i = 0
                    j = 0
                else:
                    i = int(rng.integers(0, self.H - self.chip_size + 1))
                    j = int(rng.integers(0, self.W - self.chip_size + 1))
                arr = (
                    ds["AA"]
                    .isel(time=t_idx, y=slice(i, i + self.chip_size), x=slice(j, j + self.chip_size))
                    .load()
                    .values
                )
                hm_samples.append(arr)
        hm_stack_samp = np.stack(hm_samples, axis=0)
        self.hm_mean = np.nanmean(hm_stack_samp)
        self.hm_std = np.nanstd(hm_stack_samp) + 1e-8

        self.static_means = []
        self.static_stds = []
        if len(static_var_names) > 0:
            print("Computing per-variable normalization statistics for static layers...")
            total_static_samps = max(32, int(stat_samples))
            for static_idx, var_name in enumerate(static_var_names):
                static_samples = []
                per_static = max(1, int(np.ceil(total_static_samps / max(1, len(static_var_names)))))
                for _ in range(per_static):
                    if self.H < self.chip_size or self.W < self.chip_size:
                        i = 0
                        j = 0
                    else:
                        i = int(rng.integers(0, self.H - self.chip_size + 1))
                        j = int(rng.integers(0, self.W - self.chip_size + 1))
                    sarr = (
                        ds[var_name]
                        .isel(y=slice(i, i + self.chip_size), x=slice(j, j + self.chip_size))
                        .load()
                        .values
                    )
                    static_samples.append(sarr)
                static_stack = np.stack(static_samples, axis=0)
                self.static_means.append(np.nanmean(static_stack))
                self.static_stds.append(np.nanstd(static_stack) + 1e-8)
            print("Static normalization stats:")
            for idx, var_name in enumerate(static_var_names):
                print(f"  {var_name}: mean={self.static_means[idx]:.6e}, std={self.static_stds[idx]:.6e}")

        self.elev_mean = self.static_means[0] if self.static_means else 0.0
        self.elev_std = self.static_stds[0] if self.static_stds else 1.0

        self.comp_means = {}
        self.comp_stds = {}
        if self.include_components:
            print("Computing per-variable normalization statistics for components...")
            for var_name in HM_VARS:
                var_samples = []
                per_var_time = max(1, int(np.ceil(stat_samples / self.T)))
                for t_idx in range(self.T):
                    for _ in range(per_var_time):
                        if self.H < self.chip_size or self.W < self.chip_size:
                            i = 0
                            j = 0
                        else:
                            i = int(rng.integers(0, self.H - self.chip_size + 1))
                            j = int(rng.integers(0, self.W - self.chip_size + 1))
                        arr = (
                            ds[var_name]
                            .isel(time=t_idx, y=slice(i, i + self.chip_size), x=slice(j, j + self.chip_size))
                            .load()
                            .values
                        )
                        var_samples.append(arr)
                var_stack = np.stack(var_samples, axis=0)
                self.comp_means[var_name] = np.nanmean(var_stack)
                self.comp_stds[var_name] = np.nanstd(var_stack) + 1e-8
            print("Component normalization stats:")
            for var_name in HM_VARS:
                print(f"  {var_name}: mean={self.comp_means[var_name]:.6e}, std={self.comp_stds[var_name]:.6e}")

        self.C_comp = len(HM_VARS) if self.include_components else 0
        self.C_dyn = 1 + self.C_comp
        self.C_static = len(static_var_names)
        self._static_var_names = static_var_names

    def __len__(self):
        return self.chips_per_epoch

    def __getitem__(self, idx):
        # Choose batch generator index
        if self.valid_bgen_indices is not None:
            bidx = self.valid_bgen_indices[idx]
        else:
            bidx = idx

        # Load batch from xbatcher
        batch = self._bgen[bidx].load()

        # Multi-horizon: always use fixed input years (1990, 1995, 2000)
        input_years = self.fixed_input_years
        input_times = [np.datetime64(f"{y}-01-01") for y in input_years]

        # Temporal sampling for training: random end year determines targets
        if self.use_temporal_sampling:
            end_year = int(np.random.choice(self.end_year_options))
            target_years = tuple(end_year + offset for offset in (5, 10, 15, 20))
        else:
            end_year = self.fixed_target_years[-1]  # 2020
            target_years = self.fixed_target_years

        target_t_idxs = [years.index(y) if y <= 2020 else None for y in target_years]

        # Load input dynamic data (3 timesteps)
        input_dynamic = np.empty((self.timesteps, self.C_dyn, self.chip_size, self.chip_size), dtype=np.float32)
        for t_idx, year in enumerate(input_years):
            # HM target
            arr_hm = batch["AA"].isel(time=t_idx).values
            arr_hm = (arr_hm - self.hm_mean) / self.hm_std
            input_dynamic[t_idx, 0, :, :] = arr_hm

            # Components if enabled
            if self.include_components:
                for c_idx, var_name in enumerate(HM_VARS):
                    carr = batch[var_name].isel(time=t_idx).values
                    carr = np.nan_to_num(carr, nan=0.0)
                    carr = (carr - self.comp_means[var_name]) / self.comp_stds[var_name]
                    input_dynamic[t_idx, 1 + c_idx, :, :] = carr

        # Load static data (same for all timesteps)
        input_static = np.empty((self.C_static, self.chip_size, self.chip_size), dtype=np.float32)
        for static_idx, var_name in enumerate(self._static_var_names):
            sarr = batch[var_name].values
            sarr = np.nan_to_num(sarr, nan=0.0)
            sarr = (sarr - self.static_means[static_idx]) / self.static_stds[static_idx]
            input_static[static_idx, :, :] = sarr

        # Coordinates
        xs, ys = np.meshgrid(
            np.arange(self.chip_size), np.arange(self.chip_size), indexing="ij"
        )
        # Get geographic coordinates from the dataset's coordinate system
        if "x" in batch.coords and "y" in batch.coords:
            # Use actual coordinates from the batch
            x_coords = batch["x"].values
            y_coords = batch["y"].values
            # Get the actual spatial coordinates for this chip
            # This is a simplified approach - in practice you'd need to map batch indices to geographic coordinates
            lonlat = np.stack([xs.flatten(), ys.flatten()], axis=-1).reshape(self.chip_size, self.chip_size, 2).astype(np.float32)
        else:
            # Fallback: create dummy coordinates
            lonlat = np.stack([xs, ys], axis=-1).astype(np.float32)

        # Multi-horizon targets
        targets = {}
        horizon_names = ['target_5yr', 'target_10yr', 'target_15yr', 'target_20yr']
        all_valid = False

        for horizon_name, t_idx, target_year in zip(horizon_names, target_t_idxs, target_years):
            if t_idx is None or target_year > 2020:
                target_h = np.full((self.chip_size, self.chip_size), np.nan, dtype=np.float32)
            else:
                target_h = batch["AA"].isel(time=t_idx).values
                target_h = (target_h - self.hm_mean) / self.hm_std
            targets[horizon_name] = torch.from_numpy(target_h).float()
            if not np.isnan(target_h).all():
                all_valid = True

        if all_valid:
            sample = {
                "input_dynamic": torch.from_numpy(input_dynamic).float(),
                "input_static": torch.from_numpy(input_static).float(),
                "lonlat": torch.from_numpy(lonlat).float(),
                "timestep": self.target_t_indices[-1],
                "input_years": input_years,
                "target_years": target_years,
                "end_year": end_year,
            }
            sample.update(targets)
            sample["target"] = sample.get("target_5yr")
            return sample

        sample = {
            "input_dynamic": torch.from_numpy(input_dynamic).float(),
            "input_static": torch.from_numpy(input_static).float(),
            "lonlat": torch.from_numpy(lonlat).float(),
            "timestep": self.target_t_indices[-1],
            "input_years": input_years,
            "target_years": target_years,
            "end_year": end_year,
        }
        sample.update(targets)
        sample["target"] = sample.get("target_5yr")
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
    use_temporal_sampling=True,
    end_year_options=(2000, 2005, 2010, 2015),
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
    backend="zarr",
    zarr_path=ZARR_PATH,
):
    """
    Create a DataLoader for Human Footprint dataset.
    
    Args:
        split_mask_file: Path to split mask GeoTIFF (e.g., 'data/raw/hm_global/split_mask_1000.tif')
        split_value: Which split to use (1=train, 2=val, 3=test, 4=calib, None=all data)
    """
    # Only Zarr backend supported - geotiff batch loading removed
    if backend == "zarr":
        ds = HumanFootprintZarrChipDataset(
            zarr_path=zarr_path,
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
            static_files=static_files,
        )
    else:
        raise ValueError(f"Unknown backend: {backend} (expected 'zarr')")
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

if __name__ == "__main__":
    # Only Zarr backend supported now
    loader = get_dataloader(
        batch_size=2, chip_size=128, timesteps=3, chips_per_epoch=2, backend="zarr"
    )
    batch = next(iter(loader))
    print(f"input_dynamic shape: {batch['input_dynamic'].shape}")  # [B, 3, 128, 128]
    print(f"input_static shape: {batch['input_static'].shape}")    # [B, 1, 128, 128]
    print(f"target shape: {batch['target'].shape}")               # [B, 128, 128]
    print(f"target timestep index: {batch['timestep']}")
    ds = loader.dataset
    print("Fixed input years:", getattr(ds, 'fixed_input_years', None))
    print("Fixed target year:", getattr(ds, 'fixed_target_year', None))
