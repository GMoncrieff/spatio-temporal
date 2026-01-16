import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


def _open_geotiff(path: Path, chunks: dict) -> xr.DataArray:
    # Remove _1000 suffix from filename if present
    clean_name = path.name.replace('_1000', '')
    print(f"Opening GeoTIFF: {clean_name}")
    try:
        import rioxarray  # noqa: F401

        da = xr.open_dataarray(path, engine="rasterio", chunks=chunks)
    except Exception:
        ds = xr.open_dataset(path, engine="rasterio", chunks=chunks)
        if len(ds.data_vars) != 1:
            raise ValueError(f"Expected 1 band variable in {clean_name}, found {list(ds.data_vars)}")
        da = next(iter(ds.data_vars.values()))

    if "band" in da.dims:
        if da.sizes.get("band", 1) != 1:
            raise ValueError(f"Expected a single band in {clean_name}, got band={da.sizes['band']}")
        da = da.isel(band=0, drop=True)

    if set(da.dims) != {"y", "x"}:
        raise ValueError(f"Expected dims (y, x) after band selection for {clean_name}, got {da.dims}")

    return da


def build_static_dataset(input_dir: Path, chunks_xy: dict) -> xr.Dataset:
    print(f"\n=== Building static dataset from {input_dir} ===")
    static_files = sorted(list(input_dir.glob("hm_static_*.tif")) + list(input_dir.glob("hm_static_*.tiff")))
    if not static_files:
        raise FileNotFoundError(f"No static GeoTIFFs found in {input_dir} matching hm_static_*.tif[f]")

    print(f"Found {len(static_files)} static files")
    data_vars = {}
    for fp in static_files:
        name = fp.stem
        # Remove _1000 suffix from variable name if present
        clean_name = name.replace('_1000', '')
        m = re.match(r"^hm_static_(?P<var>.+)$", clean_name)
        if not m:
            print(f"  Skipping {clean_name} (doesn't match pattern)")
            continue
        var_name = m.group("var")
        print(f"  Adding variable: {var_name} from {clean_name}")
        da = _open_geotiff(fp, chunks=chunks_xy)
        data_vars[var_name] = da

    if not data_vars:
        raise ValueError(f"No static variables could be parsed from files in {input_dir}")

    print(f"Created static dataset with variables: {list(data_vars.keys())}")
    ds = xr.Dataset(data_vars)
    ds = ds.chunk({"y": 128, "x": 128})
    print(f"Chunked static dataset: {ds.chunks}")
    return ds


def build_dynamic_datasets(input_dir: Path, chunks_xy: dict) -> dict:
    print(f"\n=== Building dynamic datasets from {input_dir} ===")
    dyn_files = sorted(list(input_dir.glob("HM_*_*.tif")) + list(input_dir.glob("HM_*_*.tiff")))
    if not dyn_files:
        raise FileNotFoundError(f"No dynamic GeoTIFFs found in {input_dir} matching HM_*_*.tif[f]")

    print(f"Found {len(dyn_files)} dynamic files")
    groups = defaultdict(list)  # var -> [(year, path)]
    pat = re.compile(r"^HM_(?P<year>\d{4})_(?P<var>[A-Za-z0-9]+)(?:_.*)?$")

    for fp in dyn_files:
        name = fp.stem
        # Remove _1000 suffix from filename if present
        clean_name = name.replace('_1000', '')
        m = pat.match(clean_name)
        if not m:
            print(f"  Skipping {clean_name} (doesn't match pattern)")
            continue
        year = int(m.group("year"))
        var = m.group("var")
        groups[var].append((year, fp))

    if not groups:
        raise ValueError(f"No dynamic variables could be parsed from files in {input_dir}")

    print(f"Grouped files by variable: {list(groups.keys())}")
    for var, year_paths in groups.items():
        print(f"  {var}: {len(year_paths)} years")

    datasets = {}
    for var, year_paths in sorted(groups.items()):
        print(f"\nProcessing variable: {var}")
        year_paths_sorted = sorted(year_paths, key=lambda t: t[0])
        years = [y for y, _ in year_paths_sorted]
        times = np.array([np.datetime64(f"{y}-01-01") for y in years])
        print(f"  Years: {years}")

        das = []
        for year, fp in year_paths_sorted:
            # Remove _1000 suffix from filename for display
            clean_fp_name = fp.name.replace('_1000', '')
            print(f"    Loading {year}: {clean_fp_name}")
            da_xy = _open_geotiff(fp, chunks=chunks_xy)
            da_xy = da_xy.expand_dims(time=[np.datetime64(f"{year}-01-01")])
            das.append(da_xy)

        print(f"  Concatenating {len(das)} time steps for {var}")
        da = xr.concat(das, dim="time", join="exact")
        da = da.assign_coords(time=times)
        print(f"  Shape for {var}: {da.dims} -> {da.sizes}")
        
        # Create dataset for this variable
        ds = xr.Dataset({var: da})
        ds = ds.chunk({"time": 1, "y": 128, "x": 128})
        print(f"  Chunked dataset for {var}: {ds.chunks}")
        datasets[var] = ds

    print(f"\nCreated {len(datasets)} dynamic datasets: {list(datasets.keys())}")
    return datasets


def build_combined_dataset(ds_static: xr.Dataset, ds_dynamic_datasets: dict) -> xr.Dataset:
    print(f"\n=== Building combined dataset ===")
    print(f"Static variables: {list(ds_static.data_vars.keys())}")
    print(f"Dynamic variables: {list(ds_dynamic_datasets.keys())}")
    
    # Merge all dynamic datasets first
    all_dynamic_das = []
    for var, ds in ds_dynamic_datasets.items():
        da = ds[var]
        all_dynamic_das.append(da)
    
    # Create combined dataset by merging static and all dynamic
    # Static variables will have dimensions (y, x)
    # Dynamic variables will have dimensions (time, y, x)
    ds_combined = ds_static.copy()
    for da in all_dynamic_das:
        ds_combined[da.name] = da
    
    print(f"Combined dataset variables: {list(ds_combined.data_vars.keys())}")
    print(f"Combined dataset dimensions: {dict(ds_combined.dims)}")
    
    # Apply chunking - keep same chunking as individual datasets
    ds_combined = ds_combined.chunk({"time": 1, "y": 128, "x": 128})
    print(f"Chunked combined dataset: {ds_combined.chunks}")
    
    return ds_combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path("data") / "raw" / "hm_global"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("data") / "raw" / "zarrs"),
    )
    parser.add_argument("--static_store", type=str, default="hm_global_static.zarr")
    parser.add_argument("--dynamic_store_prefix", type=str, default="hm_global_dynamic_")
    parser.add_argument("--combined_store", type=str, default="hm_global_combined.zarr")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"CONVERTING GEOTIFFS TO ZARR")
    print(f"{'='*60}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Static store: {args.static_store}")
    print(f"Dynamic store prefix: {args.dynamic_store_prefix}")
    print(f"Combined store: {args.combined_store}")
    print(f"Overwrite: {args.overwrite}")
    print(f"{'='*60}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_xy = {"x": 128, "y": 128}

    static_store_path = output_dir / args.static_store
    combined_store_path = output_dir / args.combined_store

    if static_store_path.exists() and not args.overwrite:
        raise FileExistsError(f"{static_store_path} exists. Use --overwrite to replace it.")
    if combined_store_path.exists() and not args.overwrite:
        raise FileExistsError(f"{combined_store_path} exists. Use --overwrite to replace it.")

    # Build datasets
    ds_static = build_static_dataset(input_dir=input_dir, chunks_xy=chunks_xy)
    ds_dynamic_datasets = build_dynamic_datasets(input_dir=input_dir, chunks_xy=chunks_xy)
    
    # Check for existing dynamic zarrs
    for var_name in ds_dynamic_datasets.keys():
        dynamic_store_path = output_dir / f"{args.dynamic_store_prefix}{var_name}.zarr"
        if dynamic_store_path.exists() and not args.overwrite:
            raise FileExistsError(f"{dynamic_store_path} exists. Use --overwrite to replace it.")
    
    # Build combined dataset
    ds_combined = build_combined_dataset(ds_static=ds_static, ds_dynamic_datasets=ds_dynamic_datasets)

    # Write to zarr
    print(f"\n{'='*60}")
    print(f"WRITING ZARR ARCHIVES")
    print(f"{'='*60}")
    print(f"Writing static dataset to: {static_store_path}")
    ds_static.to_zarr(static_store_path, mode="w", consolidated=True)
    print(f"✓ Static zarr written successfully")

    print(f"\nWriting dynamic datasets:")
    for var_name, ds in ds_dynamic_datasets.items():
        dynamic_store_path = output_dir / f"{args.dynamic_store_prefix}{var_name}.zarr"
        print(f"  Writing {var_name} to: {dynamic_store_path}")
        ds.to_zarr(dynamic_store_path, mode="w", consolidated=True)
        print(f"  ✓ {var_name} zarr written successfully")

    print(f"\nWriting combined dataset to: {combined_store_path}")
    ds_combined.to_zarr(combined_store_path, mode="w", consolidated=True)
    print(f"✓ Combined zarr written successfully")

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Static zarr: {static_store_path}")
    print(f"Dynamic zarrs:")
    for var_name in ds_dynamic_datasets.keys():
        dynamic_store_path = output_dir / f"{args.dynamic_store_prefix}{var_name}.zarr"
        print(f"  {var_name}: {dynamic_store_path}")
    print(f"Combined zarr: {combined_store_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
