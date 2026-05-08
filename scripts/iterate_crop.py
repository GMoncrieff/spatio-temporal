from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop all GeoTIFF files in a directory to a GeoJSON region."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input GeoTIFF files (.tif/.tiff).",
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        required=True,
        help="GeoJSON file with one or more polygon features used for cropping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write cropped GeoTIFF files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def load_geometries(geojson_path: Path) -> tuple[list[dict], str]:
    with geojson_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    geometries = [f["geometry"] for f in features if f.get("geometry") is not None]
    if not geometries:
        raise ValueError(f"No valid geometries found in {geojson_path}")

    crs_obj = geojson.get("crs") or {}
    crs_props = crs_obj.get("properties", {}) if isinstance(crs_obj, dict) else {}
    geojson_crs = crs_props.get("name", "EPSG:4326")
    return geometries, geojson_crs


def find_geotiff_files(input_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )


def crop_one_raster(
    src_path: Path,
    dst_path: Path,
    geometries: list[dict],
    geojson_crs: str,
) -> None:
    with rasterio.open(src_path) as src:
        if src.crs is not None and str(src.crs) != str(geojson_crs):
            geom_in_src_crs = [
                transform_geom(geojson_crs, src.crs, geom) for geom in geometries
            ]
        else:
            geom_in_src_crs = geometries

        cropped, transform = mask(src, geom_in_src_crs, crop=True)
        meta = src.meta.copy()
        meta.update(
            {
                "height": cropped.shape[1],
                "width": cropped.shape[2],
                "transform": transform,
            }
        )

        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(cropped)


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.geojson.exists() or not args.geojson.is_file():
        raise FileNotFoundError(f"GeoJSON file not found: {args.geojson}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    geometries, geojson_crs = load_geometries(args.geojson)
    tiffs = find_geotiff_files(args.input_dir)
    if not tiffs:
        raise ValueError(f"No GeoTIFF files found in: {args.input_dir}")

    print(f"Found {len(tiffs)} GeoTIFF files")
    written = 0
    skipped = 0

    for src_path in tiffs:
        dst_path = args.output_dir / src_path.name
        if dst_path.exists() and not args.overwrite:
            skipped += 1
            print(f"Skipping existing: {dst_path.name}")
            continue

        crop_one_raster(src_path, dst_path, geometries, geojson_crs)
        written += 1
        print(f"Cropped: {src_path.name} -> {dst_path.name}")

    print(
        f"Done. Wrote {written} file(s) to {args.output_dir}. "
        f"Skipped {skipped} existing file(s)."
    )


if __name__ == "__main__":
    main()
