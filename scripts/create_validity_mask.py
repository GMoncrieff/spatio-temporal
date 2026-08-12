#!/usr/bin/env python3
"""
Create validity mask and train/val/test/calibration splits for global HM dataset.

Outputs:
- validity_mask_1000.tif: Binary mask (1=valid, 0=invalid)
- split_mask_1000.tif: Split assignments (1=train, 2=val, 3=test, 4=calibration, 0=invalid)
- split_visualization.pdf: Map showing the spatial distribution of splits
"""

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Configuration
HM_DIR = "data/raw/hm_global"
TARGET_FILE = os.path.join(HM_DIR, "HM_2020_AA_1000.tiff")
VALIDITY_MASK_FILE = os.path.join(HM_DIR, "validity_mask_1000.tif")
SPLIT_MASK_FILE = os.path.join(HM_DIR, "split_mask_1000.tif")
FOLD_MASK_FILE = os.path.join(HM_DIR, "fold_mask_1000.tif")
FOLD_MANIFEST_FILE = os.path.join(HM_DIR, "fold_manifest.csv")
VISUALIZATION_FILE = "outputs/split_visualization.pdf"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.10
CALIB_RATIO = 0.10

# Chip size for spatial blocking
CHIP_SIZE = 128
MIN_VALID_RATIO = 0.2  # Minimum 20% valid pixels in a chip

# Random seed for reproducibility
RANDOM_SEED = 42

def create_validity_mask():
    """Create binary validity mask from target raster."""
    print("=" * 80)
    print("Creating Validity Mask")
    print("=" * 80)
    
    with rasterio.open(TARGET_FILE) as src:
        print(f"Reading target file: {TARGET_FILE}")
        print(f"  Shape: {src.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        
        data = src.read(1)
        profile = src.profile.copy()
        
        # Create validity mask (1=valid, 0=invalid)
        # Valid if: not NaN and >= 0
        valid_mask = (~np.isnan(data) & (data >= 0)).astype(np.uint8)
        
        total_pixels = valid_mask.size
        valid_pixels = valid_mask.sum()
        valid_percent = 100 * valid_pixels / total_pixels
        
        print(f"\nValidity Statistics:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Valid pixels: {valid_pixels:,} ({valid_percent:.2f}%)")
        print(f"  Invalid pixels: {total_pixels - valid_pixels:,} ({100 - valid_percent:.2f}%)")
        
        # Update profile for single-band uint8
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw',
            tiled=True,
            blockxsize=512,
            blockysize=512,
            nodata=None,  # Remove float nodata value for uint8 mask
        )
        
        # Write validity mask
        print(f"\nWriting validity mask: {VALIDITY_MASK_FILE}")
        with rasterio.open(VALIDITY_MASK_FILE, 'w', **profile) as dst:
            dst.write(valid_mask, 1)
            dst.set_band_description(1, "Validity: 1=valid, 0=invalid")
        
        print("✓ Validity mask created successfully")
        
        return valid_mask, profile, src.transform, src.crs

def enumerate_valid_chips(valid_mask, chip_size=CHIP_SIZE, min_valid_ratio=MIN_VALID_RATIO):
    """Enumerate (i, j) origins of non-overlapping chips with enough valid pixels.

    Shared by create_spatial_splits (70/10/10/10) and create_kfold_splits (k rotating
    folds) so both partitions are built from exactly the same chip population.
    """
    H, W = valid_mask.shape
    chip_positions = []
    for i in range(0, H - chip_size + 1, chip_size):
        for j in range(0, W - chip_size + 1, chip_size):
            chip = valid_mask[i:i+chip_size, j:j+chip_size]
            valid_ratio = chip.sum() / (chip_size ** 2)

            if valid_ratio >= min_valid_ratio:
                chip_positions.append((i, j))
    return chip_positions


def create_spatial_splits(valid_mask, profile, transform, crs):
    """Create train/val/test/calibration splits using spatial blocking."""
    print("\n" + "=" * 80)
    print("Creating Spatial Splits")
    print("=" * 80)

    H, W = valid_mask.shape
    print(f"Raster dimensions: {H} x {W}")
    print(f"Chip size: {CHIP_SIZE}")
    print(f"Minimum valid ratio per chip: {MIN_VALID_RATIO} ({MIN_VALID_RATIO*100:.0f}%)")

    # Create grid of chip positions
    chip_positions = enumerate_valid_chips(valid_mask)

    print(f"\nFound {len(chip_positions)} valid chips (>={MIN_VALID_RATIO*100:.0f}% valid pixels)")

    # Shuffle and split
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(chip_positions)

    n_chips = len(chip_positions)
    n_train = int(n_chips * TRAIN_RATIO)
    n_val = int(n_chips * VAL_RATIO)
    n_test = int(n_chips * TEST_RATIO)
    n_calib = n_chips - n_train - n_val - n_test  # Remaining goes to calibration
    
    train_chips = chip_positions[:n_train]
    val_chips = chip_positions[n_train:n_train+n_val]
    test_chips = chip_positions[n_train+n_val:n_train+n_val+n_test]
    calib_chips = chip_positions[n_train+n_val+n_test:]
    
    print(f"\nSplit Statistics:")
    print(f"  Train: {len(train_chips):,} chips ({100*len(train_chips)/n_chips:.1f}%)")
    print(f"  Val:   {len(val_chips):,} chips ({100*len(val_chips)/n_chips:.1f}%)")
    print(f"  Test:  {len(test_chips):,} chips ({100*len(test_chips)/n_chips:.1f}%)")
    print(f"  Calib: {len(calib_chips):,} chips ({100*len(calib_chips)/n_chips:.1f}%)")
    
    # Create split mask
    # 0=invalid, 1=train, 2=val, 3=test, 4=calibration
    split_mask = np.zeros((H, W), dtype=np.uint8)
    
    for i, j in train_chips:
        split_mask[i:i+CHIP_SIZE, j:j+CHIP_SIZE] = 1
    
    for i, j in val_chips:
        split_mask[i:i+CHIP_SIZE, j:j+CHIP_SIZE] = 2
    
    for i, j in test_chips:
        split_mask[i:i+CHIP_SIZE, j:j+CHIP_SIZE] = 3
    
    for i, j in calib_chips:
        split_mask[i:i+CHIP_SIZE, j:j+CHIP_SIZE] = 4
    
    # Count pixels per split
    train_pixels = (split_mask == 1).sum()
    val_pixels = (split_mask == 2).sum()
    test_pixels = (split_mask == 3).sum()
    calib_pixels = (split_mask == 4).sum()
    total_split_pixels = train_pixels + val_pixels + test_pixels + calib_pixels
    
    print(f"\nPixel Statistics:")
    print(f"  Train: {train_pixels:,} pixels ({100*train_pixels/total_split_pixels:.1f}%)")
    print(f"  Val:   {val_pixels:,} pixels ({100*val_pixels/total_split_pixels:.1f}%)")
    print(f"  Test:  {test_pixels:,} pixels ({100*test_pixels/total_split_pixels:.1f}%)")
    print(f"  Calib: {calib_pixels:,} pixels ({100*calib_pixels/total_split_pixels:.1f}%)")
    
    # Write split mask
    profile_split = profile.copy()
    profile_split.update(dtype=rasterio.uint8, count=1)
    
    print(f"\nWriting split mask: {SPLIT_MASK_FILE}")
    with rasterio.open(SPLIT_MASK_FILE, 'w', **profile_split) as dst:
        dst.write(split_mask, 1)
        dst.set_band_description(1, "Split: 0=invalid, 1=train, 2=val, 3=test, 4=calib")
    
    print("✓ Split mask created successfully")
    
    return split_mask

def create_kfold_splits(valid_mask, profile, transform, crs, k=5):
    """Create k rotating spatial folds for the out-of-sample hindcast harness.

    Same chip enumeration and same fixed RANDOM_SEED shuffle as create_spatial_splits,
    but sliced into k equal groups instead of ratio cuts. Leaves split_mask_1000.tif
    untouched — that artifact belongs to the already-trained production checkpoint.

    Writes:
      data/raw/hm_global/fold_mask_1000.tif  uint8, 0=invalid, 1..k=fold id
      data/raw/hm_global/fold_manifest.csv   fold_id, n_chips, n_pixels
    """
    print("\n" + "=" * 80)
    print(f"Creating {k}-Fold Spatial Splits")
    print("=" * 80)

    H, W = valid_mask.shape
    chip_positions = enumerate_valid_chips(valid_mask)
    print(f"Found {len(chip_positions)} valid chips (>={MIN_VALID_RATIO*100:.0f}% valid pixels)")

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(chip_positions)

    groups = np.array_split(np.arange(len(chip_positions)), k)

    fold_mask = np.zeros((H, W), dtype=np.uint8)
    rows = []
    for fold_idx, group in enumerate(groups, start=1):
        for idx in group:
            i, j = chip_positions[idx]
            fold_mask[i:i+CHIP_SIZE, j:j+CHIP_SIZE] = fold_idx
        n_pixels = int((fold_mask == fold_idx).sum())
        rows.append({"fold_id": fold_idx, "n_chips": len(group), "n_pixels": n_pixels})
        print(f"  Fold {fold_idx}: {len(group):,} chips, {n_pixels:,} pixels")

    profile_fold = profile.copy()
    profile_fold.update(dtype=rasterio.uint8, count=1, nodata=None)
    print(f"\nWriting fold mask: {FOLD_MASK_FILE}")
    with rasterio.open(FOLD_MASK_FILE, 'w', **profile_fold) as dst:
        dst.write(fold_mask, 1)
        dst.set_band_description(1, f"Fold: 0=invalid, 1..{k}=fold id")

    import csv
    with open(FOLD_MANIFEST_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["fold_id", "n_chips", "n_pixels"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ Fold manifest written: {FOLD_MANIFEST_FILE}")

    return fold_mask


def visualize_splits(split_mask, transform, crs):
    """Create PDF visualization of splits."""
    print("\n" + "=" * 80)
    print("Creating Visualization")
    print("=" * 80)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(VISUALIZATION_FILE), exist_ok=True)
    
    # Downsample for visualization if too large
    H, W = split_mask.shape
    max_dim = 4000
    if max(H, W) > max_dim:
        scale = max_dim / max(H, W)
        new_H = int(H * scale)
        new_W = int(W * scale)
        print(f"Downsampling for visualization: {H}x{W} -> {new_H}x{new_W}")
        
        # Simple downsampling using mode (most common value in each block)
        from scipy.ndimage import zoom
        split_mask_vis = zoom(split_mask, scale, order=0)  # Nearest neighbor
    else:
        split_mask_vis = split_mask
    
    # Create color map
    # 0=white (invalid), 1=blue (train), 2=green (val), 3=orange (test), 4=red (calib)
    colors = ['white', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # Create figure
    with PdfPages(VISUALIZATION_FILE) as pdf:
        fig, ax = plt.subplots(figsize=(16, 12))
        
        im = ax.imshow(split_mask_vis, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Create legend
        labels = ['Invalid', 'Train (70%)', 'Validation (10%)', 'Test (10%)', 'Calibration (10%)']
        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(5)]
        ax.legend(handles=patches, loc='upper right', fontsize=12, framealpha=0.9)
        
        ax.set_title('Train/Val/Test/Calibration Splits\nGlobal Human Footprint Dataset', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Second page: Statistics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Split Statistics', fontsize=16, fontweight='bold')
        
        # Count chips per split
        train_count = (split_mask == 1).sum()
        val_count = (split_mask == 2).sum()
        test_count = (split_mask == 3).sum()
        calib_count = (split_mask == 4).sum()
        total_count = train_count + val_count + test_count + calib_count
        
        # Pie chart - Pixel distribution
        ax = axes[0, 0]
        sizes = [train_count, val_count, test_count, calib_count]
        labels_pie = ['Train', 'Val', 'Test', 'Calib']
        colors_pie = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax.set_title('Pixel Distribution', fontweight='bold')
        
        # Bar chart - Pixel counts
        ax = axes[0, 1]
        ax.bar(labels_pie, sizes, color=colors_pie)
        ax.set_ylabel('Number of Pixels', fontsize=11)
        ax.set_title('Pixel Counts', fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')
        for i, v in enumerate(sizes):
            ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
        
        # Text summary
        ax = axes[1, 0]
        ax.axis('off')
        summary_text = f"""
Dataset Summary
{'='*40}

Total Valid Pixels: {total_count:,}

Split Breakdown:
  • Train:       {train_count:,} ({100*train_count/total_count:.1f}%)
  • Validation:  {val_count:,} ({100*val_count/total_count:.1f}%)
  • Test:        {test_count:,} ({100*test_count/total_count:.1f}%)
  • Calibration: {calib_count:,} ({100*calib_count/total_count:.1f}%)

Configuration:
  • Chip Size: {CHIP_SIZE}x{CHIP_SIZE}
  • Min Valid Ratio: {MIN_VALID_RATIO*100:.0f}%
  • Random Seed: {RANDOM_SEED}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax.transAxes)
        
        # Spatial distribution histogram
        ax = axes[1, 1]
        # Show distribution of split types across latitude bands
        H = split_mask.shape[0]
        n_bands = 20
        band_size = H // n_bands
        
        band_stats = {label: [] for label in labels_pie}
        for band_idx in range(n_bands):
            start = band_idx * band_size
            end = min((band_idx + 1) * band_size, H)
            band = split_mask[start:end, :]
            
            for i, label in enumerate(labels_pie, start=1):
                count = (band == i).sum()
                band_stats[label].append(count)
        
        x = np.arange(n_bands)
        width = 0.2
        for i, (label, color) in enumerate(zip(labels_pie, colors_pie)):
            offset = (i - 1.5) * width
            ax.bar(x + offset, band_stats[label], width, label=label, color=color, alpha=0.8)
        
        ax.set_xlabel('Latitude Band (North to South)', fontsize=11)
        ax.set_ylabel('Pixel Count', fontsize=11)
        ax.set_title('Spatial Distribution by Latitude', fontweight='bold')
        ax.legend(fontsize=9)
        ax.ticklabel_format(style='plain', axis='y')
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Visualization saved: {VISUALIZATION_FILE}")

def main(argv=None):
    """Main execution."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds_only",
        action="store_true",
        help="Only build the k-fold mask (reuses the existing validity mask; leaves "
             "split_mask_1000.tif untouched)",
    )
    parser.add_argument(
        "--make_folds",
        action="store_true",
        help="Also build the k-fold mask after the regular splits",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of spatial folds (default: 5)")
    args = parser.parse_args(argv)

    if args.folds_only:
        if not os.path.exists(VALIDITY_MASK_FILE):
            print(f"✗ Validity mask not found: {VALIDITY_MASK_FILE}. Run without --folds_only first.")
            return 1
        with rasterio.open(VALIDITY_MASK_FILE) as src:
            valid_mask = src.read(1)
            profile = src.profile.copy()
            transform, crs = src.transform, src.crs
        create_kfold_splits(valid_mask, profile, transform, crs, k=args.k)
        print("\n✓ Fold mask complete")
        return 0

    print("\n" + "=" * 80)
    print("VALIDITY MASK AND SPLIT GENERATION")
    print("=" * 80)
    print(f"Target file: {TARGET_FILE}")
    print(f"Output validity mask: {VALIDITY_MASK_FILE}")
    print(f"Output split mask: {SPLIT_MASK_FILE}")
    print(f"Output visualization: {VISUALIZATION_FILE}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Check if target file exists
    if not os.path.exists(TARGET_FILE):
        print(f"\n✗ ERROR: Target file not found: {TARGET_FILE}")
        print("Please ensure the global HM dataset is in data/raw/hm_global/")
        return 1
    
    # Step 1: Create validity mask
    valid_mask, profile, transform, crs = create_validity_mask()
    
    # Step 2: Create spatial splits
    split_mask = create_spatial_splits(valid_mask, profile, transform, crs)

    # Step 2b: Optional k-fold mask for the hindcast harness
    if args.make_folds:
        create_kfold_splits(valid_mask, profile, transform, crs, k=args.k)

    # Step 3: Visualize
    visualize_splits(split_mask, transform, crs)
    
    print("\n" + "=" * 80)
    print("✓ ALL DONE!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  1. {VALIDITY_MASK_FILE}")
    print(f"  2. {SPLIT_MASK_FILE}")
    print(f"  3. {VISUALIZATION_FILE}")
    print("\nNext steps:")
    print("  - Review the visualization PDF")
    print("  - Update dataloader to use split_mask_1000.tif")
    print("  - Use split mask to filter training/validation data")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())