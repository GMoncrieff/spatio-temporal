import matplotlib.pyplot as plt
import torch
from torchgeo_dataloader import get_dataloader
import numpy as np

# Get a batch from the validation dataloader (grid mode)
loader = get_dataloader(batch_size=4, chip_size=128, timesteps=3, chips_per_epoch=16, mode="grid")
batch = next(iter(loader))

input_dynamic = batch['input_dynamic']  # [B, 3, 128, 128]
input_static = batch['input_static']    # [B, 1, 128, 128]
target = batch['target']                # [B, 128, 128]

B = input_dynamic.shape[0]
fig, axes = plt.subplots(B, 5, figsize=(18, 4*B))

for b in range(B):
    # Static chip
    im0 = axes[b, 0].imshow(input_static[b, 0].cpu(), cmap='terrain')
    axes[b, 0].set_title('Static (elevation)')
    axes[b, 0].axis('off')
    plt.colorbar(im0, ax=axes[b, 0], fraction=0.046, pad=0.04)
    # Human footprint input chips
    for t in range(input_dynamic.shape[1]):
        im = axes[b, t+1].imshow(input_dynamic[b, t].cpu(), cmap='viridis', vmin=0, vmax=1)
        axes[b, t+1].set_title(f'Input HM t-{input_dynamic.shape[1]-t}')
        axes[b, t+1].axis('off')
        plt.colorbar(im, ax=axes[b, t+1], fraction=0.046, pad=0.04)
    # Target
    im4 = axes[b, 4].imshow(target[b].cpu(), cmap='magma', vmin=0, vmax=1)
    axes[b, 4].set_title('Target HM (next)')
    axes[b, 4].axis('off')
    plt.colorbar(im4, ax=axes[b, 4], fraction=0.046, pad=0.04)
    # Print valid pixel count
    valid = torch.isfinite(target[b]).sum().item()
    axes[b, 4].set_xlabel(f'Valid target pixels: {valid}')

plt.tight_layout()
plt.show()
