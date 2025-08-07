import matplotlib.pyplot as plt
import torch
from torchgeo_dataloader import get_dataloader

# Get a batch
loader = get_dataloader(batch_size=1, chip_size=128, timesteps=3, chips_per_epoch=1)
batch = next(iter(loader))

input_dynamic = batch['input_dynamic'][0]  # [3, 128, 128]
input_static = batch['input_static'][0, 0]  # [128, 128] (first static var)
target = batch['target'][0]  # [128, 128]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

# Static chip
im0 = axes[0].imshow(input_static.cpu(), cmap='terrain')
axes[0].set_title('Static (elevation)')
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Human footprint input chips
for t in range(input_dynamic.shape[0]):
    im = axes[t+1].imshow(input_dynamic[t].cpu(), cmap='viridis', vmin=0, vmax=1)
    axes[t+1].set_title(f'Input HM t-{input_dynamic.shape[0]-t}')
    axes[t+1].axis('off')
    plt.colorbar(im, ax=axes[t+1], fraction=0.046, pad=0.04)

# Human footprint target chip
im4 = axes[4].imshow(target.cpu(), cmap='magma', vmin=0, vmax=1)
axes[4].set_title('Target HM (next)')
axes[4].axis('off')
plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
