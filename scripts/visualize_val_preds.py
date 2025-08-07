import torch
import matplotlib.pyplot as plt
from torchgeo_dataloader import get_dataloader
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.lightning_module import SpatioTemporalLightningModule

# Path to best checkpoint
ckpt_path = os.path.join('models', 'checkpoints', 'lightning_logs', 'version_7', 'checkpoints', 'epoch=9-step=250.ckpt')

# Load model from checkpoint
model = SpatioTemporalLightningModule.load_from_checkpoint(ckpt_path, map_location='cpu')
model.eval()

# Get a batch from the validation loader
loader = get_dataloader(batch_size=4, chip_size=128, timesteps=3, chips_per_epoch=4, mode="grid")
batch = next(iter(loader))

input_dynamic = batch['input_dynamic']
input_static = batch['input_static']
target = batch['target']

with torch.no_grad():
    # Ensure input shapes are correct
    if input_dynamic.dim() == 4:
        input_dynamic = input_dynamic.unsqueeze(2)
    preds = model(input_dynamic, input_static).squeeze(1)  # [B, H, W]

B = input_dynamic.shape[0]
fig, axes = plt.subplots(B, 3, figsize=(12, 4*B))
for b in range(B):
    # Target
    im0 = axes[b, 0].imshow(target[b].cpu(), cmap='magma', vmin=0, vmax=1)
    axes[b, 0].set_title('Target')
    axes[b, 0].axis('off')
    plt.colorbar(im0, ax=axes[b, 0], fraction=0.046, pad=0.04)
    # Prediction
    im1 = axes[b, 1].imshow(preds[b].cpu(), cmap='viridis', vmin=0, vmax=1)
    axes[b, 1].set_title('Prediction')
    axes[b, 1].axis('off')
    plt.colorbar(im1, ax=axes[b, 1], fraction=0.046, pad=0.04)
    # Error
    error = (preds[b] - target[b]).abs()
    im2 = axes[b, 2].imshow(error.cpu(), cmap='hot', vmin=0, vmax=1)
    axes[b, 2].set_title('Absolute Error')
    axes[b, 2].axis('off')
    plt.colorbar(im2, ax=axes[b, 2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
