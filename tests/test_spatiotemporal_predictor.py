import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor

def test_spatiotemporal_predictor_forward():
    # Simulate dataloader output
    batch_size = 2
    timesteps = 3
    H = W = 128
    input_dynamic = torch.randn(batch_size, timesteps, 1, H, W)
    input_static = torch.randn(batch_size, 1, H, W)
    model = SpatioTemporalPredictor(hidden_dim=8, kernel_size=3, num_layers=1)
    pred = model(input_dynamic, input_static)
    # Output should be [B, 1, H, W]
    assert pred.shape == (batch_size, 1, H, W)
