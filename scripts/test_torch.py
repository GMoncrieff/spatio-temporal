import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Create a dummy dataset: y = 2x + 1 + noise
N = 100
X = torch.randn(N, 1)
y = 2 * X + 1 + 0.1 * torch.randn(N, 1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define a simple linear regression model
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleLinearModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Training loop
for epoch in range(20):
    epoch_loss = 0.0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1:2d}, Loss: {epoch_loss/N:.4f}")

# Test prediction
with torch.no_grad():
    x_test = torch.tensor([[2.0]])
    y_pred = model(x_test)
    print(f"Prediction for x=2.0: {y_pred.item():.4f}")
