import torch.nn as nn

class SimpleSequential(nn.Module):
  def __init__(self, in_features, out_features, drop = 0.2):
    super().__init__()
    self.net = nn.Sequential(nn.LayerNorm(in_features), nn.Linear(in_features, out_features), nn.GELU(), nn.Dropout(drop))

  def forward(self, x):
    return self.net(x)

