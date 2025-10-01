import torch.nn as nn

class BinaryHead(nn.Module):
  def __init__(self, in_dim, hidden=(128, 64), drop=0.2):
    super().__init__()

    layers = []
    d = in_dim
    for h in hidden:
      layers += [nn.LayerNorm(d)]
      layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
      d = h
    layers.append(nn.Linear(d, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
     return self.net(x)

