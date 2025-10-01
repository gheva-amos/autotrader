import torch.nn as nn

class MemoryTCN(nn.Module):
  def __init__(self, F, H=64, drop=0.2):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv1d(F, H, 3, padding=1), nn.GELU(),
      nn.Conv1d(H, H, 3, padding=1), nn.GELU(),
      nn.Dropout(drop),
    )

  def forward(self, x):
    z = self.net(x.transpose(1,2))
    return z.transpose(1,2)

