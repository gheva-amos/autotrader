import torch.nn as nn

class RegHead(nn.Module):
  """
    We start real simple and add layers to this network later
  """
  def __init__(self, features):
    super().__init__()
    self.net = nn.Linear(features, 1)

  def forward(self, z):
    return self.net(z).squeeze(1)


