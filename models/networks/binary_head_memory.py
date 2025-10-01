import torch.nn as nn
from models.networks.memory_lstm import MemoryLSTM
from models.networks.binary_head import BinaryHead

class BinaryHeadMemory(nn.Module):
  def __init__(self, F, H=(64, 32, 16, 8), drop=0.2):
    super().__init__()
    self.memory = MemoryLSTM(F, F)
    self.head = BinaryHead(F, H)

  def forward(self, x):
    x = self.memory(x)
    return self.head(x)
