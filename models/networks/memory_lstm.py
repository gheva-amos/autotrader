import torch.nn as nn

class MemoryLSTM(nn.Module):
  def __init__(self, F, H=64):
    super().__init__()
    self.lstm = nn.LSTM(F, H, batch_first=True)

  def forward(self, x):
    h, _ = self.lstm(x)
    return h

