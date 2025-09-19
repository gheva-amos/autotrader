import collections

class EMA:
  def __init__(self, size):
    self.size = size
    self.factor = 2 / (size + 1)
    self.alpha = self.factor
    self.ema = collections.deque(maxlen=self.size)

  def add_value(self, value):
    self.ema.append(self.alpha * value)
    self.alpha *= 1 - self.factor

  def full(self):
    return len(self.ema) == self.size

  def avg(self):
    return sum(self.ema)/len(self.ema)

  def last(self):
    return self.ema[-1]

