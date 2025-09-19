import collections

class MovingAverage:
  def __init__(self, window):
    self.window = window
    self.stock_vals = collections.deque(maxlen=self.window)
    self.averages = {}

  def process_bar(self, bar):
    close = float(bar['close'])
    self.stock_vals.append(close)
    date = bar['date']
    ret = False

    if len(self.stock_vals) == self.window:
      ret = True
      self.averages[date] = sum(self.stock_vals)/len(self.stock_vals)

    return ret
