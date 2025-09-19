
class OBV:
  def __init__(self):
    self.last_close = None
    self.obv = {}
    self.last_obv = None

  def process_bar(self, bar):
    close = float(bar['close'])
    if not self.last_close:
      self.last_close = close
      return False

    update = 0.0
    if close > self.last_close:
      update = float(bar['volume'])
    elif close < self.last_close:
      update = float(bar['volume']) * -1

    date = bar['date']
    self.last_close = close
    if not self.obv:
      self.last_obv = self.obv[date] = update
    else:
      last_obv = self.last_obv
      self.last_obv = self.obv[date] = last_obv + update

    return True
