from autotrader.lib.base_model import BaseModel
import pandas as pd
import json
import threading
import queue
import io

class BarCollector(BaseModel):
  def __init__(self, listen, send):
    super().__init__("bar_collector", listen, send)
    self.data_frames = {}
    self.done_names = queue.Queue()
    self.processor_thread = threading.Thread(target=self.process_thread, daemon=True)
    self.processor_thread.start()

  def send_results_for(self, symbol):
    self.done_names.put(symbol)

  def add_indicators(self, df):
    try:
      cols = ["open","high","low","close","volume"]
      
      df[cols] = df[cols].apply(pd.to_numeric)
      if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    except:
      pass
    return df

  def process_thread(self):
    while True:
      symbol = self.done_names.get()
      if symbol is None:
        break
      df = self.data_frames[symbol]
      if not df.empty:
        try:
          df = self.add_indicators(df)
        except:
          print(f"exception {symbol}")
          continue
        buf = io.BytesIO()
        df.to_parquet(buf, index=True)
        self.send_frames(['data_frames'.encode(), symbol.encode(), "bars_collector".encode(), buf.getvalue()])
      
  def handle_frames(self, frames):
    if frames[1].decode() == 'historical_bar_done':
      self.send_results_for(frames[2].decode())
      return
    if not frames[1].decode() == 'historical_bar':
      return
    bar = json.loads(frames[2].decode())
    symbol = bar['symbol']
    if symbol not in self.data_frames:
      self.data_frames[symbol] = pd.DataFrame(columns=["open","high","low","close","volume"])
      self.data_frames[symbol].index.name = "date"
    row = {}
    row['date'] = pd.to_datetime(bar['date'])
    for entry in ["open","high","low","close","volume"]:
      row[entry] = bar[entry]
    self.data_frames[symbol] = pd.concat([self.data_frames[symbol], pd.DataFrame([row])])


def main(listen, send, args=None):
  ind = BarCollector(listen, send)
  ind.start()
  print('loaded')
