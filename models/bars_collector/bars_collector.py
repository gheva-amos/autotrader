from autotrader.lib.base_model import BaseModel
import pandas as pd
import json
import threading
import queue
import ta
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
    except:
      pass
    df["rsi14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    atr = ta.volatility.AverageTrueRange(
      high=df["high"], low=df["low"], close=df["close"], window=14
      )
    df["atr14"] = atr.average_true_range()

    df["stoch_k"] = ta.momentum.StochasticOscillator(
      df["high"], df["low"], df["close"], window=14, smooth_window=3
    ).stoch()

    df["stoch_d"] = ta.momentum.StochasticOscillator(
      df["high"], df["low"], df["close"], window=14, smooth_window=3
    ).stoch_signal()

    df["sma20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

    boll = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_mavg"] = boll.bollinger_mavg()
    df["bb_high"] = boll.bollinger_hband()
    df["bb_low"]  = boll.bollinger_lband()

    df.dropna(inplace=True)
    return df

  def process_thread(self):
    while True:
      symbol = self.done_names.get()
      if symbol is None:
        break
      df = self.data_frames[symbol]
      if not df.empty:
        df = self.add_indicators(df)
        buf = io.BytesIO()
        df.to_parquet(buf, index=True)
        self.send_frames(['indicators_pd'.encode(), symbol.encode(), buf.getvalue()])
      
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

      


def main(listen, send):
  ind = BarCollector(listen, send)
  ind.start()
  print('loaded')
