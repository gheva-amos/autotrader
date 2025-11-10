from autotrader.lib.working_thread import WorkingThread

import zmq
import queue
import json
import pandas as pd
import io
import os
import tempfile

class Collector(WorkingThread):
  def __init__(self, host):
    super().__init__("collector", host, host, zmq.PULL, None, True)
    self.symbols = {}

  def atomic_write_file(self, path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
      tmp.write(data)
      tmp.flush()
      os.fsync(tmp.fileno())
      tmp_path = tmp.name
    os.replace(tmp_path, path)

  def step(self):
    try:
      frames = self.inbox.get_nowait()
    except queue.Empty:
      return

    if frames[1].decode() == 'indicators_pd':
      symbol = frames[2].decode()
      print(symbol)
      path = os.path.join('./data', symbol + '.parquet')
      data = io.BytesIO(frames[3])
      self.atomic_write_file(path, data.getvalue())
    elif frames[1].decode() == 'data_frames':
      symbol = frames[2].decode()
      if not symbol in self.symbols:
        self.symbols[symbol] = {}
      model_name = frames[3].decode()
      data = io.BytesIO(frames[4])
      self.symbols[symbol][model_name] = pd.read_parquet(data)
    print(self.symbols)

