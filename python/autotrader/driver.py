from autotrader.coordinator import Coordinator
from autotrader.preprocessor import PreProcessor
from autotrader.distributor import Distributor
from ui.at_gui import ATGui
import threading
import time
import sys
import json

class ATDriver:
  def __init__(self, router, publisher, dist, instrument):
    self.coordinator = Coordinator(router)
    self.preprocessor = PreProcessor(publisher)
    self.distributor = Distributor(dist)
    self.ui = ATGui(self)
    self.thread = None
    self.stop = threading.Event()
    self.scanners_requested = False
    self.scanners_selected = False
    self.scanner_list = []
    self.instrument = instrument
    self.filter_list = {}

  def start(self):
    self.coordinator.start()
    self.preprocessor.start()
    self.distributor.start()
    if self.thread is None:
      self.thread = threading.Thread(target=self.run, name="autotrader", daemon=True)
      self.thread.start()
    self.ui.mainloop()

  def request_scanner_params(self):
    self.coordinator.request_scanner_params()

  def stop_thread(self, timeout=1.0):
    self.stop.set()
    if self.thread is not None:
      self.thread.join(timeout)
      self.thread = None
    self.distributor.stop_thread()
    self.preprocessor.stop_thread()
    self.coordinator.stop_thread()

  def get_coordinator(self):
    return self.coordinator

  @staticmethod
  def unwrap(item):
    while isinstance(item, list) and len(item) == 1:
      item = item[0]
    return item

  def request_scanner(self, scanner, apply_filters={}):
    code = scanner[0]
    instr = self.instrument
    loc = self.unwrap(scanner[1][0])
    self.coordinator.request_scanner(instr, loc, code, apply_filters)

  def request_scanners(self):
    for scanner in self.scanner_list:
      self.request_scanner(scanner)

  def select_scanners(self):
    scanners = [(elem['code'], elem['locations'], elem['instruments'], elem['name']) for elem in self.preprocessor.combos if self.instrument in elem['instruments']]
    self.ui.add_scanner_list(scanners)

  def get_filter_list(self):
    filters = self.preprocessor.scanner_params.instrument_map[self.instrument]
    filter_fields = self.preprocessor.scanner_params.filter_fields
    for filt in filter_fields:
      if filt not in filters:
        continue
      self.filter_list[filt] = filter_fields[filt]
    return self.filter_list

  def instrument_list(self):
    return [inst[0] for inst in self.preprocessor.symbols]

  def process_instruments(self, instruments):
    for inst in instruments:
      self.coordinator.request_historical_data(inst)

  def handle_bars(self):
    while self.preprocessor.bars:
      key, bars = self.preprocessor.bars.popitem()
      for bar in bars:
        msg = {
          'symbol': key,
          'date': bar[0],
          'high': bar[1],
          'low': bar[2],
          'open': bar[3],
          'close': bar[4],
          'wap': bar[5],
          'volume': bar[6],
          'count': bar[7],
        }
        self.distributor.send_frames(['historical_bar'.encode(), json.dumps(msg).encode()])


  def run(self):
    while not self.stop.is_set():
      if self.preprocessor.combos and not self.scanners_selected:
        self.select_scanners()
        self.scanners_selected = True
      if self.preprocessor.bars:
        self.handle_bars()
      elif self.scanner_list and not self.scanners_requested:
        self.request_scanners()
        self.scanners_requested = True
      time.sleep(1)
