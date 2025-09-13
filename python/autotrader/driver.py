from autotrader.coordinator import Coordinator
from autotrader.preprocessor import PreProcessor
import threading
import time

class ATDriver:
  def __init__(self, router, publisher, instrument):
    self.coordinator = Coordinator(router)
    self.preprocessor = PreProcessor(publisher)
    self.thread = None
    self.stop = threading.Event()
    self.scanners_requested = False
    self.instrument = instrument

  def start(self):
    self.coordinator.start()
    self.preprocessor.start()
    self.coordinator.request_scanner_params()
    if self.thread is None:
      self.thread = threading.Thread(target=self.run, name="autotrader", daemon=True)
      self.thread.start()

  def stop_thread(self, timeout=1.0):
    self.stop.set()
    if self.thread is not None:
      self.thread.join(timeout)
      self.thread = None
    self.preprocessor.stop_thread()
    self.coordinator.stop_thread()

  def get_coordinator(self):
    return self.coordinator

  @staticmethod
  def unwrap(item):
    while isinstance(item, list) and len(item) == 1:
      item = item[0]
    return item

  def request_scanners(self, instrument):
    scanners = [(elem['code'], elem['locations'], elem['instruments']) for elem in self.preprocessor.combos if instrument in elem['instruments']]
    for scanner in scanners[0:3]:
      code = scanner[0]
      instr = instrument
      loc = self.unwrap(scanner[1][0])
      self.coordinator.request_scanner(instr, loc, code)

  def run(self):
    while not self.stop.is_set():
      if self.preprocessor.combos and not self.scanners_requested:
        self.request_scanners(self.instrument)
        self.scanners_requested = True
      time.sleep(1)
