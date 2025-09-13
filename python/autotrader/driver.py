from autotrader.coordinator import Coordinator
from autotrader.preprocessor import PreProcessor
import threading
import time
import sys

class ATDriver:
  def __init__(self, router, publisher, instrument):
    self.coordinator = Coordinator(router)
    self.preprocessor = PreProcessor(publisher)
    self.thread = None
    self.stop = threading.Event()
    self.scanners_requested = False
    self.scanners_selected = False
    self.scanner_list = []
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

  def request_scanners(self):
    for scanner in self.scanner_list:
      code = scanner[0]
      instr = self.instrument
      loc = self.unwrap(scanner[1][0])
      self.coordinator.request_scanner(instr, loc, code)

  def selct_scanners(self):
    scanners = [(elem['code'], elem['locations'], elem['instruments'], elem['name']) for elem in self.preprocessor.combos if self.instrument in elem['instruments']]
    i = 0
    line = ""
    for scanner in scanners:
      line += f"{i} - {scanner[3]}\t"
      i += 1
      if i % 4 == 0:
        print(line)
        line = ""
    print('Select up to 10 filters (one per line), hit Ctrl+d when done')
    selection = sys.stdin.read()  
    selection = selection.split('\n')
    selection = [int(s) for s in selection if s][:10]
    print (selection)
    self.scanner_list = [scanners[i] for i in selection]

  def run(self):
    while not self.stop.is_set():
      if self.preprocessor.combos and not self.scanners_selected:
        self.selct_scanners()
        self.scanners_selected = True
      elif self.scanner_list and not self.scanners_requested:
        self.request_scanners()
        self.scanners_requested = True
      time.sleep(1)
