from autotrader.working_thread import WorkingThread
from autotrader.tools.scanner_params import ScannerParams
import zmq
import queue

class PreProcessor(WorkingThread):
  def __init__(self, host):
    super().__init__("preprocessor", host, zmq.SUB)
    self.socket.setsockopt(zmq.SUBSCRIBE, b"")
    self.combos = []
    self.scanners = {}
    self.symbols = set()
    self.bars = {}

  def step(self):
    try:
      frames = self.inbox.get_nowait()
    except queue.Empty:
      return
    if frames[0].decode() == "scanner_params":
      self.handle_scanner_params(frames[1].decode())
    elif frames[0].decode() == "scanner":
      self.handle_scanner(frames)
    elif frames[0].decode() == 'history':
      if frames[1].decode() not in self.bars:
        self.bars[frames[1].decode()] = []
      self.bars[frames[1].decode()].append([bar_elem.decode() for bar_elem in frames[2:10]])

  def handle_scanner(self, frames):
    key = frames[1].decode()
    if key not in self.scanners:
      self.scanners[key] = []
    elem = {"symbol": frames[2].decode(), "conId": frames[3].decode()}
    self.scanners[key].append(elem)
    self.symbols.add((elem['symbol'], elem['conId']))

  def handle_scanner_params(self, xml):
    self.scanner_params = ScannerParams(xml)
    for st in self.scanner_params.scan_types:
      scan_type = {}
      scan_type['code'] = st['code']
      scan_type['name'] = st['name']
      filters = []
      locations = []
      scan_type['instruments'] = st['instruments']
      for inst in st['instruments']:
        if inst in self.scanner_params.instrument_map:
          filters.append(self.scanner_params.instrument_map[inst])
        if inst in self.scanner_params.instrument_locations:
          locations.append(self.scanner_params.instrument_locations[inst])
      scan_type['filter_fields'] = []
      for filt_list in filters:
        for filt in filt_list:
          if filt in self.scanner_params.filter_fields:
            scan_type['filter_fields'].append(self.scanner_params.filter_fields[filt])
        
#scan_type['filters'] = filters
      scan_type['locations'] = locations
      self.combos.append(scan_type)

