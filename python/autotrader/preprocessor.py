from autotrader.working_thread import WorkingThread
from autotrader.tools.scanner_params import ScannerParams
import zmq
import queue

class PreProcessor(WorkingThread):
  def __init__(self, host):
    super().__init__("preprocessor", host, zmq.SUB)
    self.socket.setsockopt(zmq.SUBSCRIBE, b"")
    self.combos = []

  def step(self):
    try:
      frames = self.inbox.get_nowait()
    except queue.Empty:
      return
    if frames[0].decode() == "scanner_params":
      self.handle_scanner_params(frames[1].decode())

  def handle_scanner_params(self, xml):
    self.scanner_params = ScannerParams(xml)
    for st in self.scanner_params.scan_types:
      scan_type = {}
      scan_type['code'] = st['code']
      filters = []
      locations = []
      scan_type['instruments'] = st['instruments']
      for inst in st['instruments']:
        if inst in self.scanner_params.instrument_map:
          filters.append(self.scanner_params.instrument_map[inst])
        if inst in self.scanner_params.location_instruments:
          locations.append(self.scanner_params.location_instruments[inst])
      scan_type['filter_fields'] = []
      for filt_list in filters:
        for filt in filt_list:
          if filt in self.scanner_params.filter_fields:
            scan_type['filter_fields'].append(self.scanner_params.filter_fields[filt])
        
#scan_type['filters'] = filters
      scan_type['locations'] = locations
      self.combos.append(scan_type)

