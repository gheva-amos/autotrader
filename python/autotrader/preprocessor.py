from autotrader.working_thread import WorkingThread
from autotrader.tools.scanner_params import ScannerParams
import zmq
import queue

class PreProcessor(WorkingThread):
  def __init__(self, host):
    super().__init__("preprocessor", host, zmq.SUB)
    self.socket.setsockopt(zmq.SUBSCRIBE, b"")

  def step(self):
    try:
      frames = self.inbox.get_nowait()
    except queue.Empty:
      return
    if frames[0].decode() == "scanner_params":
      self.handle_scanner_params(frames[1].decode())

  def handle_scanner_params(self, xml):
    scanner_params = ScannerParams(xml)

