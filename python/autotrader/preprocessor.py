from autotrader.working_thread import WorkingThread
import zmq

class PreProcessor(WorkingThread):
  def __init__(self, host):
    super().__init__("preprocessor", host, zmq.SUB)
    self.socket.setsockopt(zmq.SUBSCRIBE, b"")
