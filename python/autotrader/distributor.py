from autotrader.lib.working_thread import WorkingThread

import zmq
import queue

class Distributor(WorkingThread):
  def __init__(self, host):
    super().__init__("distributor", host, host, zmq.PUB, None, True)
