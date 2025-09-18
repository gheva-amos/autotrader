from autotrader.lib.working_thread import WorkingThread

import zmq
import queue

class Collector(WorkingThread):
  def __init__(self, host):
    super().__init__("collector", host, host, zmq.PULL, None, True)

