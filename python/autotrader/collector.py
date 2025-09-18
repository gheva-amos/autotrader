from autotrader.lib.working_thread import WorkingThread

import zmq
import queue

class Collector(WorkingThread):
  def __init__(self, host):
    super().__init__("collector", host, host, zmq.PULL, None, True)

  def step(self):
    try:
      frames = self.inbox.get_nowait()
    except queue.Empty:
      return
    print(frames)

