from autotrader.lib.working_thread import WorkingThread
from moving_average import MovingAverage
import zmq
import queue
import json

class BaseModel(WorkingThread):
  def __init__(self, name, listen, send):
    super().__init__(name, listen, send, zmq.SUB, None, False)
    self.socket.setsockopt(zmq.SUBSCRIBE, b"")

  def set_send_socket(self):
    return self.ctx.socket(zmq.PUSH)

  def step(self):
    try:
      frames = self.inbox.get_nowait()
    except queue.Empty:
      return
    self.handle_frames(frames)

  def handle_frames(self, frames):
    pass
