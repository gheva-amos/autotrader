from autotrader.lib.working_thread import WorkingThread
import zmq
import queue
import json
import traceback

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
    try:
      self.handle_frames(frames)
    except Exception as e:
      print("Standard error:", e)
      traceback.print_exc()
    except BaseException as e:
      print("Non-standard error:", e)
      traceback.print_exc()
    except:
      print('Caught an exception, continuing')
      traceback.print_exc()

  def handle_frames(self, frames):
    pass
