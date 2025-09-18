from autotrader.lib.working_thread import WorkingThread
import zmq
import queue

class TestModel(WorkingThread):
  def __init__(self):
    super().__init__("test_module", 'tcp://localhost:7007', 'tcp://localhost:7003', zmq.SUB, None, False)
    self.socket.setsockopt(zmq.SUBSCRIBE, b"")

  def set_send_socket(self):
    socket = self.ctx.socket(zmq.PUSH)
    return self.socket

  def step(self):
    try:
      frames = self.inbox.get_nowait()
    except queue.Empty:
      return
    print(frames)

def main():
  tm = TestModel()
  tm.start()
  print('loaded')
