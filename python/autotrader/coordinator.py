from autotrader.working_thread import WorkingThread
import zmq
import time
import queue

class Coordinator(WorkingThread):
  def __init__(self, host):
    super().__init__("coordinator", host, zmq.DEALER)

  def status(self):
    frames = ["status".encode()]
    return self.send_frames(frames)

  def stream_symbol(self, symbol):
    frames = ["data".encode(), symbol.encode()]
    return self.send_frames(frames)

  def stop_data_stream(self, req_id):
    frames = ["stop_data".encode(), str(req_id).encode()]
    return self.send_frames(frames)

  def request_historical_data(self, symbol):
    frames = ["history".encode(), symbol.encode()]
    return self.send_frames(frames)

  def step(self):
    try:
      payload = self.inbox.get_nowait()
    except queue.Empty:
      return

    print(payload)

if __name__ == "__main__":
  coordinator = Coordinator("tcp://localhost:7001")
  coordinator.start()
  time.sleep(1)
  coordinator.status()
  time.sleep(1)
  coordinator.stop_thread()
  pass
