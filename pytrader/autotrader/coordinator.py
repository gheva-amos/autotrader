from autotrader.lib.working_thread import WorkingThread
import zmq
import time
import queue

class Coordinator(WorkingThread):
  def __init__(self, host):
    super().__init__("coordinator", host, host, zmq.DEALER)

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

  def request_scanner_params(self):
    frames = ["scanner_params".encode()]
    return self.send_frames(frames)

  def request_scanner(self, inst, loc, code, apply_filters={}):
    frames = []
    frames.append("scanner".encode())
    frames.append(inst.encode())
    frames.append(loc[0].encode())
    frames.append(code.encode())
    for filt in apply_filters:
      frames.append(filt.encode())
      frames.append(apply_filters[filt].encode())
    return self.send_frames(frames)

  def step(self):
    try:
      payload = self.inbox.get_nowait()
    except queue.Empty:
      return

if __name__ == "__main__":
  coordinator = Coordinator("tcp://localhost:7001")
  coordinator.start()
  time.sleep(1)
  coordinator.status()
  time.sleep(1)
  coordinator.stop_thread()
  pass
