import zmq
import os
import time

class AutoTrader:
  def __init__(self, host):
    self.ctx = zmq.Context.instance()
    self.socket = self.ctx.socket(zmq.DEALER)
    self.socket.connect(host)
    self.poller = zmq.Poller()
    self.poller.register(self.socket, zmq.POLLIN)
    self.socket.setsockopt(zmq.RCVTIMEO, 1000)

  def send_frames(self, frames):
    req_id = os.urandom(8).hex().encode()
    frames.insert(0, req_id)
    self.socket.send_multipart(frames)
    events = dict(self.poller.poll(1000))
    if self.socket not in events:
      return None
    return self.socket.recv_multipart()

  def status(self):
    frames = ["status".encode()]
    return self.send_frames(frames)

  def stream_symbol(self, symbol):
    frames = ["data".encode(), symbol.encode()]
    return self.send_frames(frames)

  def stop_data_stream(self, req_id):
    frames = ["stop_data".encode(), str(req_id).encode()]
    return self.send_frames(frames)


if __name__ == "__main__":
  at = AutoTrader("tcp://localhost:7001")
  tmp = at.stream_symbol("AAPL")[0].decode()
  print(tmp)
  
  time.sleep(1)
  print(at.stop_data_stream(tmp))
