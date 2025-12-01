import zmq
import os
import time
import threading
import queue

class WorkingThread:
  def __init__(self, name, host, send_host,  zmq_type, ctx=None, bind=False):
    self.ctx = ctx or zmq.Context.instance()
    self.socket = self.ctx.socket(zmq_type)
    self.send_socket = self.set_send_socket()
    self.host = host
    self.send_host = send_host
    self.poller = zmq.Poller()
    self.name = name
    self.thread = None
    self.stop = threading.Event()
    self.work = queue.Queue()
    self.inbox = queue.Queue()
    self.bind = bind

  def set_send_socket(self):
    return None

  def start(self):
    if self.thread is None:
      self.thread = threading.Thread(target=self.run, name=self.name, daemon=True)
      self.thread.start()

  def stop_thread(self, timeout=1.0):
    self.stop.set()
    if self.thread is not None:
      self.thread.join(timeout)
      self.thread = None

  def connect(self):
    if self.bind:
      self.socket.bind(self.host)
      if not self.send_socket:
        self.send_socket = self.socket
      else:
        self.send_socket.bind(self.send_host)
    else:
      self.socket.connect(self.host)
      if not self.send_socket:
        self.send_socket = self.socket
      else:
        self.send_socket.connect(self.send_host)
    self.poller.register(self.socket, zmq.POLLIN)
    self.socket.setsockopt(zmq.RCVTIMEO, 1000)

  def disconnect(self):
    self.socket.setsockopt(zmq.LINGER, 0)
    self.socket.close()

  def step(self):
    pass

  def setup(self):
    pass

  def teardown(self):
    pass

  def run(self):
    self.connect()
    self.setup()
    while not self.stop.is_set():
      # get data sent to us:
      events = dict(self.poller.poll(10))
      if self.socket in events:
        frames = self.socket.recv_multipart()
        self.inbox.put(frames)

      # Send data out
      try:
        payload = self.work.get_nowait()
      except queue.Empty:
        pass
      else:
        self.send_socket.send_multipart(payload)
      
      self.step()

    self.teardown()
    self.disconnect()

  def send_frames(self, frames):
    req_id = os.urandom(8).hex().encode()
    frames.insert(0, req_id)
    self.work.put(frames)

if __name__ == "__main__":
  pass
