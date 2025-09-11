from autotrader.coordinator import Coordinator
from autotrader.preprocessor import PreProcessor
import threading
import time

class ATDriver:
  def __init__(self, router, publisher):
    self.coordinator = Coordinator(router)
    self.preprocessor = PreProcessor(publisher)
    self.thread = None
    self.stop = threading.Event()

  def start(self):
    self.coordinator.start()
    self.preprocessor.start()
    if self.thread is None:
      self.thread = threading.Thread(target=self.run, name="autotrader", daemon=True)
      self.thread.start()

  def stop_thread(self, timeout=1.0):
    self.stop.set()
    if self.thread is not None:
      self.thread.join(timeout)
      self.thread = None
    self.preprocessor.stop_thread()
    self.coordinator.stop_thread()

  def get_coordinator(self):
    return self.coordinator

  def run(self):
    while not self.stop.is_set():
      if self.preprocessor.combos:
        print (self.preprocessor.combos[3])
      time.sleep(1)
