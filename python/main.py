from autotrader.driver import ATDriver
import time

def main(router_host, publisher_host, dist, instrument):
  driver = ATDriver(router_host, publisher_host, dist, instrument)
  
  driver.start()

  driver.stop_thread()

if __name__ == "__main__":
  main("tcp://localhost:7001", "tcp://localhost:7002", "tcp://localhost:7007", 'STK')
