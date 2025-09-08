from autotrader.driver import ATDriver
import time

def main(router_host, publisher_host):
  driver = ATDriver(router_host, publisher_host)
  
  driver.start()
  driver.get_coordinator().request_historical_data("AAPL")
  time.sleep(3)
  driver.stop_thread()
  print("here")

if __name__ == "__main__":
  main("tcp://localhost:7001", "tcp://localhost:7002")
