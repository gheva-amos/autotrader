#include "ib/ib_client.h"
#include <thread>
#include <chrono>
#define DEBUG
#include "debug.h"

using autotrader::get_log_stream;

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    throw std::runtime_error("Expecting host and port arguments");
  }
  std::string host{argv[1]}; // host.docker.internal
  int port{std::atoi(argv[2])}; // 7497
  autotrader::IBClient client{host, port};

  Contract contract;
  contract.symbol = "TSLA";
  contract.secType = "STK";
  contract.currency = "USD";
  contract.exchange = "SMART";
  client.start();

  size_t index = client.start_market_data_stream(contract);

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  client.stop_market_data_stream(index);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  client.request_historical_data(contract);

  size_t id;
  if (client.next_historical_id(id))
  {
    auto bars = client.historical_bars(id);
    for (auto bar : bars)
    {
      DBG_MSG(__func__) << 
	" time " <<  bar.time <<
	" high " << bar.high <<
	" low " << bar.low <<
	" open " << bar.open <<
	" close " << bar.close <<
	" wap " << bar.wap <<
	" volume " << bar.volume <<
	" count " << bar.count << std::endl;
    }
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

