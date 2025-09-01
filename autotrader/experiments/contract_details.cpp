#include "ib/ib_client.h"
#include "Contract.h"
#include <future>
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
  contract.symbol = "INTC";
  contract.secType = "STK";
  contract.currency = "USD";
  contract.exchange = "SMART";
  client.start();
  size_t index = client.contract_details(contract);
  DBG_MSG("Index is") << index << std::endl;

  DBG_MSG("Contract ID is:") << client.contract_id(index) << std::endl;

  index = client.search_symbol("tesla");
  auto symbols = client.symbols(index);
  for (auto s : symbols)
  {
    DBG_MSG("Symbol") << s << std::endl;
  }
  return 0;
}

