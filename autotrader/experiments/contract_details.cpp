#include "ib/ib_client.h"
#include "Contract.h"
#include "OrderState.h"
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
  contract.symbol = "TSLA";
  contract.secType = "STK";
  contract.currency = "USD";
  contract.exchange = "SMART";
  contract.primaryExchange = "NASDAQ";
  client.start();
  size_t index = client.contract_details(contract);
  DBG_MSG("Index is") << index << std::endl;

  //contract.strike = 200;
  contract.conId = client.contract_id(index);
  DBG_MSG("Contract ID is:") << client.contract_id(index) << std::endl;

  index = client.search_symbol("tesla");
  auto symbols = client.symbols(index);
  for (auto s : symbols)
  {
    DBG_MSG("Symbol") << s << std::endl;
  }

  size_t how_many{9};
  client.request_order_ids(how_many);
  for (size_t i{0}; i < how_many; ++i)
  {
    DBG_MSG("OrderId:") << client.next_order_id() << std::endl;
  }

  /*
  auto id{client.sell(contract, 1)};
  auto state{client.get_order_state(id)};

  DBG_MSG("State") << "commissionAndFees " << state.commissionAndFees << std::endl;
  // */
  return 0;
}

