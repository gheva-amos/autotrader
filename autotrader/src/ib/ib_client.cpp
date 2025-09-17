#include "ib/ib_client.h"
#include <utility>
#include <thread>
#include <chrono>
#include <iostream>
#include "ScannerSubscription.h"
#define DEBUG
#include "debug.h"

namespace autotrader
{

int IBClient::client_id{0};

IBClient::IBClient(const std::string host, int port) :
  host_{std::move(host)}, port_{port}, client_id_{client_id},
  connected_{false}, started_{false},
  signal_{std::make_unique<EReaderOSSignal>(2000)},
  client_{std::make_unique<EClientSocket>(this, signal_.get())},
  running_{true}
{
  if (!client_->eConnect(host_.c_str(), port_, client_id_))
  {
    throw std::runtime_error("Could not connect to " + host_);
  }
  client_id += 1;
  connected_ = true;
  reader_ = std::make_unique<EReader>(client_.get(), signal_.get());
}

IBClient::~IBClient()
{
  if (is_connected())
  {
    client_->eDisconnect();
  }
}

bool IBClient::is_connected() const
{
  return client_->isConnected();
}

EClient* IBClient::client()
{
  return client_.get();
}

void IBClient::step()
{
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  wait_for_signal();
  try
  {
    process_messages();
  } catch (const std::exception& e)
  {
    DBG_MSG("Caught exception") << e.what() << std::endl;
  }
}

void IBClient::start()
{
  if (started_)
  {
    return;
  }
  started_ = true;
  reader_->start();
  step();
}

void IBClient::stop()
{
  running_ = false;
}

void IBClient::operator()()
{
  while (running_)
  {
    step();
  }
}

void IBClient::wait_for_signal()
{
  signal_->waitForSignal();
}

void IBClient::process_messages()
{
  reader_->processMsgs();
}

void IBClient::error(int id, time_t errorTime, int errorCode,const std::string& errorString, const std::string& advancedOrderRejectJson)
{
  std::cerr << "Handling error" << id << " " << errorCode << " " << errorString << std::endl << advancedOrderRejectJson << std::endl;
}

size_t IBClient::contract_details(Contract& contract)
{
  size_t ret{contracts_.push_back(contract)};
  client()->reqContractDetails(ret, contract);
  return ret;
}

void IBClient::contractDetails(int reqId, const ContractDetails& contractDetails)
{
  DBG_MSG("contract details") << contractDetails.contract.multiplier << std::endl;
  contracts_[reqId] = contractDetails.contract;
}

void IBClient::contractDetailsEnd(int reqId)
{
  // TODO if this is not called for a reqId, the contracts_[reqId] might not be valid
}

long IBClient::contract_id(size_t index)
{
  return contracts_[index].conId;
}

size_t IBClient::search_symbol(const std::string& symbol)
{
  size_t ret{symbols_.push_back(std::vector<std::string>())};

  client()->reqMatchingSymbols(ret, symbol);
  return ret;
}

void IBClient::symbolSamples(int reqId, const std::vector<ContractDescription> &contractDescriptions)
{
  auto& v = symbols_[reqId];
  for (auto c : contractDescriptions)
  {
    v.push_back(c.contract.symbol);
  }
}

std::vector<std::string>& IBClient::symbols(size_t index)
{
  return symbols_[index];
}

void IBClient::request_order_ids(size_t how_many)
{
  client()->reqIds(how_many);
}

void IBClient::nextValidId(OrderId orderId)
{
  order_ids_.push_back(orderId);
}

OrderId IBClient::next_order_id()
{
  return order_ids_.pop();
}

OrderId IBClient::place_order(Order order, Contract contract)
{
  request_order_ids(1);
  OrderId id = next_order_id();
  client_->placeOrder(id, contract, order);
  return id;
}

OrderId IBClient::buy(Contract contract, size_t how_many)
{
  Order order;
  order.action = "BUY";
  order.totalQuantity = how_many;
  order.orderType = "MKT";
  order.transmit = true;
  return place_order(order, contract);
}

OrderId IBClient::sell(Contract contract, size_t how_many)
{
  Order order;
  order.action = "SELL";
  order.totalQuantity = how_many;
  order.orderType = "MKT";
  order.transmit = true;
  return place_order(order, contract);
}

void IBClient::openOrder(OrderId orderId, const Contract&, const Order&, const OrderState& state)
{
  DBG_MSG("FEES") << state.commissionAndFees << std::endl;
  order_states_.insert(std::make_pair(orderId, state));
}

OrderState IBClient::get_order_state(OrderId id)
{
  return order_states_[id];
}

void IBClient::orderStatus(OrderId orderId, const std::string& status, Decimal filled,
      Decimal remaining, double avgFillPrice, long long permId, int parentId,
      double lastFillPrice, int clientId, const std::string& whyHeld, double mktCapPrice)
{
  // TODO do we need any of this?
  DBG_MSG("Order status comming in") << std::endl;
}

size_t IBClient::start_market_data_stream(Contract con)
{
  size_t ret{mkt_data_.push_back(MarketData())};
  client_->reqMktData(ret, con, "", false, false, TagValueListSPtr());
  return ret;
}

size_t IBClient::start_market_data_stream(std::string symbol)
{
  Contract contract;
  contract.symbol = std::move(symbol);
  contract.secType = "STK";
  contract.currency = "USD";
  contract.exchange = "SMART";
  return start_market_data_stream(contract);
}

void IBClient::stop_market_data_stream(size_t index)
{
  client_->cancelMktData(index);
}

void IBClient::tickPrice(TickerId tickerId, TickType field, double price, const TickAttrib& attrib)
{
  DBG_MSG(__func__) << "field " << field << " price " << price << std::endl;
  mkt_data_[tickerId].price(field) = price;
}

void IBClient::tickSize(TickerId tickerId, TickType field, Decimal size)
{
  DBG_MSG(__func__) << "field " << field << " size " << size << std::endl;
  mkt_data_[tickerId].size(field) = size;
}

void IBClient::tickGeneric(TickerId tickerId, TickType tickType, double value)
{
  DBG_MSG(__func__) << "field " << tickType << " value " << value << std::endl;
  mkt_data_[tickerId].generic(tickType) = value;
}

void IBClient::tickString(TickerId tickerId, TickType tickType, const std::string& value)
{
  DBG_MSG(__func__) << "field " << tickType << " value " << value << std::endl;
  mkt_data_[tickerId].value(tickType) = value;
}

size_t IBClient::request_historical_data(std::string symbol, std::string end, std::string duration,
    std::string bar_size)
{
  Contract contract;
  contract.symbol = symbol;
  contract.secType = "STK";
  contract.currency = "USD";
  contract.exchange = "SMART";
  return request_historical_data(contract, end, duration, bar_size);
}

size_t IBClient::request_historical_data(Contract con, std::string end, std::string duration,
    std::string bar_size)
{
  size_t ret{historic_bars_.push_back(std::vector<Bar>{})};
  history_symbols_[ret] = con.symbol;
  client_->reqHistoricalData(ret, con, end, duration, bar_size, "TRADES", 1, 1, false,
      TagValueListSPtr{});
  return ret;
}

void IBClient::historicalData(TickerId reqId, const Bar& bar)
{
  historic_bars_[reqId].push_back(bar);
}

void IBClient::historicalDataEnd(int reqId, const std::string& startDateStr, const std::string& endDateStr)
{
  historical_data_queue_.push(reqId);
}

bool IBClient::next_historical_id(size_t& ret)
{
  return historical_data_queue_.pop(ret);
}

std::pair<std::string, std::vector<Bar>> IBClient::historical_bars(size_t id) const
{
  return std::make_pair(history_symbols_[id], historic_bars_[id]);
}

void IBClient::req_scanner_params()
{
  client_->reqScannerParameters();
}

size_t IBClient::req_scanner_subscription(const std::string& instr, const std::string& loc,
    const std::string& code, std::vector<std::string>& apply_filters)
{
  ScannerSubscription ss;
  ss.instrument = instr;
  ss.locationCode = loc,
  ss.scanCode = code;
  size_t ret{scanner_data_.next_id()};
  TagValueListSPtr filters{new TagValueList()};

  for (size_t i{0}; i < apply_filters.size(); i += 2)
  {
    filters->push_back(TagValueSPtr(new TagValue(apply_filters[i], apply_filters[i + 1])));
  }
  client_->reqScannerSubscription(ret, ss, TagValueListSPtr(), filters);

  return ret;
}

void IBClient::scannerParameters(const std::string& xml)
{
  scanner_params_.push(xml);
}

bool IBClient::scanner_params(std::string& ret)
{
  return scanner_params_.pop(ret);
}

void IBClient::cancel_scanner(int tickerId)
{
  client_->cancelScannerSubscription(tickerId);
}

void IBClient::scannerData(int reqId, int rank, const ContractDetails& contractDetails,
  const std::string& distance, const std::string& benchmark, const std::string& projection,
  const std::string& legsStr)
{
  scanner_data_[reqId].push_back(contractDetails);
}

void IBClient::scannerDataEnd(int reqId)
{
  cancel_scanner(reqId);
  scanner_data_queue_.push(reqId);
}

bool IBClient::next_scanner_id(size_t& ret)
{
  return scanner_data_queue_.pop(ret);
}

std::vector<ContractDetails> IBClient::scanner_data(size_t id) const
{
  try
  {
    return scanner_data_[id];
  } catch (std::out_of_range& e)
  {
    return {};
  }
}

} // namespace
