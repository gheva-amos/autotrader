#ifndef AHG_IB_CLIENT_H__
#define AHG_IB_CLIENT_H__

#include <stdexcept>
#include <string>
#include <memory>
#include <atomic>
#include "DefaultEWrapper.h"
#include "EClientSocket.h"
#include "EReader.h"
#include "EReaderOSSignal.h"
#include "Contract.h"
#include "Order.h"
#include "OrderState.h"

#include "util/tsvector.h"
#include "util/tsmap.h"
#include "util/mpsc_queue.h"
#include "ib/market_data.h"

namespace autotrader
{

class IBClient : public DefaultEWrapper
{
public:
  IBClient(const std::string host, int port);
  virtual ~IBClient();
  bool is_connected() const;

  void start();
  void stop();
  void step();

  void operator()();

  virtual void error(int id, time_t errorTime, int errorCode,const std::string& errorString, const std::string& advancedOrderRejectJson) override;

  // Contracts
  size_t contract_details(Contract& contract);
  virtual void contractDetails(int reqId, const ContractDetails& contractDetails) override;
  virtual void contractDetailsEnd(int reqId) override;
  long contract_id(size_t index);

  // Find symbol
  size_t search_symbol(const std::string& symbol);
  virtual void symbolSamples(int reqId, const std::vector<ContractDescription> &contractDescriptions) override;
  std::vector<std::string>& symbols(size_t index);

  // Orders
  void request_order_ids(size_t how_many=1);
  virtual void nextValidId(OrderId orderId) override;
  OrderId next_order_id();
  OrderId place_order(Order order, Contract contract);
  OrderId buy(Contract contract, size_t how_many);
  OrderId sell(Contract contract, size_t how_many);
  virtual void openOrder(OrderId orderId, const Contract&, const Order&, const OrderState&) override;
  OrderState get_order_state(OrderId id);
  virtual void orderStatus(OrderId orderId, const std::string& status, Decimal filled,
	Decimal remaining, double avgFillPrice, long long permId, int parentId,
	double lastFillPrice, int clientId, const std::string& whyHeld, double mktCapPrice) override;

  // Market data
  size_t start_market_data_stream(Contract con);
  size_t start_market_data_stream(std::string symbol);
  void stop_market_data_stream(size_t index);
  virtual void tickPrice(TickerId tickerId, TickType field, double price, const TickAttrib& attrib) override;
  virtual void tickSize(TickerId tickerId, TickType field, Decimal size) override;
  virtual void tickGeneric(TickerId tickerId, TickType tickType, double value) override;
  virtual void tickString(TickerId tickerId, TickType tickType, const std::string& value) override;

  // Historical data
  size_t request_historical_data(std::string symbol, std::string end="", std::string duration="1 D", std::string bar_size="5 mins");
  size_t request_historical_data(Contract con, std::string end="", std::string duration="1 D", std::string bar_size="5 mins");
  virtual void historicalData(TickerId reqId, const Bar& bar) override;
  virtual void historicalDataEnd(int reqId, const std::string& startDateStr, const std::string& endDateStr) override;

  bool next_historical_id(size_t& ret);
  std::vector<Bar> historical_bars(size_t id) const;

  // Scanner ifc
  virtual void scannerData(int reqId, int rank, const ContractDetails& contractDetails,
    const std::string& distance, const std::string& benchmark, const std::string& projection,
    const std::string& legsStr) override;
  virtual void scannerDataEnd(int reqId) override;

  bool next_scanner_id(size_t& ret);
  std::vector<ContractDetails> scanner_data(size_t id) const;
protected:
  EClient* client();
private:
  void wait_for_signal();
  void process_messages();

  const std::string host_;
  int port_;

  int client_id_;
  bool connected_;
  bool started_;

  std::unique_ptr<EReaderOSSignal> signal_;
  std::unique_ptr<EClientSocket> client_;
  std::unique_ptr<EReader> reader_;

  ThreadSafeVector<Contract> contracts_;
  ThreadSafeVector<std::vector<std::string>> symbols_;
  ThreadSafeVector<OrderId> order_ids_;
  ThreadSafeMap<OrderId, OrderState> order_states_;
  ThreadSafeVector<MarketData> mkt_data_;
  ThreadSafeVector<std::vector<Bar>> historic_bars_;
  ThreadSafeMap<size_t, std::vector<ContractDetails>> scanner_data_;
  MPSCQueue<size_t> historical_data_queue_;
  MPSCQueue<size_t> scanner_data_queue_;

  std::atomic<bool> running_;

  static int client_id;
};

} // namespace
#endif // AHG_IB_CLIENT_H__
