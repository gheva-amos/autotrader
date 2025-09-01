#ifndef AHG_IB_CLIENT_H__
#define AHG_IB_CLIENT_H__

#include <stdexcept>
#include <string>
#include <memory>
#include "DefaultEWrapper.h"
#include "EClientSocket.h"
#include "EReader.h"
#include "EReaderOSSignal.h"
#include "Contract.h"

#include "util/tsvector.h"

namespace autotrader
{

class IBClient : public DefaultEWrapper
{
public:
  IBClient(const std::string host, int port);
  virtual ~IBClient();
  bool is_connected() const;

  void start();
  void step();

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

  static int client_id;
};

} // namespace
#endif // AHG_IB_CLIENT_H__
