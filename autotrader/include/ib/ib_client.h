#ifndef AHG_IB_CLIENT_H__
#define AHG_IB_CLIENT_H__

#include <stdexcept>
#include <string>
#include <memory>
#include "DefaultEWrapper.h"
#include "EClientSocket.h"
#include "EReader.h"
#include "EReaderOSSignal.h"

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

  static int client_id;
};

} // namespace
#endif // AHG_IB_CLIENT_H__
