#include <iostream>
#include <stdexcept>
#include <string>

#include "DefaultEWrapper.h"
#include "EClientSocket.h"
#include "EReader.h"
#include "EReaderOSSignal.h"
#include <iostream>
#include <memory>

class TWSClient : public DefaultEWrapper
{
public:
  TWSClient(const std::string& host, int port);
  virtual ~TWSClient();

  void start();
  void wait_for_signal();
  void process_messages();

  bool is_connected() { return client_->isConnected(); }

  virtual void error(int id, time_t errorTime, int errorCode,const std::string& errorString, const std::string& advancedOrderRejectJson) override;

private:
  std::string host_;
  int port_;
  int client_id_;
  bool connected_;

  std::unique_ptr<EReaderOSSignal> signal_;
  std::unique_ptr<EClientSocket> client_;
  std::unique_ptr<EReader> reader_;
  static int client_id;
};

TWSClient::TWSClient(const std::string& host, int port) :
  host_{host}, port_{port}, client_id_{client_id++},
  connected_{false},
  signal_{std::make_unique<EReaderOSSignal>(2000)},
  client_{std::make_unique<EClientSocket>(this, signal_.get())}
{
  if (!client_->eConnect(host_.c_str(), port_, client_id_))
  {
    throw std::runtime_error("Could not connect to " + host_);
  }
  connected_ = true;
  reader_ = std::make_unique<EReader>(client_.get(), signal_.get());
}

TWSClient::~TWSClient()
{
  if (connected_)
  {
    client_->eDisconnect();
  }
}

void TWSClient::start()
{
  reader_->start();
}

void TWSClient::wait_for_signal()
{
  signal_->waitForSignal();
}

void TWSClient::process_messages()
{
  reader_->processMsgs();
}

void TWSClient::error(int id, time_t errorTime, int errorCode,const std::string& errorString, const std::string& advancedOrderRejectJson)
{
  std::cerr << "IB Error: " << errorCode << " " << errorString << std::endl << advancedOrderRejectJson << std::endl;
}

int TWSClient::client_id{0};

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    throw std::runtime_error("Expecting host and port arguments");
  }
  std::string host{argv[1]}; // host.docker.internal
  int port{std::atoi(argv[2])}; // 7497
  TWSClient client(host, port);
  client.start();
  for (int i{0}; i < 20 && client.is_connected(); ++i)
  {
    client.wait_for_signal();
    client.process_messages();
  }

  return 0;
}

