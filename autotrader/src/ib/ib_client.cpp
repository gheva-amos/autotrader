#include "ib/ib_client.h"
#include <utility>
#include <thread>
#include <chrono>
#define DEBUG
#include "debug.h"

namespace autotrader
{

int IBClient::client_id{0};

IBClient::IBClient(const std::string host, int port) :
  host_{std::move(host)}, port_{port}, client_id_{client_id},
  connected_{false}, started_{false},
  signal_{std::make_unique<EReaderOSSignal>(2000)},
  client_{std::make_unique<EClientSocket>(this, signal_.get())}
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
  if (connected_)
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
  wait_for_signal();
  process_messages();
}

void IBClient::start()
{
  if (started_)
  {
    return;
  }
  started_ = true;
  reader_->start();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  step();
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
  DBG_MSG("Handling error") << id << " " << errorCode << " " << errorString << std::endl << advancedOrderRejectJson << std::endl;
}

} // namespace
