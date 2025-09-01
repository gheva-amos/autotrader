#include "ib/ib_client.h"
#include <utility>
#include <thread>
#include <chrono>
#include <iostream>
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
  std::cerr << "Handling error" << id << " " << errorCode << " " << errorString << std::endl << advancedOrderRejectJson << std::endl;
}

size_t IBClient::contract_details(Contract& contract)
{
  size_t ret{contracts_.push_back(contract)};
  client()->reqContractDetails(ret, contract);
  step();
  return ret;
}

void IBClient::contractDetails(int reqId, const ContractDetails& contractDetails)
{
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
  step();
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

} // namespace
