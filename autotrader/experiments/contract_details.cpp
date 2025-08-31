#include "ib/ib_client.h"
#include "Contract.h"
#include <future>
#define DEBUG
#include "debug.h"

using autotrader::get_log_stream;

class Client : public autotrader::IBClient
{
public:
  Client(std::string host, int port);
  void contract_details(Contract& contract);
  virtual void contractDetails(int reqId, const ContractDetails& contractDetails) override;

  long contract_id(Contract contract, size_t index);
private:
  void req_contract(Contract contract);
  bool asked_;
  Contract asked_for_;
  std::vector<Contract> contracts_;
};

Client::Client(std::string host, int port) :
  autotrader::IBClient(host, port), asked_{false}
{
}

void Client::contract_details(Contract& contract)
{
  client()->reqContractDetails(contracts_.size(), contract);
  contracts_.push_back(contract);
}

void Client::contractDetails(int reqId, const ContractDetails& contractDetails)
{
  contracts_[reqId] = contractDetails.contract;
  asked_ = true;
  asked_for_ = contractDetails.contract;
  //DBG_MSG("Contract Details:") << "id: " << contractDetails.contract.conId << std::endl;
}

void Client::req_contract(Contract contract)
{
  if (!asked_)
  {
    std::future<void> query = std::async([this, &contract] -> void {
        contract_details(contract);
        step();
        });
    query.get();
  }
}

long Client::contract_id(Contract contract, size_t index)
{
  req_contract(contract);
  return contracts_[index].conId;
}

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    throw std::runtime_error("Expecting host and port arguments");
  }
  std::string host{argv[1]}; // host.docker.internal
  int port{std::atoi(argv[2])}; // 7497
  Client client(host, port);

  Contract contract;
  contract.symbol = "INTC";
  contract.secType = "STK";
  contract.currency = "USD";
  contract.exchange = "SMART";

  client.start();
  DBG_MSG("Contract ID is:") << client.contract_id(contract, 0) << std::endl;

  return 0;
}

