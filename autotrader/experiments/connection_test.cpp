#include <iostream>
#include <stdexcept>
#include <string>


#include "DefaultEWrapper.h"
#include "EClientSocket.h"
#include "EReader.h"
#include "EReaderOSSignal.h"
#include <iostream>
#include <memory>

class App : public DefaultEWrapper {
public:
  std::unique_ptr<EClientSocket> client;
  std::unique_ptr<EReaderOSSignal> signal;
  std::unique_ptr<EReader> reader;

  App() {
    signal = std::make_unique<EReaderOSSignal>(2000);
    client = std::make_unique<EClientSocket>(this, signal.get());
  }

  bool connect(const char* host, int port, int clientId) {
    if (!client->eConnect(host, port, clientId)) {
    std::cerr << "❌ Could not connect to " << host << ":" << port << "\n";
    return false;
    }
    std::cout << "✅ Connected to " << host << ":" << port << "\n";

    reader = std::make_unique<EReader>(client.get(), signal.get());
    reader->start();

    // Run a short message loop just to handle the handshake
    for (int i = 0; i < 20 && client->isConnected(); ++i) {
      signal->waitForSignal();
      reader->processMsgs();
    }

    client->eDisconnect();
    std::cout << "🔌 Disconnected\n";
    return true;
  }

  // minimal EWrapper stubs
  virtual void error(int id, int errorCode, const std::string& errorString, const std::string& advancedOrderRejectJson)
  {
    std::cerr << "IB error " << errorCode << ": " << errorString << "\n";
  }
  void connectionClosed() override {
    std::cout << "Connection closed by TWS\n";
  }
  void nextValidId(OrderId orderId) override {
    std::cout << "Received nextValidId: " << orderId << "\n";
  }

  // no-ops for other pure virtuals
  void tickPrice(TickerId, TickType, double, const TickAttrib&) override {}
  void tickSize(TickerId, TickType, Decimal) override {}
};

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    throw std::runtime_error("Expecting host and port arguments");
  }
  std::string host{argv[1]};
  int port{std::atoi(argv[2])};

  std::cout << "Connecting to " << host << ":" << port << std::endl;
  App app;
  app.connect(host.c_str(), port, 1);
  return 0;
}

