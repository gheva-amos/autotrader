#define DEBUG
#include "debug.h"
#include "ib/ib_client.h"
#include "server/base_thread.h"
#include "server/router.h"
#include "server/pull.h"
#include "server/pub.h"
#include <thread>
#include <iostream>
#include <functional>
#include <memory>

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
  client.start();
  std::thread ib_thread{std::ref(client)};

  zmq::context_t ctx{1};
  autotrader::BaseThread::start_all_threads();
  autotrader::Router router{ctx, "tcp://0.0.0.0:7001", client};
  std::thread router_thread{std::ref(router)};

  autotrader::Pub pub{ctx, "tcp://0.0.0.0:7002", client};
  std::thread pub_thread{std::ref(pub)};

  autotrader::Pull pull{ctx, "tcp://0.0.0.0:7003", client};
  std::thread pull_thread{std::ref(pull)};

  std::cout << "Server running, press Enter to quit\n";
  std::string dummy;
  std::getline(std::cin, dummy);

  autotrader::BaseThread::stop_all_threads();
  client.stop();
  router_thread.join();
  pub_thread.join();
  pull_thread.join();
  ib_thread.join();
}

