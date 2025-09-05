#define DEBUG
#include "debug.h"
#include "server/base_thread.h"
#include "server/router.h"
#include <thread>
#include <iostream>
#include <functional>

using autotrader::get_log_stream;
int main(int argc, char** argv)
{
  if (argc != 3)
  {
    throw std::runtime_error("Expecting host and port arguments");
  }
  std::string host{argv[1]}; // host.docker.internal
  std::string port{argv[2]}; // 7001

  DBG_MSG("hello world") << std::endl;

  zmq::context_t ctx{1};
  autotrader::BaseThread::start_all_threads();
  autotrader::Router router{ctx, "tcp://0.0.0.0:7001", zmq::socket_type::router};
  std::thread router_thread{std::ref(router)};

  std::cout << "Server running, press Enter to quit\n";
  std::string dummy;
  std::getline(std::cin, dummy);

  autotrader::BaseThread::stop_all_threads();
  router_thread.join();
}

