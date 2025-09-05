#include <iostream>
#include <zmq.hpp>
#include <string>

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    throw std::runtime_error("Expecting host and port arguments");
  }
  std::string host{argv[1]}; // host.docker.internal
  std::string port{argv[2]}; // 7497

  std::string addr = host + ":" + port;
  zmq::context_t ctx{1};
  zmq::socket_t  s{ctx, zmq::socket_type::req};
  s.set(zmq::sockopt::linger, 0);
  s.connect(addr);
  s.send(zmq::buffer("hello"), zmq::send_flags::none);
  zmq::message_t reply;
  if (s.recv(reply, zmq::recv_flags::none))
  {
    std::cout << "reply: " << reply.to_string() << "\n";
    return 0;
  }
  std::cerr << "no reply\n";
  return 1;
}

