//File: zmq_conn.h
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#pragma once

#include <string>
#include <iostream>
#include <tensorflow/core/framework/tensor.pb.h>
#include "zmq.hpp"

namespace {
inline int read_int32(const char* p) {
  auto pi = reinterpret_cast<const int*>(p);
  return *pi;
}
}

class ZMQConnection {
 public:
  ZMQConnection(std::string endpoint, int zmq_socket_type):
    ctx_(1), sock_(ctx_, zmq_socket_type) {
    sock_.bind(endpoint.c_str());
  }

  tensorflow::TensorProto recv_tensor() {
    zmq::message_t message;
    bool succ = sock_.recv(&message);
    CHECK(succ);    // no EAGAIN, because we are blocking
    tensorflow::TensorProto ret{};
    CHECK(ret.ParseFromArray(message.data(), message.size()));
    return ret;
  }

  std::vector<tensorflow::TensorProto> recv_tensor_list() {
    zmq::message_t message;
    // TODO critical section
    bool succ = sock_.recv(&message);
    CHECK(succ);    // no EAGAIN, because we are blocking

    char* pos = reinterpret_cast<char*>(message.data());

    int num = read_int32(pos);

    std::vector<tensorflow::TensorProto> ret(num);
    pos += sizeof(int);
    for (int i = 0; i < num; ++i) {
      int size = read_int32(pos);
      pos += sizeof(int);
      //std::cout << "Message size:" << size << std::endl;
      CHECK(ret[i].ParseFromArray(pos, size));
      pos += size;
    }
    return ret;
  }

 private:
  zmq::context_t ctx_;
  zmq::socket_t sock_;
};
