//File: zmq_conn.h
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#pragma once

#include <string>
#include <iostream>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/lib/gtl/inlined_vector.h>
#include "zmq.hpp"

namespace {
inline int read_int32(char** p) {
  auto pi = reinterpret_cast<const int*>(*p);
  *p += 4;
  return *pi;
}
}

struct RecvTensorList {
  zmq::message_t message;

  struct TensorConstructor {
    tensorflow::DataType dtype;
    tensorflow::TensorShape shape;
    int size; // TODO bufsize
    char* buf;
  };

  tensorflow::gtl::InlinedVector<TensorConstructor, 4> tensors;
};

class ZMQConnection {
 public:
  ZMQConnection(std::string endpoint, int zmq_socket_type):
    ctx_(1), sock_(ctx_, zmq_socket_type) {
      int hwm = 100;  // TODO make it an option
      sock_.setsockopt(ZMQ_RCVHWM, &hwm, sizeof hwm);
      sock_.bind(endpoint.c_str());
  }

  void recv_tensor_list(RecvTensorList* tlist) {
    // TODO critical section
    bool succ = sock_.recv(&tlist->message);
    CHECK(succ);    // no EAGAIN, because we are blocking

    char* pos = reinterpret_cast<char*>(tlist->message.data());

    int num = read_int32(&pos);
    auto& tensors = tlist->tensors;
    tensors.resize(num);
    CHECK_LE(num, 15);  // probably a format error

    for (int i = 0; i < num; ++i) {
      int dt = read_int32(&pos);
      tensors[i].dtype = tensorflow::DataType(dt);
      int ndim = read_int32(&pos);
      CHECK_LE(ndim, 8);  // probably an error.
      for (int k = 0; k < ndim; ++k) {
        int shp = read_int32(&pos);
        tensors[i].shape.AddDim(shp);
      }
      int sz = read_int32(&pos);
      tensors[i].buf = pos;
      tensors[i].size = sz;
      pos += sz;
    }
  }

 private:
  zmq::context_t ctx_;
  zmq::socket_t sock_;
};
