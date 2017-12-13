//File: zmq_conn.h
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#pragma once

#include <string>
#include <iostream>
#include <thread>

#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/lib/gtl/inlined_vector.h>
#include <tensorflow/core/lib/strings/strcat.h>
#include <tensorflow/core/platform/mutex.h>
#include "zmq.hpp"

namespace {
inline int read_int32(char** p) {
  auto pi = reinterpret_cast<const int*>(*p);
  *p += 4;
  return *pi;
}

inline tensorflow::int64 read_int64(char** p) {
  auto pi = reinterpret_cast<const tensorflow::int64*>(*p);
  *p += 8;
  return *pi;
}
}

namespace tensorpack {

struct ZMQSocketDef {
  std::string end_point;
  int socket_type,  // ZMQ_PULL
      hwm;
  bool bind;  // bind or connect

  std::string DebugString() const {
    return tensorflow::strings::StrCat("EndPoint=", end_point, ", hwm=", std::to_string(hwm));
  }
};

struct RecvTensorList {
  zmq::message_t message;

  struct TensorConstructor {
    tensorflow::DataType dtype;
    tensorflow::TensorShape shape;
    tensorflow::int64 buf_size;
    char* buf;
  };

  tensorflow::gtl::InlinedVector<TensorConstructor, 4> tensors;
};

class ZMQConnection : public tensorflow::ResourceBase {
 public:
  explicit ZMQConnection(const ZMQSocketDef& def):
    def_{def}, ctx_{1}, sock_{ctx_, def.socket_type} {
      int linger = 0;
      sock_.setsockopt(ZMQ_LINGER, &linger , sizeof linger);

      sock_.setsockopt(ZMQ_RCVHWM, &def.hwm , sizeof def.hwm);
      if (def.bind) {
        sock_.bind(def.end_point.c_str());
      } else {
        sock_.connect(def.end_point.c_str());
      }
  }

  std::string DebugString() override { return def_.DebugString(); }

  void recv_tensor_list(RecvTensorList* tlist) {
    {
      // https://www.tensorflow.org/extend/adding_an_op#multi-threaded_cpu_kernels
      // zmq socket is not thread safe
      tensorflow::mutex_lock lk(mu_);
      bool succ = sock_.recv(&tlist->message);  // block until some data appears
      // TODO this may throw, handle exception?
      // Possible error code: http://api.zeromq.org/3-3:zmq-msg-recv
      // succ=false only if EAGAIN
      CHECK(succ);    // no EAGAIN, because we are blocking
    }

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
      tensorflow::int64 sz = read_int64(&pos);
      tensors[i].buf = pos;
      tensors[i].buf_size = sz;
      pos += sz;
    }
  }

  const ZMQSocketDef& get_socket_def() const { return def_; }

 private:
  ZMQSocketDef def_;
  tensorflow::mutex mu_;
  zmq::context_t ctx_;
  zmq::socket_t sock_;
};

} // namespace tensorpack

