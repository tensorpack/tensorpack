//File: zmq_recv_op.cc
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include <string>
#include <memory>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "zmq_conn.h"

using namespace std;
using namespace tensorflow;

REGISTER_OP("ZMQRecv")
    .Output("output: types")
    .Attr("end_point: string")
    .Attr("types: list(type) >= 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful()
    .Doc(R"doc(
Receive and return a serialized list of TensorProto from a ZMQ socket.
)doc");


class ZMQRecvOp: public OpKernel {
 public:
  explicit ZMQRecvOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("types", &component_types_));
    CHECK(conn_.get() == nullptr);

    string endpoint;
    OP_REQUIRES_OK(context, context->GetAttr("end_point", &endpoint));
    conn_.reset(new ZMQConnection(endpoint, ZMQ_PULL));
  }

  void Compute(OpKernelContext* ctx) override {
    int start, stop;
    TF_CHECK_OK(this->OutputRange("output", &start, &stop));

    //cout << "COMPUTE" << endl;
    auto protos = conn_->recv_tensor_list();

    OpOutputList outputs;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &outputs));
    CHECK(protos.size() == num_components());

    for (int i = start; i < stop; ++i) {
      Tensor output;
      int j = i - start;
      OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(
                              protos[j], ctx->output_alloc_attr(i), &output));
      OP_REQUIRES(
          ctx, component_types_[j] == output.dtype(),
          errors::InvalidArgument("Type mismatch between parsed tensor (",
                                  DataTypeString(output.dtype()), ") and dtype (",
                                  DataTypeString(component_types_[j]), ")"));
      outputs.set(j, output);
    }
  }
 private:
  DataTypeVector component_types_;
  unique_ptr<ZMQConnection> conn_;

  size_t num_components() const { return component_types_.size(); }
};

REGISTER_KERNEL_BUILDER(Name("ZMQRecv").Device(DEVICE_CPU), ZMQRecvOp);
