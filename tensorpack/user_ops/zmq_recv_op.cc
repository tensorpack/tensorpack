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
    .Attr("hwm: int >= 1 = 10")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful()
    .Doc(R"doc(
Receive a list of Tensors by connecting to a ZMQ socket and pull from it.
The serialization format is a tensorpack custom format, defined in 'zmq_recv.py'.
)doc");


namespace tensorpack {


class ZMQRecvOp: public AsyncOpKernel {
 public:
  explicit ZMQRecvOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("types", &component_types_));
    CHECK_EQ(conn_.get(), nullptr);

    string endpoint;
    OP_REQUIRES_OK(context, context->GetAttr("end_point", &endpoint));

    int hwm;
    OP_REQUIRES_OK(context, context->GetAttr("hwm", &hwm));
    conn_.reset(new ZMQConnection(endpoint, ZMQ_PULL, hwm));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    //GuardedTimer tm("Compute");
    int start, stop;
    OP_REQUIRES_OK_ASYNC(ctx, this->OutputRange("output", &start, &stop), done);

    RecvTensorList tlist;
    conn_->recv_tensor_list(&tlist);
    auto& tensors = tlist.tensors;

    OpOutputList outputs;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("output", &outputs), done);
    CHECK(tensors.size() == num_components());

    for (int i = start; i < stop; ++i) {
      Tensor* output = nullptr;
      int j = i - start;
      auto recv_dtype = tensors[j].dtype;
      OP_REQUIRES_ASYNC(
          ctx, component_types_[j] == recv_dtype,
          errors::InvalidArgument("Type mismatch between parsed tensor (",
                                  DataTypeString(recv_dtype), ") and dtype (",
                                  DataTypeString(component_types_[j]), ")"), done);


      TensorShape& shape = tensors[j].shape;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(i, shape, &output), done);
      auto ptr = output->bit_casted_shaped<char, 1>({shape.num_elements()});
      memcpy(ptr.data(), tensors[j].buf, tensors[j].size);
      outputs.set(j, *output);
    }
    done();
  }
 private:
  DataTypeVector component_types_;
  unique_ptr<ZMQConnection> conn_;

  size_t num_components() const { return component_types_.size(); }
};


REGISTER_KERNEL_BUILDER(Name("ZMQRecv").Device(DEVICE_CPU), ZMQRecvOp);

}  // namespace tensorpack

