//File: zmq_recv_op.cc
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include <string>
#include <memory>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/resource_op_kernel.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/common_shape_fns.h>

#include "zmq_conn.h"

using namespace std;
using namespace tensorflow;

namespace tensorpack {

// An op to create zmq connection as a resource.
// Use ResourceOpKernel to ensure singleton construction.
class ZMQConnectionHandleOp : public ResourceOpKernel<ZMQConnection> {
  public:
    explicit ZMQConnectionHandleOp(OpKernelConstruction* ctx)
        : ResourceOpKernel<ZMQConnection>(ctx) {}

  private:
    Status CreateResource(ZMQConnection** ret) override EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      const NodeDef& ndef = def();
      ZMQSocketDef sockdef;
      sockdef.socket_type = ZMQ_PULL;
      TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "bind", &sockdef.bind));
      TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "end_point", &sockdef.end_point));
      TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "hwm", &sockdef.hwm));
      *ret = new ZMQConnection(sockdef);
      return Status::OK();
    }

    // Can verify, but probably not necessary because python is not going to eval this op twice with
    // the same shared name
};


class ZMQRecvOp: public AsyncOpKernel {
 public:
  explicit ZMQRecvOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("types", &component_types_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    ZMQConnection* conn = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &conn), done);

    RecvTensorList tlist;
    conn->recv_tensor_list(&tlist);
    auto& tensors = tlist.tensors;
    CHECK(tensors.size() == num_components());

    for (int i = 0; i < tensors.size(); ++i) {
      Tensor* output = nullptr;
      auto recv_dtype = tensors[i].dtype;
      OP_REQUIRES_ASYNC(
          ctx, component_types_[i] == recv_dtype,
          errors::InvalidArgument("Type mismatch at index ", std::to_string(i),
                                  " between received tensor (", DataTypeString(recv_dtype),
                                  ") and dtype (", DataTypeString(component_types_[i]), ")"),
          done);


      TensorShape& shape = tensors[i].shape;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(i, shape, &output), done);
      // reinterpret cast and then memcpy
      auto ptr = output->bit_casted_shaped<char, 1>({shape.num_elements()}).data();
      memcpy(ptr, tensors[i].buf, tensors[i].buf_size);
      ctx->set_output(i, *output);
    }
    done();
  }
 private:
  DataTypeVector component_types_;

  size_t num_components() const { return component_types_.size(); }
};


REGISTER_KERNEL_BUILDER(Name("ZMQRecv").Device(DEVICE_CPU), ZMQRecvOp);
REGISTER_KERNEL_BUILDER(Name("ZMQConnection").Device(DEVICE_CPU), ZMQConnectionHandleOp);

}  // namespace tensorpack

REGISTER_OP("ZMQRecv")
    .Input("handle: resource")
    .Output("output: types")
    .Attr("types: list(type) >= 1")

    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful()
    .Doc(R"doc(
Receive a list of Tensors from a ZMQ connection handle.
The serialization format is a tensorpack custom format, defined in 'zmq_recv.py'.
)doc");


REGISTER_OP("ZMQConnection")
    .Output("handle: resource")
    .Attr("end_point: string")
    .Attr("hwm: int >= 1 = 10")
    .Attr("bind: bool = true")

    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")

    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Opens a ZMQ PULL socket and returns a handle to it as a resource.
end_point: the ZMQ end point.
hwm: ZMQ high-water mark.
bind: If false, will connect to the endpoint rather than bind to it.
container: required for a resource op kernel.
shared_name: If non-empty, this connection will be shared under the given name across multiple sessions.
)doc");
