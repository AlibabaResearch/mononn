#pragma once
#include <string>
#include <utility>

namespace mononn_engine {
namespace core {
namespace op {
// V(name, code, onefuser class name, xla op code)
#define OP_TYPE_LIST(V)                                                       \
  V("abs", abs, Abs, Abs)                                                     \
  V("add", add, Add, Add)                                                     \
  V("bitcast", bitcast, Bitcast, Bitcast)                                     \
  V("broadcast", broadcast, Broadcast, Broadcast)                             \
  V("clamp", clamp, Clamp, Clamp)                                             \
  V("compare", compare, Compare, Compare)                                     \
  V("concatenate", concatenate, Concatenate, Concatenate)                     \
  V("constant", constant, Constant, Constant)                                 \
  V("convert", convert, Convert, Convert)                                     \
  V("convolution", convolution, Convolution, Convolution)                     \
  V("copy", copy, Copy, Copy)                                                 \
  V("custom-call", custom_call, CustomCall, CustomCall)                       \
  V("divide", divide, Divide, Divide)                                         \
  V("dynamic-slice", dynamic_slice, DynamicSlice, DynamicSlice)               \
  V("dynamic-update", dynamic_update_slice, DynamicUpdateSlice,               \
    DynamicUpdateSlice)                                                       \
  V("exponential", exp, Exp, Exp)                                             \
  V("gather", gather, Gather, Gather)                                         \
  V("get-tuple-element", get_tuple_element, GetTupleElement, GetTupleElement) \
  V("iota", iota, Iota, Iota)                                                 \
  V("log", log, Log, Log)                                                     \
  V("maximum", maximum, Maximum, Maximum)                                     \
  V("minimum", minimum, Minimum, Minimum)                                     \
  V("multiply", multiply, Multiply, Multiply)                                 \
  V("negate", negate, Negate, Negate)                                         \
  V("pad", pad, Pad, Pad)                                                     \
  V("parameter", parameter, Parameter, Parameter)                             \
  V("reduce", reduce, Reduce, Reduce)                                         \
  V("reduce-window", reduce_window, ReduceWindow, ReduceWindow)               \
  V("reshape", reshape, Reshape, Reshape)                                     \
  V("rsqrt", rsqrt, Rsqrt, Rsqrt)                                             \
  V("select", select, Select, Select)                                         \
  V("sign", sign, Sign, Sign)                                                 \
  V("slice", slice, Slice, Slice)                                             \
  V("subtract", subtract, Subtract, Subtract)                                 \
  V("tanh", tanh, Tanh, Tanh)                                                 \
  V("transpose", transpose, Transpose, Transpose)                             \
  V("tuple", tuple, Tuple, Tuple)

#define OP_TYPE_LIST_CLUSTER(V) V("fusion", cluster, ClusterOp, Fusion)

#define OP_TYPE_LIST_ONE_FUSER_ADDON(V)                             \
  V("transpose-smem", transpose_smem, TransposeSmem, TransposeSmem) \
  V("global-sync", global_sync, GlobalSync, GlobalSync)             \
  V("output", output, Output, Output)

class OpType {
 public:
  OpType() = default;
  OpType(std::string _name) : name(std::move(_name)) {}
  OpType(const char* _name) : name(std::string(_name)) {}

  std::string get_name() const;
  std::string to_string() const;

#define DECLARE_OP_TYPE(op_name, op_code, ...) static const OpType op_code;

  OP_TYPE_LIST(DECLARE_OP_TYPE)
  OP_TYPE_LIST_CLUSTER(DECLARE_OP_TYPE)
  OP_TYPE_LIST_ONE_FUSER_ADDON(DECLARE_OP_TYPE)
#undef DEFINE_OP_TYPE

  bool operator==(const OpType& rhs) const;
  bool operator!=(const OpType& rhs) const;
  bool operator<(const OpType& rhs) const;

 private:
  std::string name;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine