#pragma once
#include <memory>
#include <string>

#include "mononn_engine/core/op/broadcast.h"
#include "mononn_engine/core/op/concatenate.h"
#include "mononn_engine/core/op/copy.h"
#include "mononn_engine/core/op/dynamic_slice.h"
#include "mononn_engine/core/op/dynamic_update_slice.h"
#include "mononn_engine/core/op/gather.h"
#include "mononn_engine/core/op/pad.h"
#include "mononn_engine/core/op/reduce.h"
#include "mononn_engine/core/op/reduce_window.h"
#include "mononn_engine/core/op/reshape.h"
#include "mononn_engine/core/op/slice.h"
#include "mononn_engine/core/op/transpose.h"
#include "mononn_engine/core/tensor/tensor_shape.h"

namespace mononn_engine {
namespace core {
namespace context {
struct IndexSymbols {
  static const std::string linear_index;
  static const std::string strided_index;
  static const std::string ilp_variable_suffix;
  static const std::string ilp_index_id;
};

class IndexTracer {
 public:
  using Op = mononn_engine::core::op::Op;
  using Broadcast = mononn_engine::core::op::Broadcast;
  using DynamicSlice = mononn_engine::core::op::DynamicSlice;
  using DynamicUpdateSlice = mononn_engine::core::op::DynamicUpdateSlice;
  using Gather = mononn_engine::core::op::Gather;
  using Pad = mononn_engine::core::op::Pad;
  using Transpose = mononn_engine::core::op::Transpose;
  using Slice = mononn_engine::core::op::Slice;
  using Reduce = mononn_engine::core::op::Reduce;
  using Concatenate = mononn_engine::core::op::Concatenate;
  using TensorShape = mononn_engine::core::tensor::TensorShape;
  using ReduceWindow = mononn_engine::core::op::ReduceWindow;

  IndexTracer() {}
  IndexTracer(std::string _index) : index(_index) {}

  void set_index(std::string _index);
  void set_pred(std::string _pred);
  std::string get_index() const;
  std::string get_predictive() const;

  void trace(const std::shared_ptr<const Op>& op);

  void trace_broadcast(const std::shared_ptr<const Broadcast>& op);
  void trace_concatenate(const std::shared_ptr<const Concatenate>& op,
                         int operand_id);
  void trace_dynamic_slice(const std::shared_ptr<const DynamicSlice>& op);
  void trace_dynamic_update_slice_operand(
      const std::shared_ptr<const DynamicUpdateSlice>& op);
  void trace_dynamic_update_slice_update(
      const std::shared_ptr<const DynamicUpdateSlice>& op);
  void trace_gather_operand(const std::shared_ptr<const Gather>& op);
  void trace_gather_operand_ilp(const std::shared_ptr<const Gather>& op,
                                int ilp_id);
  void trace_gather_indices(const std::shared_ptr<const Gather>& op);
  void trace_pad(const std::shared_ptr<const Pad>& op);
  void trace_reduce(const std::shared_ptr<const Reduce>& op,
                    std::string const& inverse_reduce_dim);
  void trace_reduce_window(const std::shared_ptr<const ReduceWindow>& op);
  void trace_slice(const std::shared_ptr<const Slice>& op);
  void trace_transpose(const std::shared_ptr<const Transpose>& op);

 private:
  std::string index;
  std::string pred = "true";
};
}  // namespace context
}  // namespace core
}  // namespace mononn_engine