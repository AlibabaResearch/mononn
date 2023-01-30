#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mononn_engine/core/common/ilp_node_impl_interface.h"
#include "mononn_engine/core/common/pointer_convert.h"
#include "mononn_engine/core/context/index_trace_stamp.h"
#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/core/tensor/scalar.h"
#include "mononn_engine/core/tensor/tensor.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using ILPNodeImplInterface = mononn_engine::core::common::ILPNodeImplInterface;
using PointerConvert = mononn_engine::core::common::PointerConvert;
using Tensor = mononn_engine::core::tensor::Tensor;

class OpImplBase : public ILPNodeImplInterface, public PointerConvert {
 public:
  using Scalar = mononn_engine::core::tensor::Scalar;
  using Tensor = mononn_engine::core::tensor::Tensor;
  using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
  using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;
  using ConcreteIndexStamp = mononn_engine::core::context::ConcreteIndexStamp;
  using Functor = mononn_engine::core::gpu::Functor;

  // virtual std::vector<FunctionInvocation> get_invocations() const = 0;
  void set_invocation(FunctionInvocation _invocation);
  void set_invocation_functor(Functor _invocation_functor);

  FunctionInvocation get_invocation() const;
  Functor get_invocation_functor() const;

  void set_attribute(std::string key, std::string value);
  bool check_attribute(std::string key, std::string value) const;
  bool has_attribute(std::string key) const;
  std::string get_attribute(std::string key) const;
  void propagate_attributes(
      const std::unordered_map<std::string, std::string>& attrs);

  virtual const Scalar& get_reduce_accum() const;

  std::string generate() const;

  virtual std::vector<Tensor> get_input_tensor() const = 0;
  virtual std::vector<Tensor> get_output_tensor() const = 0;
  virtual int get_elements_per_access() const = 0;

  void set_hlo_text(std::string text);
  std::string get_hlo_text() const;

  virtual int get_smem_usage_in_bytes() const;

  //        void set_concrete_index(std::vector<ConcreteIndexStamp>
  //        _concrete_index_list);
  std::vector<ConcreteIndexStamp> get_concrete_index() const;
  ConcreteIndexStamp get_concrete_index(int idx) const;
  int get_concrete_index_count() const;

  std::string get_upstream_index_trace_node(std::string index) const;
  std::string get_upstream_ilp_index_trace_node(std::string index,
                                                int ilp_id) const;

  void add_operand_reuse_mask(std::string origin_operand_name,
                              std::string reuse_operand_name);
  bool has_operand_reuse_mask(std::string operand_name) const;
  std::string get_operand_reuse_mask(std::string operand_name) const;

  void add_auxiliary_impl(std::string key, std::shared_ptr<OpImplBase> _impl);

  virtual bool need_pre_inner_loop_generation() const;
  virtual bool need_post_inner_loop_generation() const;
  virtual std::string generate_pre_inner_loop() const;
  virtual std::string generate_post_inner_loop() const;

  void set_need_generate_with_index(bool pred);
  virtual void set_instruction_parallel_factor(int _ilp_factor) override;

  void instantiate_concrete_index(
      const std::vector<SymbolicIndexStamp>& symbolic_index_list,
      const std::map<std::string, std::string>& params,
      const std::string& loop_stride);

  void instantiate_ilp_concrete_index(
      const std::vector<SymbolicIndexStamp>& symbolic_index_list,
      const std::map<std::string, std::string>& params,
      const std::string& loop_stride, const std::string& ilp_stride);

  // virtual std::unique_ptr<OpImplBase> deep_copy() const = 0;

  // void set_ilp_traced_index(int ilp_id, std::vector<IndexTraceStamp>
  // _traced_index_list) override;
 protected:
  std::vector<ConcreteIndexStamp> concrete_index_list;
  // traced index list for each ilp element
  std::unordered_map<std::string, std::string> operand_reuse_mask;
  bool need_generate_with_index = false;
  std::unordered_map<std::string, std::shared_ptr<OpImplBase>> auxiliary_impls;

  virtual std::string generate_impl() const;
  virtual std::string generate_with_index_impl() const;

  virtual void instantiate_concrete_index_impl(
      const std::vector<SymbolicIndexStamp>& symbolic_index_list,
      const std::map<std::string, std::string>& params,
      const std::string& loop_stride);
  virtual void instantiate_ilp_concrete_index_impl(
      const std::vector<SymbolicIndexStamp>& symbolic_index_list,
      const std::map<std::string, std::string>& params,
      const std::string& loop_stride, const std::string& ilp_stride);

  // virtual void propagate_attributes_impl(const
  // std::unordered_map<std::string, std::string> &attrs);

 private:
  FunctionInvocation invocation;
  Functor invocation_functor;
  std::unordered_map<std::string, std::string> attributes;
  std::string hlo_dumped_text;
};
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine