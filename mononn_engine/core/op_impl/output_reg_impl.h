#pragma once
#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/op_impl/output_impl_base.h"
#include "mononn_engine/core/tensor/tensor.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
class OutputRegImpl : public OutputImplBase {
 public:
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using Tensor = mononn_engine::core::tensor::Tensor;

  struct InputSpec {
    Tensor operand;
    std::string reg_buffer_name;
    std::string step_id;
    std::string step_cnt;
  };

  OutputRegImpl(std::shared_ptr<CUDAContext> _cuda_context,
                InputSpec _input_spec)
      : cuda_context(_cuda_context), input_spec(_input_spec) {}

  std::string generate_impl() const override;

  std::vector<Tensor> get_input_tensor() const override;
  std::vector<Tensor> get_output_tensor() const override;
  int get_elements_per_access() const override;
  void set_instruction_parallel_factor(int _ilp_factor) override;

  bool need_pre_inner_loop_generation() const override;
  std::string generate_pre_inner_loop() const override;

 private:
  std::shared_ptr<CUDAContext> cuda_context;
  InputSpec input_spec;
};
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine