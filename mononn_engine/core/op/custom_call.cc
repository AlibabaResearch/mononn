#include "mononn_engine/core/op/custom_call.h"

#include "mononn_engine/core/op_impl/conv_impl.h"
#include "mononn_engine/core/op_impl/gemm_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/helpers/env_variable.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using Tensor = mononn_engine::core::tensor::Tensor;
using GemmImpl = mononn_engine::core::op_impl::GemmImpl;
using ConvImpl = mononn_engine::core::op_impl::ConvImpl;
using EnvVar = mononn_engine::helpers::EnvVar;

OpType CustomCall::get_type() const { return OpType::custom_call; }

std::vector<std::shared_ptr<OpImpl>>
CustomCall::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  if (this->is_gemm()) {
    GemmImpl::InputSpec input_spec;
    input_spec.A = Tensor(this->get_operand(0)->get_name(),
                          this->get_operand(0)->get_output_spec(0));
    input_spec.B = Tensor(this->get_operand(1)->get_name(),
                          this->get_operand(1)->get_output_spec(0));
    if (this->get_operands().size() > 2) {
      input_spec.C =
          std::make_shared<Tensor>(this->get_operand(2)->get_name(),
                                   this->get_operand(2)->get_output_spec(0));
    }

    Tensor output(this->get_name(), this->get_output_spec(0));

    std::vector<std::shared_ptr<OpImpl>> impls =
        GemmImpl::get_available_implementations(
            context, input_spec, this->get_backend_config_str(), output);

    for (auto& impl : impls) {
      impl->set_hlo_text(this->get_hlo_text());
    }

    return impls;
  } else if (this->is_conv()) {
    ConvImpl::InputSpec input_spec;
    input_spec.A = Tensor(this->get_operand(0)->get_name(),
                          this->get_operand(0)->get_output_spec(0));
    input_spec.B = Tensor(this->get_operand(1)->get_name(),
                          this->get_operand(1)->get_output_spec(0));
    if (this->get_operands().size() > 2) {
      input_spec.C =
          std::make_shared<Tensor>(this->get_operand(2)->get_name(),
                                   this->get_operand(2)->get_output_spec(0));
    }

    input_spec.filter_size = this->filter_size;
    input_spec.filter_stride = this->filter_stride;
    input_spec.padding_low = this->padding_high;
    input_spec.padding_high = this->padding_high;

    std::string output_tensor_name = EnvVar::is_true("TF_MONONN_ENABLED")
                                         ? this->conv_output_GET_node_name
                                         :  // Output of conv is a tuple
                                         this->get_name();

    Tensor output(output_tensor_name, this->get_output_spec(0));

    std::vector<std::shared_ptr<OpImpl>> impls =
        ConvImpl::get_available_implementations(
            context, input_spec, this->get_backend_config_str(), output);

    for (auto& impl : impls) {
      impl->set_hlo_text(this->get_hlo_text());
    }

    return impls;
  }

  LOG(ERROR) << "No available implementation. Node: " << this->get_name();
  LOG(FATAL) << "Not implemented";
}

void CustomCall::set_custom_call_target(std::string _custom_call_target) {
  if (_custom_call_target != Target::cublas_gemm &&
      _custom_call_target != Target::cudnn_conv_forward) {
    LOG(FATAL) << "Unsupported custom call target " << _custom_call_target;
  }

  this->custom_call_target = _custom_call_target;
}

std::string CustomCall::get_custom_call_target() const {
  return this->custom_call_target;
}

bool CustomCall::is_gemm() const {
  return this->get_custom_call_target() == Target::cublas_gemm;
}

bool CustomCall::is_conv() const {
  return this->get_custom_call_target() == Target::cudnn_conv_forward;
}

void CustomCall::set_backend_config_str(std::string str) {
  this->backend_config_str = str;
}

std::string CustomCall::get_backend_config_str() const {
  return this->backend_config_str;
}

void CustomCall::set_filter_size(const std::vector<int>& _filter_size) {
  this->filter_size = _filter_size;
}

const std::vector<int>& CustomCall::get_filter_size() const {
  return this->filter_size;
}

void CustomCall::set_filter_stride(const std::vector<int>& _filter_stride) {
  this->filter_stride = _filter_stride;
}

const std::vector<int>& CustomCall::get_filter_stride() const {
  return this->filter_stride;
}

void CustomCall::set_padding_low(const std::vector<int>& _padding_low) {
  this->padding_low = _padding_low;
}

const std::vector<int>& CustomCall::get_padding_low() const {
  return this->padding_low;
}

void CustomCall::set_padding_high(const std::vector<int>& _padding_high) {
  this->padding_high = _padding_high;
}

const std::vector<int>& CustomCall::get_padding_high() const {
  return this->padding_high;
}

void CustomCall::set_conv_output_GTE_node_name(const std::string& node_name) {
  this->conv_output_GET_node_name = node_name;
}

const std::string& CustomCall::get_conv_output_GTE_node_name() const {
  return this->conv_output_GET_node_name;
}

std::string const CustomCall::Target::cublas_gemm = "__cublas$gemm";
std::string const CustomCall::Target::cudnn_conv_forward =
    "__cudnn$convForward";
}  // namespace op
}  // namespace core
}  // namespace mononn_engine