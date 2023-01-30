#include "mononn_engine/core/op/pad.h"

#include "mononn_engine/core/context/index_transform.h"
#include "mononn_engine/core/op_impl/pad_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using Tensor = mononn_engine::core::tensor::Tensor;
using PadImpl = mononn_engine::core::op_impl::PadImpl;
using IndexTransform = mononn_engine::core::context::IndexTransform;

OpType Pad::get_type() const { return OpType::pad; }

std::vector<std::shared_ptr<OpImpl>> Pad::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  PadImpl::InputSpec input_spec;
  input_spec.operand = this->get_operand(0)->get_output_tensor(0);
  input_spec.padding_low = this->get_padding_low();
  input_spec.padding_high = this->get_padding_high();
  input_spec.padding_value = this->get_operand(1)->get_name();

  Tensor output = this->get_output_tensor(0);

  std::vector<std::shared_ptr<OpImpl>> impls =
      PadImpl::get_available_implementations(context, input_spec, output);

  for (auto& impl : impls) {
    impl->set_hlo_text(this->get_hlo_text());
  }

  return impls;
}

void Pad::set_padding_low(std::vector<int> _padding_low) {
  this->padding_low = _padding_low;
}

std::vector<int> Pad::get_padding_low() const { return this->padding_low; }

void Pad::set_padding_high(std::vector<int> _padding_high) {
  this->padding_high = _padding_high;
}

std::vector<int> Pad::get_padding_high() const { return this->padding_high; }
//
//    bool Pad::need_post_inner_loop_generation() const {
//        return true;
//    }
//
//    std::string Pad::generate_post_inner_loop() const {
//        std::stringstream ss;
//        ss << this->get_output_spec(0).get_dtype().to_string() + " " +
//        this->get_name() + ";\n";
//
//        std::string pad_index = this->traced_index_list[0].index_before_trace;
//        std::vector<std::string> multi_index =
//                IndexTransform::offset_to_multi_index(this->get_output_spec(0).get_shape(),
//                pad_index);
//
//        ss <<
//        this->get_implementation()->as<PadImpl>()->generate_if_statement_begin(multi_index);
//
//        return ss.str();
//    }
}  // namespace op
}  // namespace core
}  // namespace mononn_engine