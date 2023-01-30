#include "mononn_engine/core/op/reduce_window.h"

#include "mononn_engine/helpers/stl_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using Scalar = mononn_engine::core::tensor::Scalar;
using ReductionFunctorGenerator =
    mononn_engine::codegen::ReductionFunctorGenerator;

// An interim workaround to ImplementationAssignmentPass pass
class MockImpl : public OpImpl {
 public:
  std::vector<Tensor> get_input_tensor() const override {
    LOG(FATAL) << "Unimplemented";
  }

  std::vector<Tensor> get_output_tensor() const override {
    LOG(FATAL) << "Unimplemented";
  }

  int get_elements_per_access() const override {
    LOG(FATAL) << "Unimplemented";
  }

 private:
};

OpType ReduceWindow::get_type() const { return OpType::reduce_window; }

std::vector<std::shared_ptr<OpImpl>>
ReduceWindow::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  return {std::make_shared<MockImpl>()};
}

void ReduceWindow::set_init_value(const Scalar& _init_value) {
  this->init_value = _init_value;
}

const Scalar& ReduceWindow::get_init_value() const { return this->init_value; }

void ReduceWindow::set_filter_size(const std::vector<int>& _filter_size) {
  this->filter_size = _filter_size;
}

const std::vector<int>& ReduceWindow::get_filter_size() const {
  return this->filter_size;
}

void ReduceWindow::set_filter_stride(const std::vector<int>& _filter_stride) {
  this->filter_stride = _filter_stride;
}

const std::vector<int>& ReduceWindow::get_filter_stride() const {
  return this->filter_stride;
}

void ReduceWindow::set_padding_low(const std::vector<int>& _padding_low) {
  this->padding_low = _padding_low;
}

const std::vector<int>& ReduceWindow::get_padding_low() const {
  return this->padding_low;
}

void ReduceWindow::set_padding_high(const std::vector<int>& _padding_high) {
  this->padding_high = _padding_high;
}

const std::vector<int>& ReduceWindow::get_padding_high() const {
  return this->padding_high;
}

void ReduceWindow::set_reduction_functor_generator(
    const ReductionFunctorGenerator* _reduction_functor_generator) {
  this->reduction_functor_generator = _reduction_functor_generator;
}

const ReductionFunctorGenerator* ReduceWindow::get_reduction_functor_generator()
    const {
  return this->reduction_functor_generator;
}

std::vector<std::vector<int>> ReduceWindow::get_window_positions() const {
  std::vector<std::vector<int>> positions(this->filter_size[0]);

  for (int i = 0; i < this->filter_size[0]; ++i) {
    positions[i].push_back(i);
  }

  for (int idx = 1; idx < this->filter_size.size(); ++idx) {
    std::vector<int> new_pos;
    for (int i = 0; i < this->filter_size[idx]; ++i) {
      new_pos.push_back(i);
    }

    positions = mononn_engine::helpers::cartesian_join<std::vector<int>, int,
                                                       std::vector<int>>(
        positions, new_pos,
        [](const std::vector<int>& vec, const int& pos) -> std::vector<int> {
          auto res = vec;
          res.push_back(pos);
          return res;
        });
  }

  return positions;
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine