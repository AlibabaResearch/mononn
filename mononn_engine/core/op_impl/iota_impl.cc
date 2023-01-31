// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mononn_engine/core/op_impl/iota_impl.h"

#include <memory>

#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Tensor = mononn_engine::core::tensor::Tensor;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using Dtype = mononn_engine::core::tensor::Dtype;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string IotaImpl::generate_with_index_impl() const {
  std::string node_name = this->output.get_name();
  Dtype type = this->output.get_dtype();

  std::stringstream ss;

  if (this->input_spec.iota_dimension == 0) {  // iota on the lowest dimension
    if (this->output.rank() == 1) {
      if (this->is_instruction_parallelized()) {
        for (int ilp_id; ilp_id < this->get_instruction_parallel_factor();
             ++ilp_id) {
          std::string index =
              this->ilp_concrete_index_list[ilp_id][0].index_after_trace;
          ss << type.to_string() << " "
             << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
             << " = ";
          ss << mononn_engine::helpers::string_format("(%s);\n", index.c_str());
        }
      } else {
        std::string index = this->concrete_index_list[0].index_after_trace;
        ss << type.to_string() << " " << node_name << " = ";
        ss << mononn_engine::helpers::string_format("(%s);\n", index.c_str());
      }
    } else {
      if (this->is_instruction_parallelized()) {
        for (int ilp_id; ilp_id < this->get_instruction_parallel_factor();
             ++ilp_id) {
          std::string index =
              this->ilp_concrete_index_list[ilp_id][0].index_after_trace;
          std::string div = std::to_string(
              this->output.get_shape().slice_dim(1, -1).element_count());
          ss << type.to_string() << " "
             << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
             << " = ";
          ss << mononn_engine::helpers::string_format(
              "(%s / %s);\n", index.c_str(), div.c_str());
        }
      } else {
        std::string index = this->concrete_index_list[0].index_after_trace;
        std::string div = std::to_string(
            this->output.get_shape().slice_dim(1, -1).element_count());
        ss << type.to_string() << " " << node_name << " = ";
        ss << mononn_engine::helpers::string_format("(%s / %s);\n",
                                                    index.c_str(), div.c_str());
      }
    }
  } else if (this->input_spec.iota_dimension ==
             this->output.rank() - 1) {  // iota on the highest dimension
    if (this->is_instruction_parallelized()) {
      for (int ilp_id; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        std::string index =
            this->ilp_concrete_index_list[ilp_id][0].index_after_trace;
        std::string mod = std::to_string(this->output.get_shape(-1));

        if (type.is_vectorized()) {
          ss << type.to_string() << " "
             << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
             << ";\n";

          for (int element_id = 0; element_id < type.get_elements_per_access();
               ++element_id) {
            std::string linear_index = mononn_engine::helpers::string_format(
                "(%s %% %s)", index.c_str(), mod.c_str());
            std::string iota_value = mononn_engine::helpers::string_format(
                "%s * %d + %d", linear_index.c_str(),
                type.get_elements_per_access(), element_id);
            ss << mononn_engine::helpers::string_format(
                "%s[%d] = %s;\n",
                mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
                    .c_str(),
                element_id, iota_value.c_str());
          }
        } else {
          ss << type.to_string() << " "
             << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
             << " = ";
          ss << mononn_engine::helpers::string_format(
              "(%s %% %s);\n", index.c_str(), mod.c_str());
        }
      }
    } else {
      std::string index = this->concrete_index_list[0].index_after_trace;
      std::string mod = std::to_string(this->output.get_shape(-1));

      if (type.is_vectorized()) {
        ss << type.to_string() << " " << node_name << ";\n";

        for (int element_id = 0; element_id < type.get_elements_per_access();
             ++element_id) {
          std::string linear_index = mononn_engine::helpers::string_format(
              "(%s %% %s)", index.c_str(), mod.c_str());
          std::string iota_value = mononn_engine::helpers::string_format(
              "%s * %d + %d", linear_index.c_str(),
              type.get_elements_per_access(), element_id);
          ss << mononn_engine::helpers::string_format(
              "%s[%d] = %s;\n", node_name.c_str(), element_id,
              iota_value.c_str());
        }
      } else {
        ss << type.to_string() << " " << node_name << " = ";
        ss << mononn_engine::helpers::string_format("(%s %% %s);\n",
                                                    index.c_str(), mod.c_str());
      }
    }
  } else {  // iota on other dimensions
    if (this->is_instruction_parallelized()) {
      for (int ilp_id; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        std::string index =
            this->ilp_concrete_index_list[ilp_id][0].index_after_trace;
        std::string mod =
            std::to_string(this->output.get_shape()
                               .slice_dim(this->input_spec.iota_dimension, -1)
                               .element_count());
        std::string div = std::to_string(
            this->output.get_shape()
                .slice_dim(this->input_spec.iota_dimension + 1, -1)
                .element_count());

        ss << type.to_string() << " "
           << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
           << " = ";
        ss << mononn_engine::helpers::string_format(
            "((%s %% %s) / %s);\n", index.c_str(), mod.c_str(), div.c_str());
      }
    } else {
      std::string index = this->concrete_index_list[0].index_after_trace;
      std::string mod =
          std::to_string(this->output.get_shape()
                             .slice_dim(this->input_spec.iota_dimension, -1)
                             .element_count());
      std::string div =
          std::to_string(this->output.get_shape()
                             .slice_dim(this->input_spec.iota_dimension + 1, -1)
                             .element_count());
      ss << type.to_string() << " " << node_name << " = ";
      ss << mononn_engine::helpers::string_format(
          "((%s %% %s) / %s);\n", index.c_str(), mod.c_str(), div.c_str());
    }
  }

  return ss.str();
}

std::vector<Tensor> IotaImpl::get_input_tensor() const { return {}; }

std::vector<Tensor> IotaImpl::get_output_tensor() const {
  return {this->output};
}

int IotaImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

void IotaImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

std::vector<std::shared_ptr<OpImplBase>>
IotaImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, IotaImpl::InputSpec input_spec,
    Tensor output) {
  std::shared_ptr<IotaImpl> impl =
      std::make_shared<IotaImpl>(cuda_context, input_spec, output);
  return {std::static_pointer_cast<OpImplBase>(impl)};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine