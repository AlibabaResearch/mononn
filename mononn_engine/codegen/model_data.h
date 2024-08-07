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

#pragma once
#include <vector>

#include "Eigen/Core"
#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace codegen {
class ModelData {
 public:
  using Dtype = mononn_engine::core::tensor::Dtype;
  ModelData(std::string _file_name, Dtype _data_type, std::vector<int> _shape)
      : file_name(_file_name), data_type(_data_type), shape(_shape) {}

  void add_float_data(std::vector<float> const& _data_float);
  void add_half_data(std::vector<Eigen::half> const& _data_half);
  void add_int_data(std::vector<int> const& _data_int);

  void add_data(void* _data);

  void save_to_dir(std::string save_dir) const;

 private:
  std::string file_name;
  Dtype data_type;
  std::vector<int> shape;
  std::vector<Eigen::half> data_half;
  std::vector<float> data_float;
  std::vector<int> data_int;
};
}  // namespace codegen
}  // namespace mononn_engine
