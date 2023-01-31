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

#include "mononn_engine/core/gpu/cutlass/conv_argument.h"

#include <sstream>

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
std::string ConvArgument::define_variable(const std::string& kernel_name,
                                          const std::string& var_name) const {
  std::stringstream ss;

  ss << "typename " << kernel_name << "::Arguments " << var_name << "{"
     << "\n";
  ss << this->problem_size << ",\n";
  ss << mononn_engine::helpers::string_format(
      "{(%s::ElementA *)%s, %s::LayoutA::packed({%s.N, %s.H, %s.W, %s.C})},\n",
      kernel_name.c_str(), this->ptr_a.c_str(), kernel_name.c_str(),
      this->problem_size.c_str(), this->problem_size.c_str(),
      this->problem_size.c_str(), this->problem_size.c_str());
  ss << mononn_engine::helpers::string_format(
      "{(%s::ElementB *)%s, %s::LayoutB::packed({%s.K, %s.R, %s.S, %s.C})},\n",
      kernel_name.c_str(), this->ptr_b.c_str(), kernel_name.c_str(),
      this->problem_size.c_str(), this->problem_size.c_str(),
      this->problem_size.c_str(), this->problem_size.c_str());
  ss << mononn_engine::helpers::string_format(
      "{(%s::ElementC *)%s, %s::LayoutC::packed({%s.N, %s.P, %s.Q, %s.K})},\n",
      kernel_name.c_str(), this->ptr_c.c_str(), kernel_name.c_str(),
      this->problem_size.c_str(), this->problem_size.c_str(),
      this->problem_size.c_str(), this->problem_size.c_str());
  ss << mononn_engine::helpers::string_format(
      "{(%s::ElementC *)%s, %s::LayoutC::packed({%s.N, %s.P, %s.Q, %s.K})},\n",
      kernel_name.c_str(), this->ptr_d.c_str(), kernel_name.c_str(),
      this->problem_size.c_str(), this->problem_size.c_str(),
      this->problem_size.c_str(), this->problem_size.c_str());
  ss << mononn_engine::helpers::string_format(
      "{%s::ElementC(%s), %s::ElementC(%s)}\n", kernel_name.c_str(),
      this->alpha.c_str(), kernel_name.c_str(), this->beta.c_str());
  ss << "};\n";
  return ss.str();
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine