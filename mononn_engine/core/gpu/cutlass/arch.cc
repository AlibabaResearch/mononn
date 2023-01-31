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

#include "mononn_engine/core/gpu/cutlass/arch.h"

#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
Arch const Arch::Sm70 = Arch("cutlass::arch::Sm70");
Arch const Arch::Sm75 = Arch("cutlass::arch::Sm75");
Arch const Arch::Sm80 = Arch("cutlass::arch::Sm80");
Arch const Arch::Sm86 = Arch("cutlass::arch::Sm86");
Arch const Arch::OpClassSimt = Arch("cutlass::arch::OpClassSimt");
Arch const Arch::OpClassTensorOp = Arch("cutlass::arch::OpClassTensorOp");
Arch const Arch::OpMultiplyAdd = Arch("cutlass::arch::OpMultiplyAdd");

bool Arch::newer_or_equal(const Arch& a, const Arch& b) {
  if (!Arch::is_sm_architecture(a) || !Arch::is_sm_architecture(b))
    LOG(FATAL) << "Invalid";
  int code_a = std::stoi(a.to_string().substr(17, 2));
  int code_b = std::stoi(b.to_string().substr(17, 2));

  return code_a >= code_b;
}

bool Arch::is_sm_architecture(const Arch& arch) {
  return arch == Arch::Sm70 || arch == Arch::Sm75 || arch == Arch::Sm80 ||
         arch == Arch::Sm86;
}

std::string Arch::to_string() const { return this->name; }

bool Arch::operator==(Arch const& rhs) const { return this->name == rhs.name; }

std::unique_ptr<tensorflow::mononn_extra::proto::Arch> Arch::ConvertToProto()
    const {
  std::unique_ptr<tensorflow::mononn_extra::proto::Arch> arch =
      std::make_unique<tensorflow::mononn_extra::proto::Arch>();

  arch->set_name(this->name);

  return std::move(arch);
}

void Arch::ParseFromProto(const tensorflow::mononn_extra::proto::Arch* arch) {
  this->name = arch->name();
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine