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

#include <memory>
#include <string>

#include "mononn_engine/core/common/proto_converter.h"
#include "tensorflow/mononn_extra/proto/arch.pb.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
class Arch : public mononn_engine::core::common::ProtoConverter<
                 tensorflow::mononn_extra::proto::Arch> {
 public:
  Arch() {}
  explicit Arch(std::string _name) : name(_name) {}
  explicit Arch(const char* _name) : Arch(std::string(_name)) {}

  static Arch const Sm70;
  static Arch const Sm75;
  static Arch const Sm80;
  static Arch const Sm86;
  static Arch const OpClassSimt;
  static Arch const OpClassTensorOp;
  static Arch const OpMultiplyAdd;

  static bool newer_or_equal(const Arch& a, const Arch& b);
  static bool is_sm_architecture(const Arch& arch);

  std::string to_string() const;

  bool operator==(Arch const& rhs) const;

  std::unique_ptr<tensorflow::mononn_extra::proto::Arch> ConvertToProto()
      const override;
  void ParseFromProto(
      tensorflow::mononn_extra::proto::Arch const* arch) override;

 private:
  std::string name;
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine