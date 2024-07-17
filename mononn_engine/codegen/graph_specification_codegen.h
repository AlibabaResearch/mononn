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

#include "mononn_engine/codegen/graph_codegen.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"

namespace mononn_engine {
namespace codegen {
class GraphSpecificationCodegen {
 public:
  using GraphSpecification =
      tensorflow::mononn_extra::proto::GraphSpecification;
  static std::unique_ptr<CUDAProgram> generate(
      GraphSpecification const* graph_specification);

 private:
};
}  // namespace codegen
}  // namespace mononn_engine
