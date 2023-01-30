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
