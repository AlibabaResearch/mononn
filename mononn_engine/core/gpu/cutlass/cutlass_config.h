#pragma once

#include <string>

#include "mononn_engine/core/common/proto_converter.h"
#include "mononn_engine/core/gpu/cutlass/arch.h"
#include "mononn_engine/core/gpu/cutlass/gemm_shape.h"
#include "tensorflow/mononn_extra/proto/cutlass_config.pb.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
struct CutlassConfig : public mononn_engine::core::common::ProtoConverter<
                           tensorflow::mononn_extra::proto::CutlassConfig> {
  cutlass::GemmShape ThreadBlockShape;
  cutlass::GemmShape WarpShape;
  cutlass::GemmShape InstructionShape;
  cutlass::Arch OperatorClass;
  cutlass::Arch ArchTag;
  int stages;

  std::string to_string() const;

  std::unique_ptr<tensorflow::mononn_extra::proto::CutlassConfig>
  ConvertToProto() const override;
  void ParseFromProto(tensorflow::mononn_extra::proto::CutlassConfig const*
                          cutlass_config) override;
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine