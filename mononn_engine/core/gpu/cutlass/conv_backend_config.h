#pragma once

#include <string>
#include <vector>

#include "mononn_engine/core/common/proto_converter.h"
#include "tensorflow/mononn_extra/proto/conv_backend_config.pb.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
struct ConvBackendConfig
    : public mononn_engine::core::common::ProtoConverter<
          tensorflow::mononn_extra::proto::ConvBackendConfig> {
  float conv_result_scale;
  float side_input_scale;
  bool tensor_ops_enabled;
  std::string activation_mode;

  std::unique_ptr<tensorflow::mononn_extra::proto::ConvBackendConfig>
  ConvertToProto() const override;
  void ParseFromProto(tensorflow::mononn_extra::proto::ConvBackendConfig const*
                          conv_backend_config) override;
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine