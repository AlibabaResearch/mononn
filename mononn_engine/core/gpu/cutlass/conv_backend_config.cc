#include "mononn_engine/core/gpu/cutlass/conv_backend_config.h"

#include <algorithm>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
std::unique_ptr<tensorflow::mononn_extra::proto::ConvBackendConfig>
ConvBackendConfig::ConvertToProto() const {
  std::unique_ptr<tensorflow::mononn_extra::proto::ConvBackendConfig>
      conv_backend_config = std::make_unique<
          tensorflow::mononn_extra::proto::ConvBackendConfig>();

  conv_backend_config->set_conv_result_scale(this->conv_result_scale);
  conv_backend_config->set_side_input_scale(this->side_input_scale);
  conv_backend_config->set_tensor_ops_enabled(this->tensor_ops_enabled);
  conv_backend_config->set_activation_mode(this->activation_mode);

  return std::move(conv_backend_config);
}

void ConvBackendConfig::ParseFromProto(
    const tensorflow::mononn_extra::proto::ConvBackendConfig*
        conv_backend_config) {
  this->conv_result_scale = conv_backend_config->conv_result_scale();
  this->side_input_scale = conv_backend_config->side_input_scale();
  this->tensor_ops_enabled = conv_backend_config->tensor_ops_enabled();
  this->activation_mode = conv_backend_config->activation_mode();
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine