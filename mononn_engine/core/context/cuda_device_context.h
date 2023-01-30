#pragma once

#include <string>
#include <vector>

#include "mononn_engine/core/common/proto_converter.h"
#include "mononn_engine/core/gpu/cutlass/arch.h"
#include "tensorflow/mononn_extra/proto/cuda_device_context.pb.h"

namespace mononn_engine {
namespace core {
namespace context {
namespace cutlass = mononn_engine::core::gpu::cutlass;
struct CUDADeviceContext
    : public mononn_engine::core::common::ProtoConverter<
          tensorflow::mononn_extra::proto::CUDADeviceContext> {
 public:
  // CUDADeviceContext() {}
  // CUDADeviceContext(
  //     int _device_count,
  //     int _sm_count,
  //     int _static_smem_size,
  //     int _max_configurable_smem_size,
  //     std::string _cuda_arch,
  //     int _register_per_block,
  //     int _warp_size,
  //     int reserved_smem_per_block) :
  //         device_count(_device_count),
  //         sm_count(_sm_count),
  //         static_smem_size(_static_smem_size),
  //         max_configurable_smem_size(_max_configurable_smem_size),
  //         cuda_arch(_cuda_arch),
  //         register_per_block(_register_per_block),
  //         warp_size(_warp_size) {}

  static CUDADeviceContext get_cuda_device_context();

  int device_count;
  int sm_count;
  int static_smem_size;
  int max_configurable_smem_size;
  std::string cuda_arch;
  int register_per_block;
  int warp_size;
  int reserved_smem_per_block;

  std::string to_string() const;
  cutlass::Arch get_cutlass_arch_tag() const;
  std::string get_cuda_arch_global_macro() const;

  std::unique_ptr<tensorflow::mononn_extra::proto::CUDADeviceContext>
  ConvertToProto() const;
  void ParseFromProto(tensorflow::mononn_extra::proto::CUDADeviceContext const*
                          cuda_device_context);

 private:
  static std::string to_cuda_arch(int major, int minor);
  static std::vector<std::string> supported_cuda_arch;
};
}  // namespace context
}  // namespace core
}  // namespace mononn_engine