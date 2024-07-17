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

#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace core {
namespace context {
CUDAContext CUDAContext::get_cuda_context(Dim3 _grid_dim, Dim3 _block_dim,
                                          std::string _stream) {
  CUDADeviceContext _cuda_device_context =
      CUDADeviceContext::get_cuda_device_context();
  int block_per_sm = (_grid_dim.XYZ() + _cuda_device_context.sm_count - 1) /
                     _cuda_device_context.sm_count;

  CUDARuntimeContext cuda_runtime_context =
      CUDARuntimeContext(_grid_dim, _block_dim,
                         get_max_smem_size_per_block(
                             block_per_sm, _block_dim.XYZ(),
                             _cuda_device_context.max_configurable_smem_size,
                             _cuda_device_context.reserved_smem_per_block),
                         _stream);

  return CUDAContext(cuda_runtime_context, _cuda_device_context);
}

CUDAContext CUDAContext::get_cuda_context(
    Dim3 _grid_dim, Dim3 _block_dim, std::string _stream,
    CUDADeviceContext _cuda_device_context) {
  int block_per_sm = (_grid_dim.XYZ() + _cuda_device_context.sm_count - 1) /
                     _cuda_device_context.sm_count;

  CUDARuntimeContext cuda_runtime_context =
      CUDARuntimeContext(_grid_dim, _block_dim,
                         get_max_smem_size_per_block(
                             block_per_sm, _block_dim.XYZ(),
                             _cuda_device_context.max_configurable_smem_size,
                             _cuda_device_context.reserved_smem_per_block),
                         _stream);

  return CUDAContext(cuda_runtime_context, _cuda_device_context);
}

int CUDAContext::get_block_per_sm() const {
  return (this->cuda_runtime_context.grid_dim.XYZ() +
          this->cuda_device_context.sm_count - 1) /
         this->cuda_device_context.sm_count;
}

std::unique_ptr<tensorflow::mononn_extra::proto::CUDAContext>
CUDAContext::ConvertToProto() const {
  std::unique_ptr<tensorflow::mononn_extra::proto::CUDAContext> cuda_context =
      std::make_unique<tensorflow::mononn_extra::proto::CUDAContext>();

  cuda_context->set_allocated_cuda_runtime_context(
      this->cuda_runtime_context.ConvertToProto().release());
  cuda_context->set_allocated_cuda_device_context(
      this->cuda_device_context.ConvertToProto().release());

  return std::move(cuda_context);
}

void CUDAContext::ParseFromProto(
    const tensorflow::mononn_extra::proto::CUDAContext* cuda_context) {
  this->cuda_runtime_context.ParseFromProto(
      &cuda_context->cuda_runtime_context());
  this->cuda_device_context.ParseFromProto(
      &cuda_context->cuda_device_context());
}
}  // namespace context
}  // namespace core
}  // namespace mononn_engine