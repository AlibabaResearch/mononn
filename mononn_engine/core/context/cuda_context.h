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

#include "mononn_engine/core/common/proto_converter.h"
#include "mononn_engine/core/context/cuda_device_context.h"
#include "mononn_engine/core/context/cuda_runtime_context.h"
#include "mononn_engine/core/context/cuda_utils.h"
#include "tensorflow/mononn_extra/proto/cuda_context.pb.h"

namespace mononn_engine {
namespace core {
namespace context {
struct CUDAContext : mononn_engine::core::common::ProtoConverter<
                         tensorflow::mononn_extra::proto::CUDAContext> {
  using Dim3 = mononn_engine::core::gpu::Dim3;
  CUDAContext() {}
  CUDAContext(CUDARuntimeContext _cuda_runtime_context,
              CUDADeviceContext _cuda_device_context)
      : cuda_runtime_context(_cuda_runtime_context),
        cuda_device_context(_cuda_device_context) {}

  static CUDAContext get_cuda_context(Dim3 _grid_dim, Dim3 _block_dim,
                                      std::string _stream);
  static CUDAContext get_cuda_context(Dim3 _grid_dim, Dim3 _block_dim,
                                      std::string _stream,
                                      CUDADeviceContext _cuda_device_context);

  CUDARuntimeContext cuda_runtime_context;
  CUDADeviceContext cuda_device_context;

  int get_block_per_sm() const;

  std::unique_ptr<tensorflow::mononn_extra::proto::CUDAContext> ConvertToProto()
      const;
  void ParseFromProto(
      tensorflow::mononn_extra::proto::CUDAContext const* cuda_context);
};
}  // namespace context
}  // namespace core
}  // namespace mononn_engine