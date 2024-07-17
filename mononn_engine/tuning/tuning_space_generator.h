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
#include <vector>

#include "mononn_engine/core/graph/graph.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"
namespace mononn_engine {
namespace tuning {
class TuningSpaceGenerator {
 public:
  using Graph = mononn_engine::core::graph::Graph;
  using GraphSpecification =
      tensorflow::mononn_extra::proto::GraphSpecification;
  using CUDAContext = mononn_engine::core::context::CUDAContext;

  static std::vector<std::unique_ptr<GraphSpecification>> generate_tuning_space(
      Graph* graph, std::string hlo_proto_file, std::vector<std::string> feeds,
      std::vector<std::string> input_data_files,
      std::vector<std::string> fetches);

  static std::vector<std::unique_ptr<GraphSpecification>> generate_tuning_space(
      Graph* graph, std::shared_ptr<CUDAContext> cuda_context,
      std::string hlo_proto_file, std::vector<std::string> feeds,
      std::vector<std::string> input_data_files,
      std::vector<std::string> fetches);

  static std::unique_ptr<GraphSpecification> get_default_graph_specification(
      Graph* graph, std::shared_ptr<CUDAContext> cuda_context,
      std::string hlo_proto_file, std::vector<std::string> feeds,
      std::vector<std::string> input_data_files,
      std::vector<std::string> fetches);

 private:
  static void set_allow_and_reject_list(GraphSpecification* graph_spec,
                                        Graph* graph, std::string allowed_node);
};
}  // namespace tuning
}  // namespace mononn_engine
