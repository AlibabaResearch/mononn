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
#include <unordered_map>
#include <vector>

#include "mononn_engine/core/gpu/smem_manager.h"
#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/schedule/schedule.h"
#include "mononn_engine/core/tensor/tensor_shape.h"
namespace mononn_engine {
namespace core {
namespace op {
class ClusterOp : public Op {
 public:
  using TensorShape = mononn_engine::core::tensor::TensorShape;
  using Schedule = mononn_engine::core::schedule::Schedule;
  using Graph = mononn_engine::core::graph::Graph;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
  using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
  using Loop = mononn_engine::core::schedule::Loop;
  using SmemManager = mononn_engine::core::gpu::SmemManager;

  ClusterOp(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
            std::vector<TensorSpec> _output_specs);

  const std::vector<std::string>& get_hlo_instruction_name_list() const;
  void add_hlo_instruction_name(const std::string& _hlo_instruction_name);

  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

  virtual TensorShape get_loop_shape() const = 0;
  virtual Schedule construct_schedule(LocalityTier::Tier tier) = 0;

  void set_schedule(Schedule _schedule);
  Schedule get_schedule() const;

  void set_graph(std::shared_ptr<Graph> _graph);
  std::shared_ptr<Graph> get_graph();
  std::shared_ptr<const Graph> get_graph() const;
  Graph* get_graph_ptr();
  const Graph* get_graph_ptr() const;

  void set_cluster_type(ClusterType _cluster_type) override;

  virtual void setup_codegen() = 0;
  virtual std::string generate_cluster_code() const = 0;
  std::string generate_cluster_invocation() const;

  std::shared_ptr<OpImpl> get_implementation() const override;
  void set_implementation(std::shared_ptr<OpImpl> _impl) override;

  //        void replace_operand(std::string old_operand_name,
  //        std::shared_ptr<Op> new_operand) override;

  void set_instruction_parallel_factor(int _ilp_factor) override;
  // void propagate_ilp_index_to_implementation() override;

  virtual void trace_symbolic_index();

  void append_sub_cluster_tag(const std::string& sub_cluster_tag);
  void append_sub_cluster_tags(
      const std::vector<std::string>& sub_cluster_tags);
  void append_sub_cluster_type(const std::string& sub_cluster_type);
  void append_sub_cluster_types(
      const std::vector<std::string>& sub_cluster_types);
  void set_sub_cluster_tag_order(
      const std::vector<std::string>& _sub_cluster_tag_order);
  void set_sub_cluster_type_order(
      const std::vector<std::string>& _sub_cluster_type_order);
  const std::vector<std::string>& get_sub_cluster_tag_order() const;
  const std::vector<std::string>& get_sub_cluster_type_order() const;

  // For async prefetch.
  bool is_cluster_contain_async_prefetched_nodes() const;
  virtual std::string generate_async_pipeline_initialization() const;
  virtual std::string generate_async_pipeline_prefetch() const;
  virtual std::string generate_async_pipeline_stage_increment() const;
  virtual std::string generate_async_pipeline_flush() const;

  void set_cuda_context(std::shared_ptr<CUDAContext> _cuda_context);
  virtual void initialize_smem_manager();
  SmemManager* get_smem_manager();

  int get_horizontal_fusion_count() const;
  void set_horizontal_fusion_count(const int& _horizontal_fusion_count);

 protected:
  int horizontal_fusion_count = 1;

  std::shared_ptr<Graph> graph;
  std::shared_ptr<Schedule> schedule;
  std::vector<std::string> hlo_instruction_name_list;

  // Sequential order of sub cluster tag
  std::vector<std::string> sub_cluster_tag_order;
  // Sequential order of sub cluster type
  std::vector<std::string> sub_cluster_type_order;

  std::shared_ptr<CUDAContext> cuda_context;

  const std::string async_pipeline_stage_count_codegen_var_name =
      "total_stage_count";

  std::shared_ptr<SmemManager> smem_manager;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine