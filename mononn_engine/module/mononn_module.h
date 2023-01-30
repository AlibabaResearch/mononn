#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/codegen/cuda_program.h"
#include "mononn_engine/core/graph/graph.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"

namespace mononn_engine {
namespace module {
using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
using CUDAProgram = mononn_engine::codegen::CUDAProgram;
using Graph = mononn_engine::core::graph::Graph;
using CompileOutputType = mononn_engine::core::common::CompileOutputType::Type;

class MonoNNModule {
 public:
  MonoNNModule() = delete;
  MonoNNModule(const xla::HloModule* _hlo_module,
               std::unique_ptr<GraphSpecification> _tuning_spec,
               const std::string& _kernel_name,
               const std::vector<xla::BufferAllocation>* _allocation_list,
               const xla::HloAliasAnalysis* _alias_analysis)
      : hlo_module(_hlo_module),
        graph(nullptr),
        tuning_spec(std::move(_tuning_spec)),
        kernel_name(_kernel_name),
        allocation_list(_allocation_list),
        alias_analysis(_alias_analysis) {}

  MonoNNModule(const xla::HloModule* _hlo_module, std::unique_ptr<Graph> _graph,
               std::unique_ptr<GraphSpecification> _tuning_spec,
               const std::string& _kernel_name,
               const std::vector<xla::BufferAllocation>* _allocation_list,
               const xla::HloAliasAnalysis* _alias_analysis)
      : hlo_module(_hlo_module),
        graph(std::move(_graph)),
        tuning_spec(std::move(_tuning_spec)),
        kernel_name(_kernel_name),
        allocation_list(_allocation_list),
        alias_analysis(_alias_analysis) {}

  void set_tuning_spec(std::unique_ptr<GraphSpecification> _tuning_spec);
  const GraphSpecification* get_tuning_spec() const;

  void generate_code();
  void build_assembly(CompileOutputType type);
  void set_optimizations_have_done(int _optimizations_have_done);

  bool has_cubin() const;
  bool has_ptx() const;

  void set_cubin(const std::vector<uint8_t>& _cubin);
  void set_ptx(const std::string& _ptx);

  const std::vector<uint8_t>& get_cubin() const;
  const std::string& get_ptx() const;

  const CUDAProgram* get_cuda_program() const;

 private:
  const xla::HloModule* hlo_module;
  std::unique_ptr<CUDAProgram> cuda_program;
  std::unique_ptr<Graph> graph;
  std::unique_ptr<GraphSpecification> tuning_spec;
  std::string ptx;
  std::vector<uint8_t> cubin;
  const std::string kernel_name;
  int optimizations_have_done = 0;
  const std::vector<xla::BufferAllocation>* allocation_list;
  const xla::HloAliasAnalysis* alias_analysis;
};
}  // namespace module
}  // namespace mononn_engine