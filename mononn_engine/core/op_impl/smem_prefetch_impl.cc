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

#include "mononn_engine/core/op_impl/smem_prefetch_impl.h"

#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/defined.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Dtype = mononn_engine::core::tensor::Dtype;
using Memory = mononn_engine::core::gpu::Memory;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using CUDADefined = mononn_engine::core::gpu::CUDADefined;

std::string SmemPrefetchImpl::generate_with_index_impl() const {
  std::string op_name = this->output.get_name();

  Dtype type = this->output.get_dtype();
  std::string default_value = mononn_engine::helpers::string_format(
      "(%s)0", type.get_primitive_type().to_string().c_str());

  if ((!this->is_instruction_parallelized() &&
       this->smem_access_concrete_index_list.size() > 1) ||
      (this->is_instruction_parallelized() &&
       this->ilp_smem_access_concrete_index_list[0].size() >
           1)) {  // traced multiple index
    std::stringstream ss;
    if (this->is_instruction_parallelized()) {  // ilp
      for (auto const& traced_index :
           this->ilp_smem_access_concrete_index_list[0]) {
        auto const& index = traced_index.index_after_trace;
        // std::string reuse_op_name = op_name + "_reuse_" +
        // this->get_upstream_ilp_index_trace_node(index, 0);
        std::string reuse_op_name =
            op_name + "_reuse_" + traced_index.traced_by;
        for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
             ++ilp_id) {
          ss << type.to_string() << " "
             << mononn_engine::helpers::get_node_ilp_name(reuse_op_name, ilp_id)
             << ";\n";
        }
      }

      for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        for (auto const& traced_index :
             this->ilp_smem_access_concrete_index_list[ilp_id]) {
          const std::string& index = traced_index.index_after_trace;
          const std::string& pred = traced_index.pred_after_trace;
          // std::string reuse_op_name = op_name + "_reuse_" +
          // this->get_upstream_ilp_index_trace_node(index, ilp_id);
          std::string reuse_op_name =
              op_name + "_reuse_" + traced_index.traced_by;
          std::string ilp_reuse_op_name =
              mononn_engine::helpers::get_node_ilp_name(reuse_op_name, ilp_id);
          std::string smem_buffer_name = reuse_op_name + "_smem_buf";

          ss << ilp_reuse_op_name + " = " +
                    this->access_smem_buf(smem_buffer_name, "stage_id", index,
                                          pred, default_value)
             << ";\n";
        }
      }
    } else {  // no ilp
      for (auto const& traced_index : this->smem_access_concrete_index_list) {
        const std::string& index = traced_index.index_after_trace;
        const std::string& pred = traced_index.pred_after_trace;
        // std::string reuse_op_name = op_name + "_reuse_" +
        // this->get_upstream_index_trace_node(index);
        std::string reuse_op_name =
            op_name + "_reuse_" + traced_index.traced_by;
        std::string smem_buffer_name = reuse_op_name + "_smem_buf";
        ss << type.to_string() + " " + reuse_op_name + " = " +
                  this->access_smem_buf(smem_buffer_name, "stage_id", index,
                                        pred, default_value)
           << ";\n";
      }
    }

    return ss.str();

  } else {                                      // traced single index
    if (this->is_instruction_parallelized()) {  // ilp
      std::stringstream ss;

      for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        const std::string& index =
            this->ilp_smem_access_concrete_index_list[ilp_id][0]
                .index_after_trace;
        const std::string& pred =
            this->ilp_smem_access_concrete_index_list[ilp_id][0]
                .pred_after_trace;
        std::string ilp_op_name =
            mononn_engine::helpers::get_node_ilp_name(op_name, ilp_id);
        std::string smem_buffer_name = op_name + "_smem_buf";
        ss << type.to_string() + " " + ilp_op_name + " = " +
                  this->access_smem_buf(smem_buffer_name, "stage_id", index,
                                        pred, default_value)
           << ";\n";
      }

      return ss.str();
    } else {  // no ilp
      const std::string& index =
          this->smem_access_concrete_index_list[0].index_after_trace;
      const std::string& pred =
          smem_access_concrete_index_list[0].pred_after_trace;
      std::string smem_buffer_name = op_name + "_smem_buf";

      return type.to_string() + " " + op_name + " = " +
             this->access_smem_buf(smem_buffer_name, "stage_id", index, pred,
                                   default_value) +
             ";\n";
    }
  }
}

int SmemPrefetchImpl::get_elements_per_access() const {
  this->output.get_dtype().get_elements_per_access();
}

std::vector<Tensor> SmemPrefetchImpl::get_input_tensor() const { return {}; }

std::vector<Tensor> SmemPrefetchImpl::get_output_tensor() const {
  return {this->output};
}

void SmemPrefetchImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

void SmemPrefetchImpl::instantiate_concrete_index_impl(
    const std::vector<SymbolicIndexStamp>& symbolic_index_list,
    const std::map<std::string, std::string>& params,
    const std::string& loop_stride) {
  std::map<std::string, std::string> default_params;
  std::map<std::string, std::string> pipeline_initialization_params;
  std::map<std::string, std::string> pipeline_prefetch_params;
  std::map<std::string, std::string> smem_access_params;

  default_params["strided_index"] = params.at("strided_index");
  default_params["linear_index"] = params.at("linear_index");
  pipeline_initialization_params["strided_index"] =
      params.at("pipeline_initialization_strided_index");
  pipeline_initialization_params["linear_index"] =
      params.at("pipeline_initialization_linear_index");
  pipeline_prefetch_params["strided_index"] =
      params.at("pipeline_prefetch_strided_index");
  pipeline_prefetch_params["linear_index"] =
      params.at("pipeline_prefetch_linear_index");
  smem_access_params["linear_index"] = params.at("linear_index");

  if (this->input_spec.tier == LocalityTier::kT1) {
    this->async_pipeline_initialization_pred =
        mononn_engine::helpers::string_format(
            "%s + stage_id * %s < %s", CUDADefined::warp_global_id.c_str(),
            CUDADefined::warp_global_count.c_str(),
            params.at("prefetch_loop_boundary").c_str());
    this->async_pipeline_prefetch_pred = mononn_engine::helpers::string_format(
        "(%s + (%s - 1) * %s) < %s", params.at("strided_index").c_str(),
        params.at("async_pipeline_stage_count_codegen_var_name").c_str(),
        CUDADefined::warp_global_count.c_str(),
        params.at("prefetch_loop_boundary").c_str());
  } else if (this->input_spec.tier == LocalityTier::kT2) {
    this->async_pipeline_initialization_pred =
        mononn_engine::helpers::string_format(
            "%s + stage_id * %s < %s", CUDADefined::blockIdx_x.c_str(),
            CUDADefined::gridDim_x.c_str(),
            params.at("prefetch_loop_boundary").c_str());
    this->async_pipeline_prefetch_pred = mononn_engine::helpers::string_format(
        "(%s + (%s - 1) * %s) < %s", params.at("strided_index").c_str(),
        params.at("async_pipeline_stage_count_codegen_var_name").c_str(),
        CUDADefined::warp_global_count.c_str(),
        params.at("prefetch_loop_boundary").c_str());
  } else {
    LOG(FATAL) << "Unsupported tier: " << this->input_spec.tier.to_string();
  }

  default_params["ilp_variable_suffix"] = "";
  pipeline_initialization_params["ilp_variable_suffix"] = "";
  pipeline_prefetch_params["ilp_variable_suffix"] = "";

  this->concrete_index_list.clear();
  this->async_pipeline_initialization_concrete_index_list.clear();
  this->async_pipeline_prefetch_concrete_index_list.clear();
  this->smem_access_concrete_index_list.clear();

  for (auto const& symbolic_index : symbolic_index_list) {
    this->async_pipeline_initialization_concrete_index_list.push_back(
        symbolic_index.instantiate(pipeline_initialization_params));
    this->async_pipeline_prefetch_concrete_index_list.push_back(
        symbolic_index.instantiate(pipeline_prefetch_params));
    this->concrete_index_list.push_back(
        symbolic_index.instantiate(default_params));

    SymbolicIndexStamp smem_symbolic_index = SymbolicIndexStamp::create(
        "{linear_index}", "{linear_index}", symbolic_index.traced_by);
    this->smem_access_concrete_index_list.push_back(
        smem_symbolic_index.instantiate(smem_access_params));
  }
}

void SmemPrefetchImpl::instantiate_ilp_concrete_index_impl(
    const std::vector<SymbolicIndexStamp>& symbolic_index_list,
    const std::map<std::string, std::string>& params,
    const std::string& loop_stride, const std::string& ilp_stride) {
  this->ilp_concrete_index_list.clear();
  this->ilp_async_pipeline_initialization_concrete_index_list.clear();
  this->ilp_async_pipeline_prefetch_concrete_index_list.clear();
  this->ilp_smem_access_concrete_index_list.clear();

  this->ilp_concrete_index_list.resize(this->get_instruction_parallel_factor());
  this->ilp_async_pipeline_initialization_concrete_index_list.resize(
      this->get_instruction_parallel_factor());
  this->ilp_async_pipeline_prefetch_concrete_index_list.resize(
      this->get_instruction_parallel_factor());
  this->ilp_smem_access_concrete_index_list.resize(
      this->get_instruction_parallel_factor());

  if (this->input_spec.tier == LocalityTier::kT1) {
    this->async_pipeline_initialization_pred =
        mononn_engine::helpers::string_format(
            "%s + stage_id * %s < %s", CUDADefined::warp_global_id.c_str(),
            CUDADefined::warp_global_count.c_str(),
            params.at("prefetch_loop_boundary").c_str());
    this->async_pipeline_prefetch_pred = mononn_engine::helpers::string_format(
        "(%s + (%s - 1) * %s) < %s", params.at("strided_index").c_str(),
        params.at("async_pipeline_stage_count_codegen_var_name").c_str(),
        CUDADefined::warp_global_count.c_str(),
        params.at("prefetch_loop_boundary").c_str());
  } else if (this->input_spec.tier == LocalityTier::kT2) {
    this->async_pipeline_initialization_pred =
        mononn_engine::helpers::string_format(
            "%s + stage_id * %s < %s", CUDADefined::blockIdx_x.c_str(),
            CUDADefined::gridDim_x.c_str(),
            params.at("prefetch_loop_boundary").c_str());
    this->async_pipeline_prefetch_pred = mononn_engine::helpers::string_format(
        "(%s + (%s - 1) * %s) < %s", params.at("strided_index").c_str(),
        params.at("async_pipeline_stage_count_codegen_var_name").c_str(),
        CUDADefined::warp_global_count.c_str(),
        params.at("prefetch_loop_boundary").c_str());
  } else {
    LOG(FATAL) << "Unsupported tier: " << this->input_spec.tier.to_string();
  }

  for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
       ++ilp_id) {
    std::map<std::string, std::string> ilp_default_params;
    std::map<std::string, std::string> ilp_pipeline_initialization_params;
    std::map<std::string, std::string> ilp_pipeline_prefetch_params;
    std::map<std::string, std::string> ilp_smem_access_params;

    ilp_default_params["strided_index"] = params.at("strided_index");
    ilp_default_params["linear_index"] = mononn_engine::helpers::string_format(
        "((%s) + (%s * %d))", params.at("linear_index").c_str(),
        ilp_stride.c_str(), ilp_id);
    ilp_pipeline_initialization_params["strided_index"] =
        params.at("pipeline_initialization_strided_index");
    ilp_pipeline_initialization_params["linear_index"] =
        mononn_engine::helpers::string_format(
            "((%s) + (%s * %d))",
            params.at("pipeline_initialization_linear_index").c_str(),
            ilp_stride.c_str(), ilp_id);
    ilp_pipeline_prefetch_params["strided_index"] =
        params.at("pipeline_prefetch_strided_index");
    ilp_pipeline_prefetch_params["linear_index"] =
        mononn_engine::helpers::string_format(
            "((%s) + (%s * %d))",
            params.at("pipeline_prefetch_linear_index").c_str(),
            ilp_stride.c_str(), ilp_id);

    ilp_smem_access_params["linear_index"] =
        mononn_engine::helpers::string_format("((%s) + (%s * %d))",
                                              params.at("linear_index").c_str(),
                                              ilp_stride.c_str(), ilp_id);

    ilp_default_params["ilp_variable_suffix"] = "";
    ilp_pipeline_initialization_params["ilp_variable_suffix"] = "";
    ilp_pipeline_prefetch_params["ilp_variable_suffix"] = "";

    for (auto const& symbolic_index : symbolic_index_list) {
      this->ilp_async_pipeline_initialization_concrete_index_list[ilp_id]
          .push_back(
              symbolic_index.instantiate(ilp_pipeline_initialization_params));
      this->ilp_async_pipeline_prefetch_concrete_index_list[ilp_id].push_back(
          symbolic_index.instantiate(ilp_pipeline_prefetch_params));
      this->ilp_concrete_index_list[ilp_id].push_back(
          symbolic_index.instantiate(ilp_default_params));

      SymbolicIndexStamp smem_symbolic_index = SymbolicIndexStamp::create(
          "{linear_index}", "{linear_index}", symbolic_index.traced_by);

      this->ilp_smem_access_concrete_index_list[ilp_id].push_back(
          smem_symbolic_index.instantiate(ilp_smem_access_params));
    }
  }
}

// void SmemPrefetchImpl::propagate_attributes_impl(const
// std::unordered_map<std::string, std::string> &attrs) {
//     if (!attrs.count(OpAttribute::prefetch_predicate)) {
//         LOG(FATAL) << "SmemPrefetchImpl need prefetch_predicate attribute";
//     }

//     this->set_attribute(OpAttribute::prefetch_predicate,
//     attrs.at(OpAttribute::prefetch_predicate));
// }

std::string SmemPrefetchImpl::generate_async_pipeline_initialization() const {
  std::string op_name = this->output.get_name();
  std::string global_mem_buffer_name =
      BufferManager::get_buffer_name(op_name) + "_input";
  Dtype type = this->output.get_dtype();

  if ((!this->is_instruction_parallelized() &&
       this->concrete_index_list.size() > 1) ||
      (this->is_instruction_parallelized() &&
       this->ilp_concrete_index_list[0].size() > 1)) {  // traced multiple index
    std::stringstream ss;
    if (this->is_instruction_parallelized()) {  // ilp
      for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        for (int index_id = 0;
             index_id < this->ilp_concrete_index_list[ilp_id].size();
             ++index_id) {
          std::string global_mem_index =
              this->ilp_async_pipeline_initialization_concrete_index_list
                  [ilp_id][index_id]
                      .index_after_trace;
          std::string access_pred =
              this->ilp_async_pipeline_initialization_concrete_index_list
                  [ilp_id][index_id]
                      .pred_after_trace;
          std::string smem_index =
              this->ilp_smem_access_concrete_index_list[ilp_id][index_id]
                  .index_after_trace;
          std::string reuse_op_name =
              op_name + "_reuse_" +
              this->get_upstream_ilp_index_trace_node(
                  this->ilp_concrete_index_list[ilp_id][index_id]
                      .index_after_trace,
                  ilp_id);
          std::string smem_buffer_name = reuse_op_name + "_smem_buf";
          std::string smem_ptr =
              "&" + this->access_smem_buf(smem_buffer_name, "stage_id",
                                          smem_index, "true");
          std::string global_mem_ptr =
              "&" + this->access_global_mem_buf(global_mem_buffer_name,
                                                global_mem_index, type);

          std::string all_pred;
          if (access_pred != "true") {
            all_pred = mononn_engine::helpers::string_format(
                "(%s) && (%s)",
                this->async_pipeline_initialization_pred.c_str(),
                access_pred.c_str());
          } else {
            all_pred = this->async_pipeline_initialization_pred;
          }

          ss << this->generate_async_copy_invocation(
              type.size_in_bytes(), smem_ptr, global_mem_ptr, all_pred.c_str());
        }
      }
    } else {  // no ilp
      for (int index_id = 0; index_id < this->concrete_index_list.size();
           ++index_id) {
        std::string global_mem_index =
            this->async_pipeline_initialization_concrete_index_list[index_id]
                .index_after_trace;
        std::string access_pred =
            this->async_pipeline_initialization_concrete_index_list[index_id]
                .pred_after_trace;
        std::string smem_index =
            this->smem_access_concrete_index_list[index_id].index_after_trace;
        std::string reuse_op_name =
            op_name + "_reuse_" +
            this->get_upstream_index_trace_node(
                this->concrete_index_list[index_id].index_after_trace);
        std::string smem_buffer_name = reuse_op_name + "_smem_buf";
        std::string smem_ptr =
            "&" + this->access_smem_buf(smem_buffer_name, "stage_id",
                                        smem_index, "true");
        std::string global_mem_ptr =
            "&" + this->access_global_mem_buf(global_mem_buffer_name,
                                              global_mem_index, type);

        std::string all_pred;
        if (access_pred != "true") {
          all_pred = mononn_engine::helpers::string_format(
              "(%s) && (%s)", this->async_pipeline_initialization_pred.c_str(),
              access_pred.c_str());
        } else {
          all_pred = this->async_pipeline_initialization_pred;
        }

        ss << this->generate_async_copy_invocation(
            type.size_in_bytes(), smem_ptr, global_mem_ptr, all_pred.c_str());
      }
    }

    return ss.str();

  } else {                                      // traced single index
    if (this->is_instruction_parallelized()) {  // ilp
      std::stringstream ss;

      for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        std::string global_mem_index =
            this
                ->ilp_async_pipeline_initialization_concrete_index_list[ilp_id]
                                                                       [0]
                .index_after_trace;
        std::string access_pred =
            this
                ->ilp_async_pipeline_initialization_concrete_index_list[ilp_id]
                                                                       [0]
                .pred_after_trace;
        std::string smem_index =
            this->ilp_smem_access_concrete_index_list[ilp_id][0]
                .index_after_trace;
        std::string smem_buffer_name = op_name + "_smem_buf";
        std::string smem_ptr =
            "&" + this->access_smem_buf(smem_buffer_name, "stage_id",
                                        smem_index, "true");
        std::string global_mem_ptr =
            "&" + this->access_global_mem_buf(global_mem_buffer_name,
                                              global_mem_index, type);

        std::string all_pred;
        if (access_pred != "true") {
          all_pred = mononn_engine::helpers::string_format(
              "(%s) && (%s)", this->async_pipeline_initialization_pred.c_str(),
              access_pred.c_str());
        } else {
          all_pred = this->async_pipeline_initialization_pred;
        }

        ss << this->generate_async_copy_invocation(
            type.size_in_bytes(), smem_ptr, global_mem_ptr, all_pred);
      }

      return ss.str();
    } else {  // no ilp
      std::string global_mem_index =
          this->async_pipeline_initialization_concrete_index_list[0]
              .index_after_trace;
      std::string access_pred =
          this->async_pipeline_initialization_concrete_index_list[0]
              .pred_after_trace;
      std::string smem_index =
          this->smem_access_concrete_index_list[0].index_after_trace;
      std::string smem_buffer_name = op_name + "_smem_buf";
      std::string smem_ptr =
          "&" + this->access_smem_buf(smem_buffer_name, "stage_id", smem_index,
                                      "true");
      std::string global_mem_ptr =
          "&" + this->access_global_mem_buf(global_mem_buffer_name,
                                            global_mem_index, type);

      std::string all_pred;
      if (access_pred != "true") {
        all_pred = mononn_engine::helpers::string_format(
            "(%s) && (%s)", this->async_pipeline_initialization_pred.c_str(),
            access_pred.c_str());
      } else {
        all_pred = this->async_pipeline_initialization_pred;
      }

      return this->generate_async_copy_invocation(
          type.size_in_bytes(), smem_ptr, global_mem_ptr, all_pred.c_str());
    }
  }
}

std::string SmemPrefetchImpl::generate_async_pipeline_prefetch() const {
  std::string op_name = this->output.get_name();
  std::string global_mem_buffer_name =
      BufferManager::get_buffer_name(op_name) + "_input";
  Dtype type = this->output.get_dtype();

  if ((!this->is_instruction_parallelized() &&
       this->concrete_index_list.size() > 1) ||
      (this->is_instruction_parallelized() &&
       this->ilp_concrete_index_list[0].size() > 1)) {  // traced multiple index
    std::stringstream ss;
    if (this->is_instruction_parallelized()) {  // ilp
      for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        for (int index_id = 0;
             index_id < this->ilp_concrete_index_list[ilp_id].size();
             ++index_id) {
          std::string global_mem_index =
              this
                  ->ilp_async_pipeline_prefetch_concrete_index_list[ilp_id]
                                                                   [index_id]
                  .index_after_trace;
          std::string access_pred =
              this
                  ->ilp_async_pipeline_prefetch_concrete_index_list[ilp_id]
                                                                   [index_id]
                  .pred_after_trace;
          std::string smem_index =
              this->ilp_smem_access_concrete_index_list[ilp_id][index_id]
                  .index_after_trace;
          std::string reuse_op_name =
              op_name + "_reuse_" +
              this->get_upstream_ilp_index_trace_node(
                  this->ilp_concrete_index_list[ilp_id][index_id]
                      .index_after_trace,
                  ilp_id);
          std::string smem_buffer_name = reuse_op_name + "_smem_buf";

          std::string pipeline_prefetch_stage_index =
              "((stage_id + total_stage_count - 1) % total_stage_count)";

          std::string smem_ptr =
              "&" + this->access_smem_buf(smem_buffer_name,
                                          pipeline_prefetch_stage_index,
                                          smem_index, "true");
          std::string global_mem_ptr =
              "&" + this->access_global_mem_buf(global_mem_buffer_name,
                                                global_mem_index, type);

          std::string all_pred;
          if (access_pred != "true") {
            all_pred = mononn_engine::helpers::string_format(
                "(%s) && (%s)", this->async_pipeline_prefetch_pred.c_str(),
                access_pred.c_str());
          } else {
            all_pred = this->async_pipeline_prefetch_pred;
          }

          ss << this->generate_async_copy_invocation(
              type.size_in_bytes(), smem_ptr, global_mem_ptr, all_pred);
        }
      }
    } else {  // no ilp
      for (int index_id = 0; index_id < this->concrete_index_list.size();
           ++index_id) {
        std::string global_mem_index =
            this->async_pipeline_prefetch_concrete_index_list[index_id]
                .index_after_trace;
        std::string access_pred =
            this->async_pipeline_prefetch_concrete_index_list[index_id]
                .pred_after_trace;
        std::string smem_index =
            this->smem_access_concrete_index_list[index_id].index_after_trace;
        std::string reuse_op_name =
            op_name + "_reuse_" +
            this->get_upstream_index_trace_node(
                this->concrete_index_list[index_id].index_after_trace);
        std::string smem_buffer_name = reuse_op_name + "_smem_buf";

        std::string pipeline_prefetch_stage_index =
            "((stage_id + total_stage_count - 1) % total_stage_count)";

        std::string smem_ptr =
            "&" + this->access_smem_buf(smem_buffer_name,
                                        pipeline_prefetch_stage_index,
                                        smem_index, "true");
        std::string global_mem_ptr =
            "&" + this->access_global_mem_buf(global_mem_buffer_name,
                                              global_mem_index, type);

        std::string all_pred;
        if (access_pred != "true") {
          all_pred = mononn_engine::helpers::string_format(
              "(%s) && (%s)", this->async_pipeline_prefetch_pred.c_str(),
              access_pred.c_str());
        } else {
          all_pred = this->async_pipeline_prefetch_pred;
        }

        ss << this->generate_async_copy_invocation(
            type.size_in_bytes(), smem_ptr, global_mem_ptr, all_pred);
      }
    }

    return ss.str();

  } else {                                      // traced single index
    if (this->is_instruction_parallelized()) {  // ilp
      std::stringstream ss;

      for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        std::string global_mem_index =
            this->ilp_async_pipeline_prefetch_concrete_index_list[ilp_id][0]
                .index_after_trace;
        std::string access_pred =
            this->ilp_async_pipeline_prefetch_concrete_index_list[ilp_id][0]
                .pred_after_trace;
        std::string smem_index =
            this->ilp_smem_access_concrete_index_list[ilp_id][0]
                .index_after_trace;
        std::string smem_buffer_name = op_name + "_smem_buf";

        std::string pipeline_prefetch_stage_index =
            "((stage_id + total_stage_count - 1) % total_stage_count)";

        std::string smem_ptr =
            "&" + this->access_smem_buf(smem_buffer_name,
                                        pipeline_prefetch_stage_index,
                                        smem_index, "true");
        std::string global_mem_ptr =
            "&" + this->access_global_mem_buf(global_mem_buffer_name,
                                              global_mem_index, type);

        std::string all_pred;
        if (access_pred != "true") {
          all_pred = mononn_engine::helpers::string_format(
              "(%s) && (%s)", this->async_pipeline_prefetch_pred.c_str(),
              access_pred.c_str());
        } else {
          all_pred = this->async_pipeline_prefetch_pred;
        }

        ss << this->generate_async_copy_invocation(
            type.size_in_bytes(), smem_ptr, global_mem_ptr, all_pred);
      }

      return ss.str();
    } else {  // no ilp
      std::string global_mem_index =
          this->async_pipeline_prefetch_concrete_index_list[0]
              .index_after_trace;
      std::string access_pred =
          this->async_pipeline_prefetch_concrete_index_list[0].pred_after_trace;
      std::string smem_index =
          this->smem_access_concrete_index_list[0].index_after_trace;
      std::string smem_buffer_name = op_name + "_smem_buf";

      std::string pipeline_prefetch_stage_index =
          "((stage_id + total_stage_count - 1) % total_stage_count)";

      std::string smem_ptr =
          "&" + this->access_smem_buf(smem_buffer_name,
                                      pipeline_prefetch_stage_index, smem_index,
                                      "true");
      std::string global_mem_ptr =
          "&" + this->access_global_mem_buf(global_mem_buffer_name,
                                            global_mem_index, type);

      std::string all_pred;
      if (access_pred != "true") {
        all_pred = mononn_engine::helpers::string_format(
            "(%s) && (%s)", this->async_pipeline_prefetch_pred.c_str(),
            access_pred.c_str());
      } else {
        all_pred = this->async_pipeline_prefetch_pred;
      }

      return this->generate_async_copy_invocation(
          type.size_in_bytes(), smem_ptr, global_mem_ptr, all_pred);
    }
  }
}

std::string SmemPrefetchImpl::access_smem_buf(
    const std::string& smem_buf_name, const std::string& stage_id,
    const std::string& index, const std::string& pred,
    const std::string& default_value) const {
  std::string result = smem_buf_name;

  if (this->input_spec.tier == LocalityTier::kT1) {
    result += mononn_engine::helpers::string_format(
        "[%s]", CUDADefined::warp_block_id.c_str());
  }

  result += mononn_engine::helpers::string_format("[%s][%s]", stage_id.c_str(),
                                                  index.c_str());

  if (pred != "true") {
    result = mononn_engine::helpers::string_format("((%s) ? %s : %s)",
                                                   pred.c_str(), result.c_str(),
                                                   default_value.c_str());
  }

  return result;
}

std::string SmemPrefetchImpl::generate_async_copy_invocation(
    int bytes_per_access, std::string smem_ptr, std::string global_mem_ptr,
    std::string pred) const {
  return mononn_engine::helpers::string_format(
      "asynchronous::copy<%d, %d>()(%s, %s, %s);\n", bytes_per_access,
      bytes_per_access, smem_ptr.c_str(), global_mem_ptr.c_str(), pred.c_str());
}

std::string SmemPrefetchImpl::access_global_mem_buf(const std::string& buf_name,
                                                    const std::string& index,
                                                    const Dtype& type) const {
  return mononn_engine::helpers::string_format("reinterpret_cast<%s *>(%s)[%s]",
                                               type.to_string().c_str(),
                                               buf_name.c_str(), index.c_str());
}

std::vector<std::shared_ptr<OpImplBase>>
SmemPrefetchImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output) {
  std::shared_ptr<OpImplBase> impl =
      std::make_shared<SmemPrefetchImpl>(cuda_context, input_spec, output);

  return {impl};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine