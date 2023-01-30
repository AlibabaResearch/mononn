#include "mononn_engine/core/op_impl/op_impl_base.h"

#include <typeinfo>

#include "mononn_engine/config/config.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Scalar = mononn_engine::core::tensor::Scalar;
using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;
using ConcreteIndexStamp = mononn_engine::core::context::ConcreteIndexStamp;
using Config = mononn_engine::config::Config;
using Functor = mononn_engine::core::gpu::Functor;

void OpImplBase::set_invocation(FunctionInvocation _invocation) {
  this->invocation = _invocation;
}

void OpImplBase::set_invocation_functor(Functor _invocation_functor) {
  this->invocation_functor = _invocation_functor;
}

FunctionInvocation OpImplBase::get_invocation() const {
  return this->invocation;
}

Functor OpImplBase::get_invocation_functor() const {
  return this->invocation_functor;
}

const Scalar& OpImplBase::get_reduce_accum() const {
  LOG(FATAL) << "Not reduce operator";
}

std::string OpImplBase::generate() const {
  std::string text;
  if (Config::get()->print_hlo_text) text += "//" + this->get_hlo_text() + "\n";

  if (this->need_generate_with_index) {
    if (this->concrete_index_list.empty()) {
      LOG(FATAL) << "Concrete index not initialized";
    }

    if (this->is_instruction_parallelized() &&
        this->ilp_concrete_index_list.empty()) {
      LOG(FATAL) << "ILP concrete index not initialized";
    }

    text += this->generate_with_index_impl();
  } else {
    text += this->generate_impl();
  }

  for (auto& [key, impl] : this->auxiliary_impls) {
    text += impl->generate();
  }

  return text;
}

std::string OpImplBase::generate_impl() const {
  LOG(FATAL)
      << "OpImplBase::generate_impl need to be implemented in derived class. "
      << typeid(*this).name();
}

std::string OpImplBase::generate_with_index_impl() const {
  LOG(FATAL) << "OpImplBase::generate_with_index_impl need to be implemented "
                "in derived class."
             << typeid(*this).name();
}

void OpImplBase::set_attribute(std::string key, std::string value) {
  this->attributes[key] = value;
}

bool OpImplBase::check_attribute(std::string key, std::string value) const {
  if (this->attributes.find(key) != this->attributes.end() &&
      this->attributes.at(key) == value) {
    return true;
  } else {
    return false;
  }
}

bool OpImplBase::has_attribute(std::string key) const {
  return this->attributes.find(key) != this->attributes.end();
}

std::string OpImplBase::get_attribute(std::string key) const {
  if (!this->attributes.count(key)) {
    LOG(FATAL) << "Attribute " << key << " does not exists";
  }

  return this->attributes.at(key);
}

void OpImplBase::propagate_attributes(
    const std::unordered_map<std::string, std::string>& attrs) {
  // this->propagate_attributes_impl(attrs);
  for (auto const& [key, value] : attrs) {
    this->set_attribute(key, value);
  }

  for (auto const& [key, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->propagate_attributes(attrs);
  }
}

// void OpImplBase::propagate_attributes_impl(const
// std::unordered_map<std::string, std::string> &attrs) {

// }

void OpImplBase::set_hlo_text(std::string text) {
  this->hlo_dumped_text = text;
}

std::string OpImplBase::get_hlo_text() const { return this->hlo_dumped_text; }

int OpImplBase::get_smem_usage_in_bytes() const { return 0; }

//    void OpImplBase::set_concrete_index(std::vector<ConcreteIndexStamp>
//    _concrete_index_list) {
//        this->concrete_index_list = _concrete_index_list;
//
//        for (auto &[key, auxiliary_impl] : this->auxiliary_impls) {
//            auxiliary_impl->set_concrete_index(_concrete_index_list);
//        }
//    }

std::vector<ConcreteIndexStamp> OpImplBase::get_concrete_index() const {
  return this->concrete_index_list;
}

ConcreteIndexStamp OpImplBase::get_concrete_index(int idx) const {
  return this->concrete_index_list[idx];
}

int OpImplBase::get_concrete_index_count() const {
  return (int)this->concrete_index_list.size();
}

std::string OpImplBase::get_upstream_index_trace_node(std::string index) const {
  auto iter = std::find_if(this->concrete_index_list.begin(),
                           this->concrete_index_list.end(),
                           [&](ConcreteIndexStamp const& its) -> bool {
                             return its.index_after_trace == index;
                           });

  EXPECT_TRUE(iter != this->concrete_index_list.end(),
              "Cannot find traced index equal to " + index);

  return iter->traced_by;
}

std::string OpImplBase::get_upstream_ilp_index_trace_node(std::string index,
                                                          int ilp_id) const {
  auto iter = std::find_if(this->ilp_concrete_index_list[ilp_id].begin(),
                           this->ilp_concrete_index_list[ilp_id].end(),
                           [&](ConcreteIndexStamp const& its) -> bool {
                             return its.index_after_trace == index;
                           });

  EXPECT_TRUE(iter != this->ilp_concrete_index_list[ilp_id].end(),
              "Cannot find traced index equal to" + index +
                  " with ilp factor " + std::to_string(ilp_id));

  return iter->traced_by;
}

void OpImplBase::add_operand_reuse_mask(std::string origin_operand_name,
                                        std::string reuse_operand_name) {
  this->operand_reuse_mask[origin_operand_name] = reuse_operand_name;
}

bool OpImplBase::has_operand_reuse_mask(std::string operand_name) const {
  return this->operand_reuse_mask.find(operand_name) !=
         this->operand_reuse_mask.end();
}

std::string OpImplBase::get_operand_reuse_mask(std::string operand_name) const {
  return this->operand_reuse_mask.at(operand_name);
}

void OpImplBase::add_auxiliary_impl(std::string key,
                                    std::shared_ptr<OpImplBase> _impl) {
  this->auxiliary_impls[key] = _impl;
}

bool OpImplBase::need_pre_inner_loop_generation() const {
  for (auto& [key, impl] : this->auxiliary_impls) {
    if (impl->need_pre_inner_loop_generation()) return true;
  }

  return false;
}

bool OpImplBase::need_post_inner_loop_generation() const {
  for (auto& [key, impl] : this->auxiliary_impls) {
    if (impl->need_post_inner_loop_generation()) return true;
  }

  return false;
}

std::string OpImplBase::generate_pre_inner_loop() const {
  std::stringstream ss;

  for (auto& [key, impl] : this->auxiliary_impls) {
    if (impl->need_pre_inner_loop_generation()) {
      ss << impl->generate_pre_inner_loop();
    }
  }

  return ss.str();
}

std::string OpImplBase::generate_post_inner_loop() const {
  std::stringstream ss;

  for (auto const& [key, impl] : this->auxiliary_impls) {
    if (impl->need_post_inner_loop_generation()) {
      ss << impl->generate_post_inner_loop();
    }
  }

  return ss.str();
}

void OpImplBase::set_need_generate_with_index(bool pred) {
  this->need_generate_with_index = pred;
}

void OpImplBase::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [key, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(ilp_factor);
  }
}

void OpImplBase::instantiate_concrete_index(
    const std::vector<SymbolicIndexStamp>& symbolic_index_list,
    const std::map<std::string, std::string>& params,
    const std::string& loop_stride) {
  this->instantiate_concrete_index_impl(symbolic_index_list, params,
                                        loop_stride);

  for (auto& [key, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->instantiate_concrete_index(symbolic_index_list, params,
                                               loop_stride);
  }
}

void OpImplBase::instantiate_ilp_concrete_index(
    const std::vector<SymbolicIndexStamp>& symbolic_index_list,
    const std::map<std::string, std::string>& params,
    const std::string& loop_stride, const std::string& ilp_stride) {
  this->instantiate_ilp_concrete_index_impl(symbolic_index_list, params,
                                            loop_stride, ilp_stride);
  for (auto& [key, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->instantiate_ilp_concrete_index(symbolic_index_list, params,
                                                   loop_stride, ilp_stride);
  }
}

void OpImplBase::instantiate_concrete_index_impl(
    const std::vector<SymbolicIndexStamp>& symbolic_index_list,
    const std::map<std::string, std::string>& params,
    const std::string& loop_stride) {
  this->concrete_index_list.clear();

  auto complete_params = params;
  complete_params["ilp_variable_suffix"] = "";

  for (auto const& symbolic_index : symbolic_index_list) {
    this->concrete_index_list.push_back(
        symbolic_index.instantiate(complete_params));
  }
}

void OpImplBase::instantiate_ilp_concrete_index_impl(
    const std::vector<SymbolicIndexStamp>& symbolic_index_list,
    const std::map<std::string, std::string>& params,
    const std::string& loop_stride, const std::string& ilp_stride) {
  this->ilp_concrete_index_list.clear();
  this->ilp_concrete_index_list.resize(this->get_instruction_parallel_factor());

  for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
       ++ilp_id) {
    std::map<std::string, std::string> ilp_params;
    for (auto const& [key, value] : params) {
      if (key == "linear_index") {
        ilp_params[key] = mononn_engine::helpers::string_format(
            "((%s) + (%s * %d))", value.c_str(), ilp_stride.c_str(), ilp_id);
      } else {
        ilp_params[key] = value;
      }
    }

    ilp_params["ilp_variable_suffix"] = "__i" + std::to_string(ilp_id);

    for (auto const& symbolic_index : symbolic_index_list) {
      this->ilp_concrete_index_list[ilp_id].push_back(
          symbolic_index.instantiate(ilp_params));
    }
  }
}

// void OpImplBase::set_ilp_traced_index(int ilp_id,
// std::vector<IndexTraceStamp> _traced_index_list) {
//     if (ilp_id > this->ilp_traced_index_list.size()) {
//         LOG(FATAL) << "ILP id " << ilp_id << " out of range. Limit " <<
//         this->ilp_traced_index_list.size();
//     }

//     this->ilp_traced_index_list[ilp_id] = _traced_index_list;

//     for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
//         auxiliary_impl->set_ilp_traced_index(ilp_id, _traced_index_list);
//     }
// }
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine