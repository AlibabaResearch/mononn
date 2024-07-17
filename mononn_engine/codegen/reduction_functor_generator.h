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

#include <unordered_map>

#include "tensorflow/compiler/xla/service/hlo_computation.h"

namespace mononn_engine {
namespace codegen {
class ReductionFunctorGeneratorRegistry;

class ReductionFunctorGenerator {
 public:
  // The hlo computation name
  const std::string& id() const;
  // The struct type name of the functor.
  const std::string& type_name() const;
  // The struct instance name of the functor.
  const std::string& instance_name() const;
  // Code for functor definition.
  std::string generate_functor_definition() const;

  static ReductionFunctorGeneratorRegistry* Registry();

  // ReductionFunctorGenerator();
  ReductionFunctorGenerator(const std::string& _reduction_node_name,
                            const std::string& _computation_name,
                            const std::string& _functor_type_name,
                            const std::string& _functor_instance_name,
                            const std::string& _functor_definition)
      : reduction_node_name(_reduction_node_name),
        computation_name(_computation_name),
        functor_type_name(_functor_type_name),
        functor_instance_name(_functor_instance_name),
        functor_definition(_functor_definition) {}

 private:
  const std::string reduction_node_name;
  const std::string computation_name;
  const std::string functor_type_name;
  const std::string functor_instance_name;
  const std::string functor_definition;

  static thread_local std::unique_ptr<ReductionFunctorGeneratorRegistry>
      registry;

  // friend class ReductionFunctorGeneratorRegistry;
  // friend std::unique_ptr<ReductionFunctorGenerator>
  //         std::make_unique<ReductionFunctorGenerator>(const std::string&,
  //         const std::string&, std::string&, std::string&, std::string&&);
};

class ReductionFunctorGeneratorRegistry {
 public:
  using ComputationName = std::string;

  // The computation should be the one in 'to_apply'.
  void add_generator(const std::string& reduction_node_name,
                     const xla::HloComputation* computation);
  const std::unordered_map<ComputationName,
                           std::unique_ptr<ReductionFunctorGenerator>>&
  get_generator() const;
  const ReductionFunctorGenerator* get_generator(
      const std::string& computaiton_name) const;

 private:
  std::unordered_map<ComputationName,
                     std::unique_ptr<ReductionFunctorGenerator>>
      registry;
};
}  // namespace codegen
}  // namespace mononn_engine