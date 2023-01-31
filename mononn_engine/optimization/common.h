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
#include <string>

namespace mononn_engine {
namespace optimization {

struct PassName {
  static std::string BufferAssignmentPass;
  static std::string CachePrefetchPass;
  static std::string ClusteringSingleNodePass;
  static std::string ElementWiseConcatenationPass;
  static std::string ExplicitOutputPass;
  static std::string GlobalSynchronizationAssignmentPass;
  static std::string GlobalSynchronizationEliminationPass;
  static std::string ImplementationAssignmentPass;
  static std::string IntraOpReschedulePass;
  static std::string LocalityEscalationPass;
  static std::string MergeDependentPass;
  static std::string MergeIndependentPass;
  static std::string RegionalSynchronizationAssignmentPass;
  static std::string SmemPrefetchPass;
  static std::string VectorizationPass;
  static std::string ScheduleAssignmentPass;
  static std::string AccessPatternAnalysisPass;
  static std::string CacheTemporalAccessPass;
  static std::string AttributePropagationPass;
  static std::string CacheBypassPass;
  static std::string AssignCUDAContextPass;
  static std::string InitializeSmemManagerPass;
  static std::string TraceSymbolicIndexPass;
  static std::string TopologySimplificationPass;
};

}  // namespace optimization
}  // namespace mononn_engine
