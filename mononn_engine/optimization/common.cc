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
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
std::string PassName::BufferAssignmentPass = "BufferAssignmentPass";
std::string PassName::CachePrefetchPass = "CachePrefetchPass";
std::string PassName::ClusteringSingleNodePass = "ClusteringSingleNodePass";
std::string PassName::ElementWiseConcatenationPass = "ElewiseConcat";
std::string PassName::ExplicitOutputPass = "ExplicitOutputPass";
std::string PassName::GlobalSynchronizationAssignmentPass =
    "GlobalSynchronizationAssignmentPass";
std::string PassName::GlobalSynchronizationEliminationPass =
    "GlobalSynchronizationEliminationPass";
std::string PassName::ImplementationAssignmentPass =
    "ImplementationAssignmentPass";
std::string PassName::IntraOpReschedulePass = "IntraOpReschedulePass";
std::string PassName::LocalityEscalationPass = "LocalityEscalationPass";
std::string PassName::MergeDependentPass = "MD";
std::string PassName::MergeIndependentPass = "MI";
std::string PassName::RegionalSynchronizationAssignmentPass =
    "RegionalSynchronizationAssignmentPass";
std::string PassName::SmemPrefetchPass = "SmemPrefetchPass";
std::string PassName::VectorizationPass = "VectorizationPass";
std::string PassName::ScheduleAssignmentPass = "ScheduleAssignmentPass";
std::string PassName::AccessPatternAnalysisPass = "AccessPatternAnalysisPass";
std::string PassName::CacheTemporalAccessPass = "CacheTemporalAccessPass";
std::string PassName::AttributePropagationPass = "AttributePropagationPass";
std::string PassName::CacheBypassPass = "CacheBypassPass";
std::string PassName::AssignCUDAContextPass = "AssignCUDAContextPass";
std::string PassName::InitializeSmemManagerPass = "InitializeSmemManagerPass";
std::string PassName::TraceSymbolicIndexPass = "TraceSymbolicIndexPass";
std::string PassName::TopologySimplificationPass = "TopologySimplificationPass";
}  // namespace optimization
}  // namespace mononn_engine