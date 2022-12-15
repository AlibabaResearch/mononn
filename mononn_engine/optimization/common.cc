#pragma once
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
    std::string PassName::BufferAssignmentPass = "BufferAssignmentPass";
    std::string PassName::CachePrefetchPass = "CachePrefetchPass";
    std::string PassName::ClusteringSingleNodePass = "ClusteringSingleNodePass";
    std::string PassName::ElementWiseConcatenationPass = "ElewiseConcat";
    std::string PassName::ExplicitOutputPass = "ExplicitOutputPass";
    std::string PassName::GlobalSynchronizationAssignmentPass = "GlobalSynchronizationAssignmentPass";
    std::string PassName::GlobalSynchronizationEliminationPass = "GlobalSynchronizationEliminationPass";
    std::string PassName::ImplementationAssignmentPass = "ImplementationAssignmentPass";
    std::string PassName::IntraOpReschedulePass = "IntraOpReschedulePass";
    std::string PassName::LocalityEscalationPass = "LocalityEscalationPass";
    std::string PassName::MergeDependentPass = "MD";
    std::string PassName::MergeIndependentPass = "MI";
    std::string PassName::RegionalSynchronizationAssignmentPass = "RegionalSynchronizationAssignmentPass";
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
} // onefuser
} // optimization