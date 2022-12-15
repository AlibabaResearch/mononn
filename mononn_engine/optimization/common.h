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

} // onefuser
} // optimization

