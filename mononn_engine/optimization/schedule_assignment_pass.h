#pragma once
#include "mononn_engine/optimization/graph_pass.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"

namespace mononn_engine {
namespace optimization {

    class ScheduleAssignmentPass : public GraphPass {
    public:
        ScheduleAssignmentPass(const tensorflow::mononn_extra::proto::GraphSpecification *_graph_specification)
            : graph_specification(_graph_specification) {}
        std::string name() const override;
        bool run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) override;

    private:
        const tensorflow::mononn_extra::proto::GraphSpecification *graph_specification;
    };

} // onefuser
} // optimization
