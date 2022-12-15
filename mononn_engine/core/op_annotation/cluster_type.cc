#include "mononn_engine/core/op_annotation/cluster_type.h"


namespace mononn_engine {
namespace core {
namespace op_annotation {
    ClusterType const ClusterType::None = "None";
    ClusterType const ClusterType::Reduce = "Reduce";
    ClusterType const ClusterType::Elewise = "Elewise";
    ClusterType const ClusterType::GemmEpilogue = "GemmEpilogue";
    ClusterType const ClusterType::Gemm = "Gemm";
    ClusterType const ClusterType::Conv = "Conv";
    ClusterType const ClusterType::ConvEpilogue = "ConvEpilogue";

    std::string ClusterType::to_string() const {
        return this->name;
    }

    bool ClusterType::operator== (ClusterType const &rhs) const {
        return this->name == rhs.name;
    }

    bool ClusterType::operator!= (ClusterType const &rhs) const {
        return this->name != rhs.name;
    }
}
}
}