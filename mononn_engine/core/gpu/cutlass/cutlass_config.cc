#include "mononn_engine/core/gpu/cutlass/cutlass_config.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    std::string CutlassConfig::to_string() const {
        return mononn_engine::helpers::string_format("ThreadBlockShape<%d,%d,%d>, WarpShape<%d,%d,%d>, InstShape<%d, %d, %d>, Arch %s, stages %d",
                                                this->ThreadBlockShape.m(),
                                                this->ThreadBlockShape.n(),
                                                this->ThreadBlockShape.k(),
                                                this->WarpShape.m(),
                                                this->WarpShape.n(),
                                                this->WarpShape.k(),
                                                this->InstructionShape.m(),
                                                this->InstructionShape.n(),
                                                this->InstructionShape.k(),
                                                this->ArchTag.to_string().c_str(),
                                                this->stages);
    }

    std::unique_ptr<tensorflow::mononn_extra::proto::CutlassConfig> CutlassConfig::ConvertToProto() const {
        std::unique_ptr<tensorflow::mononn_extra::proto::CutlassConfig> cutlass_config = std::make_unique<tensorflow::mononn_extra::proto::CutlassConfig>();
        cutlass_config->set_allocated_threadblockshape(this->ThreadBlockShape.ConvertToProto().release());
        cutlass_config->set_allocated_warpshape(this->WarpShape.ConvertToProto().release());
        cutlass_config->set_allocated_instructionshape(this->InstructionShape.ConvertToProto().release());
        cutlass_config->set_allocated_operatorclass(this->OperatorClass.ConvertToProto().release());
        cutlass_config->set_allocated_archtag(this->ArchTag.ConvertToProto().release());
        cutlass_config->set_stages(this->stages);

        return std::move(cutlass_config);
    }

    void CutlassConfig::ParseFromProto(const tensorflow::mononn_extra::proto::CutlassConfig *cutlass_config) {
        this->ThreadBlockShape.ParseFromProto(&cutlass_config->threadblockshape());
        this->WarpShape.ParseFromProto(&cutlass_config->warpshape());
        this->InstructionShape.ParseFromProto(&cutlass_config->instructionshape());
        this->OperatorClass.ParseFromProto(&cutlass_config->operatorclass());
        this->ArchTag.ParseFromProto(&cutlass_config->archtag());
        this->stages = cutlass_config->stages();
    }
}
}
}
}