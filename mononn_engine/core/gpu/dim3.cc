#include "mononn_engine/core/gpu/dim3.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
    int Dim3::XYZ() const {
        return this->x * this->y * this->z;
    }

    std::string Dim3::to_string() const {
        return mononn_engine::helpers::string_format("(%d, %d, %d)", this->x, this->y, this->z);
    }

    std::unique_ptr<tensorflow::mononn_extra::proto::Dim3> Dim3::ConvertToProto() const {
        std::unique_ptr<tensorflow::mononn_extra::proto::Dim3> dim3 = std::make_unique<tensorflow::mononn_extra::proto::Dim3>();

        dim3->set_x(this->x);
        dim3->set_y(this->y);
        dim3->set_z(this->z);

        return std::move(dim3);
    }

    void Dim3::ParseFromProto(const tensorflow::mononn_extra::proto::Dim3 *dim3) {
        this->x = dim3->x();
        this->y = dim3->y();
        this->z = dim3->z();
    }
}
}
}