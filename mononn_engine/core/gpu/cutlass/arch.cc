#include "mononn_engine/core/gpu/cutlass/arch.h"
#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    Arch const Arch::Sm70 = Arch("cutlass::arch::Sm70");
    Arch const Arch::Sm75 = Arch("cutlass::arch::Sm75");
    Arch const Arch::Sm80 = Arch("cutlass::arch::Sm80");
    Arch const Arch::Sm86 = Arch("cutlass::arch::Sm86");
    Arch const Arch::OpClassSimt = Arch("cutlass::arch::OpClassSimt");
    Arch const Arch::OpClassTensorOp = Arch("cutlass::arch::OpClassTensorOp");
    Arch const Arch::OpMultiplyAdd = Arch("cutlass::arch::OpMultiplyAdd");

    bool Arch::newer_or_equal(const Arch &a, const Arch &b) {
        if (!Arch::is_sm_architecture(a) || !Arch::is_sm_architecture(b)) LOG(FATAL) << "Invalid";
        int code_a = std::stoi(a.to_string().substr(17, 2));
        int code_b = std::stoi(b.to_string().substr(17, 2));

        return code_a >= code_b;
    }

    bool Arch::is_sm_architecture(const Arch &arch) {
        return arch == Arch::Sm70 || arch == Arch::Sm75 || arch == Arch::Sm80 || arch == Arch::Sm86;
    }

    std::string Arch::to_string() const {
        return this->name;
    }

    bool Arch::operator== (Arch const &rhs) const {
        return this->name == rhs.name;
    }

    std::unique_ptr<tensorflow::mononn_extra::proto::Arch> Arch::ConvertToProto() const {
        std::unique_ptr<tensorflow::mononn_extra::proto::Arch> arch = std::make_unique<tensorflow::mononn_extra::proto::Arch>();

        arch->set_name(this->name);

        return std::move(arch);
    }

    void Arch::ParseFromProto(const tensorflow::mononn_extra::proto::Arch *arch) {
        this->name = arch->name();
    }
}
}
}
}