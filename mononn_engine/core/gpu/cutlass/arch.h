#pragma once 

#include <memory>
#include <string>
#include "tensorflow/mononn_extra/proto/arch.pb.h"
#include "mononn_engine/core/common/proto_converter.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    class Arch : public mononn_engine::core::common::ProtoConverter<tensorflow::mononn_extra::proto::Arch> {
    public:
        Arch() {}
        explicit Arch(std::string _name) : name(_name) {}
        explicit Arch(const char * _name) : Arch(std::string(_name)) {}

        static Arch const Sm70;
        static Arch const Sm75;
        static Arch const Sm80;
        static Arch const Sm86;
        static Arch const OpClassSimt;
        static Arch const OpClassTensorOp;
        static Arch const OpMultiplyAdd;

        static bool newer_or_equal(const Arch &a, const Arch &b);
        static bool is_sm_architecture(const Arch &arch);

        std::string to_string() const;

        bool operator == (Arch const &rhs) const;

        std::unique_ptr<tensorflow::mononn_extra::proto::Arch> ConvertToProto() const override;
        void ParseFromProto(tensorflow::mononn_extra::proto::Arch const *arch) override;

    private:
        std::string name;
    };
}
}
}
}