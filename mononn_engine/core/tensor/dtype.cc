#include "mononn_engine/core/tensor/dtype.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/helpers/string_helpers.h"

#define USE_ALIGNED_ARRAY
namespace mononn_engine {
namespace core {
namespace tensor {
    Dtype const Dtype::BOOL = Dtype::from_string("bool");
    Dtype const Dtype::INT8 = Dtype::from_string("int8");
    Dtype const Dtype::INT16 = Dtype::from_string("int16");
    Dtype const Dtype::INT32 = Dtype::from_string("int32");
    Dtype const Dtype::INT64 = Dtype::from_string("int64");
    Dtype const Dtype::UINT8 = Dtype::from_string("uint8");
    Dtype const Dtype::UINT16 = Dtype::from_string("uint16");
    Dtype const Dtype::UINT32 = Dtype::from_string("uint32");
    Dtype const Dtype::UINT64 = Dtype::from_string("uint64");
    Dtype const Dtype::FLOAT16 = Dtype::from_string("float16");
    Dtype const Dtype::FLOAT32 = Dtype::from_string("float32");
    Dtype const Dtype::FLOAT64 = Dtype::from_string("float64");

    Dtype Dtype::from_string(std::string str) {
        if (str.compare("bool") == 0) return {std::string("bool"), 1};

        if (str.compare("int8") == 0) return {std::string("int8_t"), 1};
        if (str.compare("int16") == 0) return {std::string("int16_t"), 2};
        if (str.compare("int32") == 0) return {std::string("int32_t"), 4};
        if (str.compare("int64") == 0) return {std::string("int64_t"), 8};

        if (str.compare("uint8") == 0) return {std::string("uint8_t"), 1};
        if (str.compare("uint16") == 0) return {std::string("uint16_t"), 2};
        if (str.compare("uint32") == 0) return {std::string("uint32_t"), 4};
        if (str.compare("uint64") == 0) return {std::string("uint64_t"), 8};

        if (str.compare("float16") == 0) return {std::string("half"), 2};
        if (str.compare("float32") == 0) return {std::string("float"), 4};
        if (str.compare("float64") == 0) return {std::string("double"), 8};

        if (str.compare("tf32") == 0) return {std::string("double"), 8};
        if (str.compare("bfloat16") == 0) return {std::string("double"), 8};

        EXPECT_TRUE(false, "undefined data type: " + str);
    }

    Dtype Dtype::from_string(const char *str) {
        return Dtype::from_string(std::string(str));
    }

    bool Dtype::is_vectorized() const {
        return this->vectorized;
    }

    Dtype Dtype::vectorize(int N) const {
        return Dtype(
            this->type,
            this->get_elements_per_access() * N,
            this->bytes * N,
            this->vectorized || N > 1 // already vectorized or N > 1,
        );
    }

    Dtype Dtype::vectorize_to_bits(int bits) const {
        if (bits % this->size_in_bits() != 0) LOG(FATAL) << "Cannot vectorize: " << this->to_string() << " to " << bits << " bits";

        int factor = bits / this->size_in_bits();

        return this->vectorize(factor);
    }

    std::string Dtype::to_string() const {
        if (this->vectorized) {
            return mononn_engine::helpers::string_format("cutlass::AlignedArray<%s, %d>",
                            this->type.c_str(), this->elements_per_access);
        } else {
            if (this->elements_per_access != 1) LOG(FATAL) << "Unvectorized element per access: " << this->elements_per_access;

            return this->type;
        }
    }

    Dtype Dtype::get_pointer_type() const {
        return Dtype(this->to_string() + " *", 8);
    }

    Dtype Dtype::get_primitive_type() const {
        return Dtype(
            this->type,
            1,
            this->bytes / this->elements_per_access,
            false
        );
    }

    int Dtype::get_elements_per_access() const {
        return this->elements_per_access;
    }

    int Dtype::size_in_bytes() const {
        return this->bytes;
    }

    int Dtype::size_in_bits() const {
        return this->bytes * 8;
    }

    Dtype Dtype::to_cutlass_type() const {
        return Dtype(
            this->type == "half" ? "cutlass::half_t" : this->type,
            this->elements_per_access,
            this->bytes,
            this->vectorized
        );
    }

//    int Dtype::get_instruction_parallel_factor() const {
//        return this->ilp_factor;
//    }
//
//    bool Dtype::is_instruction_parallelized() const {
//        return this->ilp_factor != 1;
//    }
//
//    Dtype Dtype::instruction_parallelize(int _ilp_factor) {
//        return Dtype(
//            this->type,
//            this->elements_per_access,
//            this->bytes,
//            this->vectorized,
//            _ilp_factor
//        );
//    }

    bool Dtype::operator == (const Dtype &rhs) const {
        return this->type == rhs.type &&
        this->get_elements_per_access() == rhs.elements_per_access &&
        this->bytes == rhs.bytes &&
        this->vectorized == rhs.vectorized;
    }

    bool Dtype::operator != (const Dtype &rhs) const {
        return !(*this == rhs);
    }
}
}
}