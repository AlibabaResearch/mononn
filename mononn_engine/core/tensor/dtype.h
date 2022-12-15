#pragma once 

#include <vector>
#include <string>

namespace mononn_engine {
namespace core {
namespace tensor {
    class Dtype {
    public:
        Dtype() = default;
        Dtype(std::string _type, int _bytes) : 
            type(_type), elements_per_access(1), bytes(_bytes), vectorized(false) {}
        Dtype(std::string _type, int _elements_per_access, int _bytes, bool _vectorized) :
            type(_type), elements_per_access(_elements_per_access), bytes(_bytes), vectorized(_vectorized) {}

        bool is_vectorized() const;
        Dtype vectorize(int N) const;
        Dtype vectorize_to_bits(int bits) const;
        std::string to_string() const;
        Dtype get_pointer_type() const;
        Dtype get_primitive_type() const;

        int get_elements_per_access() const;
        int size_in_bytes() const;
        int size_in_bits() const;

        Dtype to_cutlass_type() const;

        bool operator == (const Dtype &rhs) const;
        bool operator != (const Dtype &rhs) const;

        static Dtype from_string(std::string str);
        static Dtype from_string(const char *str);

//        int get_instruction_parallel_factor() const;
//        bool is_instruction_parallelized() const;
//        Dtype instruction_parallelize(int _ilp_factor);

        static Dtype const BOOL;
        static Dtype const INT8;
        static Dtype const INT16;
        static Dtype const INT32;
        static Dtype const INT64;
        static Dtype const UINT8;
        static Dtype const UINT16;
        static Dtype const UINT32;
        static Dtype const UINT64;
        static Dtype const FLOAT16;
        static Dtype const FLOAT32;
        static Dtype const FLOAT64;

    private:
        std::string type;
        int elements_per_access = 1;
        int bytes;
        bool vectorized;

//        int ilp_factor = 1;
    };
}
}
}