#pragma once 

#include <utility>
#include <vector>
#include <string>

namespace mononn_engine {
namespace core {
namespace tensor {
    class TensorShape {
    public:
        TensorShape() = default;
        TensorShape(std::vector<int> _shape) : shape(std::move(_shape)) {}
        
        int rank() const;
        int get_shape(int index) const;
        void set_shape(int index, int _shape);
        const std::vector<int>& get_shape() const;
        int element_count() const;

        TensorShape flatten() const;
        TensorShape concat(const TensorShape &rhs) const;
        TensorShape concat_on_dim(const TensorShape &rhs, int dim) const;
        TensorShape reduce_dim(int index) const;
        TensorShape slice_dim(int start, int end) const;

        TensorShape reshape(std::vector<int> to_shape) const;
        bool can_reshape_to(std::vector<int> to_shape) const;

        TensorShape permute(std::vector<int> perm) const;

        std::string to_string() const;

        bool is_scalar() const;

        bool operator == (const TensorShape& rhs) const;
        bool operator != (const TensorShape& rhs) const;

    private:
        std::vector<int> shape;
    };
}
}
}