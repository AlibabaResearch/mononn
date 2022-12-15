#include <sstream>
#include "mononn_engine/core/tensor/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace tensor {
    int TensorShape::rank() const {
        return this->shape.size();
    }

    int TensorShape::get_shape(int index) const {
        if (index < 0) index = int(this->shape.size()) + index;
        if (index < 0 || index > this->rank()) LOG(FATAL) << "Index " << index << " out of range";
        return this->shape[index];
    }

    void TensorShape::set_shape(int index, int _shape) {
        if (index < 0) index = int(this->shape.size()) + index;
        if (index < 0 || index > this->rank()) LOG(FATAL) << "Index " << index << " out of range";
        this->shape[index] = _shape;
    }

    const std::vector<int>& TensorShape::get_shape() const {
        return this->shape;
    }

    int TensorShape::element_count() const {
        int ret = 1;
        for (int s : this->shape) {
            ret *= s;
        }

        return ret;
    }

    TensorShape TensorShape::flatten() const {
        int size = 1;
        for (int s : this->shape) size *= s;

        return std::vector<int>{size};
    }

    TensorShape TensorShape::concat(const TensorShape &rhs) const {
        std::vector<int> shape_ret = this->get_shape();
        std::vector<int> shape_rhs = rhs.get_shape();

        shape_ret.insert(shape_ret.end(), shape_rhs.begin(), shape_rhs.end());
        return shape_ret;
    }

    TensorShape TensorShape::concat_on_dim(const TensorShape &rhs, int dim) const {
        if (this->rank() != rhs.rank()) LOG(FATAL) << "Two shape have different rank";

        std::vector<int> new_shape;

        for (int idx = 0; idx < this->rank(); ++idx) {
            if (this->get_shape(idx) != rhs.get_shape(idx) && idx != dim) LOG(FATAL) << "Unable to concat_dim two shapes";

            if (idx == dim) {
                new_shape.push_back(this->get_shape(idx) + rhs.get_shape(idx));
            } else {
                new_shape.push_back(this->get_shape(idx));
            }
        }

        return TensorShape(new_shape);
    }

    TensorShape TensorShape::reduce_dim(int index) const {
        if (index < 0) index = int(this->shape.size()) + index;

        if (index < 0 || index > this->rank()) LOG(FATAL) << "Index " << index << " out of range";
        
        std::vector<int> new_shape = this->get_shape();
        new_shape[index] = 1;
        return TensorShape(new_shape);
    }

    TensorShape TensorShape::slice_dim(int start, int end) const {
        if (start < 0) start = int(this->shape.size()) + start;
        if (end < 0) end = int(this->shape.size()) + end + 1;

        if (start < 0 || start > this->rank()) LOG(FATAL) << "Index " << start << " out of range";
        if (end < 0 || end > this->rank()) LOG(FATAL) << "Index " << end << " out of range";

        std::vector<int> new_shape;

        for (int idx = start; idx < end; ++idx) {
            new_shape.push_back(this->shape[idx]);
        }

        return TensorShape(new_shape);
    }

    TensorShape TensorShape::reshape(std::vector<int> to_shape) const {
        if (!this->can_reshape_to(to_shape)) LOG(FATAL) << "Cannot reshape " << this->to_string() << " to " << TensorShape(to_shape).to_string();

        return {to_shape};
    }

    bool TensorShape::can_reshape_to(std::vector<int> to_shape) const {
        int idx1 = 0, idx2 = 0;
        
        int acc1, acc2;

        while (idx1 < (int)this->shape.size() && idx2 < (int)to_shape.size()) {
            acc1 = this->shape[idx1];
            acc2 = to_shape[idx2];
            
            if (acc1 == acc2) {
                ++idx1;
                ++idx2;
                continue;
            }

            if (acc1 < acc2) {
                while (acc1 < acc2) {
                    if (idx1 + 1 >= (int)this->shape.size()) return false;
                    acc1 *= this->shape[++idx1];
                }

                if (acc1 != acc2) return false;

                ++idx1;
                ++idx2;
                continue;
            }

            if (acc1 > acc2) {
                while (acc1 > acc2) {
                    if (idx2 + 1 >= (int)to_shape.size()) return false;

                    acc2 *= to_shape[++idx2];
                }

                if (acc1 != acc2) return false;

                ++idx1;
                ++idx2;

                continue;
            }
        }
        
        return idx1 == (int)this->shape.size() && idx2 == (int)to_shape.size();
    }

    TensorShape TensorShape::permute(std::vector<int> perm) const {
        int mask = 0;
        for (auto const &p : perm) mask |= (1 << p);

        if (mask != ((1 << perm.size()) - 1)) LOG(FATAL) << "Input is not permutation " << mononn_engine::helpers::to_string(perm);

        std::vector<int> new_shape;

        for (auto const &p : perm) new_shape.push_back(this->shape[p]);

        return { new_shape };
    }

    std::string TensorShape::to_string() const {
        std::stringstream ss;
        ss << "[";

        for (int idx = 0; idx < this->rank(); ++idx) {
            if (idx == 0) {
                ss << std::to_string(this->get_shape(idx));
            } else {
                ss << ",";
                ss << std::to_string(this->get_shape(idx));
            }
        }

        ss << "]";
        return ss.str();
    }

    bool TensorShape::is_scalar() const {
        return (int)this->shape.size() == 1 && this->shape[0] == 1;
    }

    bool TensorShape::operator == (const TensorShape& rhs) const {
        return this->shape == rhs.shape;
    }

    bool TensorShape::operator != (const TensorShape& rhs) const {
        return this->shape != rhs.shape;
    }
}
}
}