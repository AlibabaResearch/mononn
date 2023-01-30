
#include "mononn_engine/core/tensor/memory_layout.h"

#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace tensor {
bool MemoryLayout::valid() const {
  unsigned int mask = 0;
  for (int p : this->perm) {
    mask |= (1 << p);
  }

  return mask == ((1 << this->rank()) - 1);
}

int MemoryLayout::rank() const { return this->perm.size(); }

std::vector<int> MemoryLayout::get() const { return this->perm; }

int MemoryLayout::get(int index) const {
  if (index < 0) index = int(this->perm.size()) + index;
  if (index < 0 || index > this->rank())
    LOG(FATAL) << "Index " << index << " out of range";

  return this->perm[index];
}

std::vector<int> MemoryLayout::get_layout() const { return this->get(); }

int MemoryLayout::get_layout(int index) const { return this->get(index); }

MemoryLayout MemoryLayout::flatten() const { return {std::vector<int>{0}}; }

MemoryLayout MemoryLayout::concat(const MemoryLayout& rhs) const {
  std::vector<int> layout_ret = this->get();
  std::vector<int> layout_rhs = rhs.get();

  for (int idx = 0; idx < (int)layout_ret.size(); ++idx) {
    layout_ret[idx] += rhs.rank();
  }

  layout_ret.insert(layout_ret.end(), layout_rhs.begin(), layout_rhs.end());

  MemoryLayout(layout_ret).assert_layout_valid();

  return {layout_ret};
}

MemoryLayout MemoryLayout::reduce_dim(int index) const {
  std::vector<int> layout_ret = this->get();
  if (index < 0) index += this->rank();

  if (index < 0 || index > this->rank())
    LOG(FATAL) << "Index " << index << " out of range";

  layout_ret.erase(layout_ret.begin() + index);

  MemoryLayout(layout_ret).normalize().assert_layout_valid();

  return MemoryLayout(layout_ret).normalize();
}

MemoryLayout MemoryLayout::slice_dim(int start, int end) const {
  if (start < 0) start = int(this->perm.size()) + start;
  if (end < 0) end = int(this->perm.size()) + end + 1;

  std::vector<int> new_perm;

  for (int idx = start; idx < end; ++idx) {
    new_perm.push_back(this->perm[idx]);
  }

  MemoryLayout(new_perm).normalize().assert_layout_valid();

  return MemoryLayout(new_perm).normalize();
}

MemoryLayout MemoryLayout::permute(std::vector<int> _perm) const {
  int mask = 0;
  for (auto const& p : _perm) mask |= (1 << p);

  if (mask != ((1 << _perm.size()) - 1))
    LOG(FATAL) << "Input is not permutation "
               << mononn_engine::helpers::to_string(_perm);

  std::vector<int> new_layout;

  for (auto const& p : _perm) new_layout.push_back(this->perm[p]);

  return {new_layout};
}

std::string MemoryLayout::to_string() const {
  std::stringstream ss;
  ss << "{";

  for (int idx = 0; idx < this->rank(); ++idx) {
    if (idx == 0) {
      ss << std::to_string(this->get(idx));
    } else {
      ss << ",";
      ss << std::to_string(this->get(idx));
    }
  }

  ss << "}";
  return ss.str();
}

// normalize layout to [0, rank)
MemoryLayout MemoryLayout::normalize() const {
  std::vector<int> layout_ret;

  for (int idx = 0; idx < this->rank(); ++idx) {
    int r = 0;
    for (auto const& n : this->perm) {
      if (n < this->perm[idx]) {
        ++r;
      }
    }

    layout_ret.push_back(r);
  }

  return {layout_ret};
}

bool MemoryLayout::operator==(const MemoryLayout& rhs) const {
  return this->perm == rhs.perm;
}

void MemoryLayout::assert_layout_valid() const {
  if (!this->valid()) LOG(FATAL) << "Invalid layout " << this->to_string();
}
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine