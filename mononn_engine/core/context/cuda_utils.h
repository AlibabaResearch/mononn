#pragma once

namespace mononn_engine {
namespace core {
namespace context {
int get_max_smem_size_per_block(int desired_block_count_per_sm, int block_size,
                                int max_configurable_smem,
                                int reserved_smem_per_block);
}
}  // namespace core
}  // namespace mononn_engine