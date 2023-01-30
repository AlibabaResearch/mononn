#include "mononn_engine/core/gpu/synchronization.h"

namespace mononn_engine {
namespace core {
namespace gpu {
Synchronization const Synchronization::None = Synchronization("None", 0);
Synchronization const Synchronization::Warp =
    Synchronization("__syncwarp(0x7ffffffff)", 1);
Synchronization const Synchronization::ThreadBlock =
    Synchronization("__syncthreads()", 2);
Synchronization const Synchronization::Global =
    Synchronization("synchronization::grid_sync()", 3);

std::string Synchronization::to_string() const { return this->type; }

bool Synchronization::operator<(const Synchronization& rhs) const {
  return this->synchronization_level < rhs.synchronization_level;
}

bool Synchronization::operator>(const Synchronization& rhs) const {
  return this->synchronization_level > rhs.synchronization_level;
}

bool Synchronization::operator!=(const Synchronization& rhs) const {
  return this->type != rhs.type ||
         this->synchronization_level != rhs.synchronization_level;
}

bool Synchronization::operator==(const Synchronization& rhs) const {
  return this->type == rhs.type &&
         this->synchronization_level == rhs.synchronization_level;
}

std::string Synchronization::get_prerequisite_definition() {
  return R"(
namespace synchronization {

__device__ uint32_t grid_barrier[4096] = {0};

__device__ __forceinline__ void grid_sync() {
  // Threadfence and syncthreads to make sure global writes are visible before
  // thread-0 reports in with its sync counter
  //__threadfence();
  //__syncthreads();

  if(blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      grid_barrier[blockIdx.x] = 1;
    }
    //__syncthreads();
    __threadfence();

    // Wait for everyone else to report in.
    // Each thread corresponds to 1 or more peer blocks.
    for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
      while (atomicAdd(&grid_barrier[peer_block], 0) == 0) {
        //__threadfence_block();
      }
    }
    __syncthreads();
    for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
     //atomicExch(&grid_barrier[peer_block], 0);
     grid_barrier[peer_block] = 0;
    }
  } else {
    if (threadIdx.x == 0) {
      // Report in
      grid_barrier[blockIdx.x] = 1;
      //atomicExch(&grid_barrier[blockIdx.x], 1);
      __threadfence();

      // Wait for acknowledgment
      while (atomicAdd(&grid_barrier[blockIdx.x], 0) == 1) {
        //__threadfence_block();
      }
    }

   // __syncthreads();
  }

  //__threadfence();
  __syncthreads();
}
}
)";
}
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine