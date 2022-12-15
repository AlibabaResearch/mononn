#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include "third_party/gpus/cuda/include/cuda_device_runtime_api.h"

#include "mononn_engine/core/context/cuda_device_context.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/helpers/string_helpers.h"


namespace mononn_engine {
namespace core {
namespace context {
    static const char *_cudaGetErrorEnum(cudaError_t error) {
        return cudaGetErrorName(error);
    }

    template <typename T>
    void check(T result, char const *const func, const char *const file,
            int const line) {
        if (result) {
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                    static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
            exit(EXIT_FAILURE);
        }
    }

    #define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

    std::string CUDADeviceContext::to_cuda_arch(int major, int minor) {
        return mononn_engine::helpers::string_format("%d", major * 100 + minor * 10);
    }

    CUDADeviceContext CUDADeviceContext::get_cuda_device_context() {
        static CUDADeviceContext *device_context = nullptr;

        if (device_context == nullptr) {
            device_context = new CUDADeviceContext;
            device_context->device_count = 0;

            checkCudaErrors(cudaGetDeviceCount(&(device_context->device_count)));
            
            if (device_context->device_count == 0) {
                LOG(FATAL) << "No visible CUDA device";
            }

            checkCudaErrors(cudaSetDevice(0));
            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

            device_context->sm_count = deviceProp.multiProcessorCount;
            device_context->static_smem_size = deviceProp.sharedMemPerBlock;
            device_context->max_configurable_smem_size = deviceProp.sharedMemPerBlockOptin;
            device_context->cuda_arch = CUDADeviceContext::to_cuda_arch(deviceProp.major, deviceProp.minor);
            device_context->register_per_block = deviceProp.regsPerBlock;
            device_context->warp_size = deviceProp.warpSize;

            if (std::find(CUDADeviceContext::supported_cuda_arch.begin(), 
                CUDADeviceContext::supported_cuda_arch.end(), 
                device_context->cuda_arch) == CUDADeviceContext::supported_cuda_arch.end()) {
                LOG(FATAL) << "Unsupported compute capability: " << device_context->cuda_arch;
            }
        }

        return *device_context;
    }

    std::string CUDADeviceContext::to_string() const {
        return mononn_engine::helpers::string_format(""
                                                "CUDA device context:\n"
                                                "\tdevice count %d,\n"
                                                "\tsm_count %d, \n"
                                                "\tstatic smem size %d,\n"
                                                "\tmax smem size %d,\n"
                                                "\tcuda arch %s,\n"
                                                "\tregister per block %d,\n"
                                                "\twarp size %d",
            this->device_count,
            this->sm_count,
            this->static_smem_size,
            this->max_configurable_smem_size,
            this->cuda_arch.c_str(),
            this->register_per_block,
            this->warp_size);
    }

    cutlass::Arch CUDADeviceContext::get_cutlass_arch_tag() const {
        if (this->cuda_arch == "700") {
            return cutlass::Arch::Sm70;
        } else if (this->cuda_arch == "750") {
            return cutlass::Arch::Sm75;
        } else if (this->cuda_arch == "800") {
            return cutlass::Arch::Sm80;
        } else if (this->cuda_arch == "860") {
            return cutlass::Arch::Sm86;
        } else {
            LOG(FATAL) << "Unsupported arch " << this->cuda_arch;
        }
    }

    std::string CUDADeviceContext::get_cuda_arch_global_macro() const {
        return mononn_engine::helpers::string_format("#define __CUDA_ARCH_GLOBAL__ %s", this->cuda_arch.c_str());
    }

    std::unique_ptr<tensorflow::mononn_extra::proto::CUDADeviceContext> CUDADeviceContext::ConvertToProto() const {
        std::unique_ptr<tensorflow::mononn_extra::proto::CUDADeviceContext> cuda_device_context = std::make_unique<tensorflow::mononn_extra::proto::CUDADeviceContext>();

        cuda_device_context->set_device_count(this->device_count);
        cuda_device_context->set_sm_count(this->sm_count);
        cuda_device_context->set_static_smem_size(this->static_smem_size);
        cuda_device_context->set_max_configurable_smem_size(this->max_configurable_smem_size);
        cuda_device_context->set_cuda_arch(this->cuda_arch);
        cuda_device_context->set_register_per_block(this->register_per_block);
        cuda_device_context->set_warp_size(this->warp_size);

        return std::move(cuda_device_context);
    }

    void CUDADeviceContext::ParseFromProto(const tensorflow::mononn_extra::proto::CUDADeviceContext *cuda_device_context) {
        this->device_count = cuda_device_context->device_count();
        this->sm_count = cuda_device_context->sm_count();
        this->static_smem_size = cuda_device_context->static_smem_size();
        this->max_configurable_smem_size = cuda_device_context->max_configurable_smem_size();
        this->cuda_arch = cuda_device_context->cuda_arch();
        this->register_per_block = cuda_device_context->register_per_block();
        this->warp_size = cuda_device_context->warp_size();
    }

    std::vector<std::string> CUDADeviceContext::supported_cuda_arch = {"700", "750", "800", "860"};
}
}
}