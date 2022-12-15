#include "mononn_engine/core/op_impl/reduce_impl.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/core/schedule/loop.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/tensor/scalar.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "mononn_engine/core/gpu/defined.h"
#include "mononn_engine/core/tensor/tensor_shape.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/core/gpu/limits.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/config/config.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Dtype = mononn_engine::core::tensor::Dtype;
    using Loop = mononn_engine::core::schedule::Loop;
    using Scalar = mononn_engine::core::tensor::Scalar;
    using CUDADefined = mononn_engine::core::gpu::CUDADefined;
    using TensorShape = mononn_engine::core::tensor::TensorShape;
    using Tensor = mononn_engine::core::tensor::Tensor;
    using CUDAContext = mononn_engine::core::context::CUDAContext;
    using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
    using Functor = mononn_engine::core::gpu::Functor;
    using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
    using Limits = mononn_engine::core::gpu::Limits;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;
    using Config = mononn_engine::config::Config;
    using BufferManager = mononn_engine::core::gpu::BufferManager;

    ReduceImpl::ReduceImpl(std::shared_ptr<CUDAContext> _context, ReduceImpl::InputSpec _input_spec, Tensor _output) {
        this->context = _context;
        this->tier = _input_spec.tier;
        this->input_spec = _input_spec;
        this->output = _output;
        this->elements_per_access = this->input_spec.operands[0].get_dtype().get_elements_per_access();
        this->reduce_accum = _input_spec.reduce_accum;

        // this->reducer = ReduceImpl::get_reduce_functor(
        //         _input_spec.reducer,
        //         this->input_spec.operand.get_dtype());

        // std::string reduce_accum_init;
        
        // {
        //     std::vector<std::string> init_list;
        //     for (int param_idx = this->input_spec.operands.size() / 2; param_idx < this->input_spec.operands.size(); ++param_idx) {
        //         init_list.push_back(this->input_spec.operands[param_idx].get_name());
        //     }

        //     reduce_accum_init = mononn_engine::helpers::join(", ", init_list);

        //     if (init_list.size() > 1) {
        //         reduce_accum_init = "cuda::std::make_tuple(" + reduce_accum_init + ")";
        //     }
        // }
        

        // if (this->input_spec.reducer == Reducer::Max) {
        //     reduce_accum_init = Limits::get_min_negative(this->output.get_dtype().get_primitive_type());
        // } else if (this->input_spec.reducer == Reducer::Min) {
        //     reduce_accum_init = Limits::get_max_positive(this->output.get_dtype().get_primitive_type());
        // } else if (this->input_spec.reducer == Reducer::Sum) {
        //     if (this->output.get_dtype().get_primitive_type() == Dtype::FLOAT16) {
        //         reduce_accum_init = "half(0)";
        //     } else {
        //         reduce_accum_init = "0";
        //     }
        // }

        // std::vector<Dtype> init_type_list;
        // std::vector<std::string> init_value_list;

        // for (int param_idx = this->input_spec.operands.size() / 2; param_idx < this->input_spec.operands.size(); ++param_idx) {
        //     init_type_list.push_back(this->input_spec.operands[param_idx].get_dtype().get_primitive_type());
        //     init_value_list.push_back(this->input_spec.operands[param_idx].get_name());
        // }

        // this->reduce_accum = Scalar(this->output.get_name() + "_accum", init_type_list, init_value_list);

        if (this->tier == LocalityTier::kT0) {
            this->post_reduce_if_statement = mononn_engine::helpers::string_format("if (true) {\n");
        } else if (this->tier == LocalityTier::kT1) {
            this->post_reduce_if_statement = mononn_engine::helpers::string_format("if (lane_id == 0) {\n");
        } else if (this->tier == LocalityTier::kT2) {
            this->post_reduce_if_statement = mononn_engine::helpers::string_format("if (blockIdx.x == 0) {\n");
        } else {
            LOG(FATAL) << this->tier.to_string() << " is not supported";
        }
    }

    std::string ReduceImpl::get_post_reduce_if_statement() const {
        return this->post_reduce_if_statement;
    }

    std::string ReduceImpl::get_post_reduce_if_end() const {
        return "}";
    }

    int ReduceImpl::get_smem_usage_in_bytes() const {
        if (this->tier == LocalityTier::kT2) {
            int size_in_bytes;

            for (int operand_id = 0; operand_id < this->input_spec.operands.size(); ++operand_id) {
                size_in_bytes += this->input_spec.operands[operand_id].get_dtype().size_in_bytes();
            }

            return size_in_bytes * 32;
        } else if (this->tier == LocalityTier::kT1) {
            return 0; 
        } else {
            LOG(FATAL) << "Unsupported tier: " << this->tier.to_string();
        }
    }

    std::vector<std::shared_ptr<OpImplBase>> ReduceImpl::get_available_implementations(std::shared_ptr<CUDAContext> context, ReduceImpl::InputSpec input_spec, Tensor output) {
        Tier tier = input_spec.tier;

        
        // Functor reducer = ReduceImpl::get_reduce_functor(
        //     input_spec.reducer,
        //     input_spec.operand.get_dtype().get_primitive_type());

        if (tier == LocalityTier::kT1) {
            std::shared_ptr<ReduceImpl> impl_warp_custom_reduce = std::make_shared<ReduceImpl>(context, input_spec, output);
            std::shared_ptr<ReduceImpl> impl_warp_reduce_cub = std::make_shared<ReduceImpl>(context, input_spec, output);

            Scalar reduce_accum = impl_warp_custom_reduce->get_reduce_accum();

            FunctionInvocation warp_custom_reduce("warp_custom_reduce");
            // warp_custom_reduce.add_template_arg(input_spec.operand.get_dtype().get_primitive_type().to_string());
            warp_custom_reduce.add_template_arg(input_spec.reduce_accum.get_type_string());
            // warp_custom_reduce.add_template_arg(reducer.get_functor_type());
            warp_custom_reduce.add_template_arg(input_spec.reduction_functor_generator->type_name());
            warp_custom_reduce.add_template_arg(std::to_string(context->cuda_device_context.warp_size));
            warp_custom_reduce.add_template_arg("1");

            FunctionInvocation warp_reduce_cub("warp_reduce_cub");

            // warp_reduce_cub.add_template_arg(input_spec.operand.get_dtype().get_primitive_type().to_string());
            warp_reduce_cub.add_template_arg(input_spec.reduce_accum.get_type_string());
            // warp_reduce_cub.add_template_arg(reducer.get_functor_type());
            warp_reduce_cub.add_template_arg(input_spec.reduction_functor_generator->type_name());
            warp_reduce_cub.add_template_arg(std::to_string(context->cuda_device_context.warp_size));
            warp_reduce_cub.add_template_arg("1");

            // bool vectorized_mem_access = input_spec.operands[0].get_dtype().is_vectorized();

            // if (vectorized_mem_access) {
            //     FunctionInvocation vector_to_scalar("to_scalar");
            //     vector_to_scalar.add_template_arg(input_spec.reduce_accum.get_type_string());
            //     vector_to_scalar.add_template_arg(std::to_string(input_spec.operands[0].get_dtype().get_elements_per_access()));
            //     vector_to_scalar.add_template_arg(input_spec.reduction_functor_generator->type_name());
            //     vector_to_scalar.add_arg(reduce_accum.get_name());

            //     warp_custom_reduce.add_arg(vector_to_scalar.to_string());
            //     warp_reduce_cub.add_arg(vector_to_scalar.to_string());
            // } else {
            warp_custom_reduce.add_arg(reduce_accum.get_name());
            warp_reduce_cub.add_arg(reduce_accum.get_name());
            // }

            impl_warp_custom_reduce->set_invocation(warp_custom_reduce);
            impl_warp_custom_reduce->set_attribute("name", "warp_custom_reduce");
            impl_warp_reduce_cub->set_invocation(warp_reduce_cub);
            impl_warp_reduce_cub->set_attribute("name", "warp_reduce_cub");

            return {
                std::static_pointer_cast<OpImplBase>(impl_warp_custom_reduce), 
                std::static_pointer_cast<OpImplBase>(impl_warp_reduce_cub)
            };
        }

        if (tier == LocalityTier::kT2) {
            std::shared_ptr<ReduceImpl> impl_block_custom_reduce = std::make_shared<ReduceImpl>(context, input_spec, output);
            std::shared_ptr<ReduceImpl> impl_block_reduce_cub_RAKING = std::make_shared<ReduceImpl>(context, input_spec, output);
            std::shared_ptr<ReduceImpl> impl_block_reduce_cub_RAKING_COMMUTATIVE_ONLY = std::make_shared<ReduceImpl>(context, input_spec, output);
            std::shared_ptr<ReduceImpl> impl_block_reduce_cub_WARP_REDUCTIONS = std::make_shared<ReduceImpl>(context, input_spec, output);

            FunctionInvocation block_custom_reduce("block_custom_reduce");
            FunctionInvocation block_reduce_cub_RAKING("block_reduce_cub_RAKING");
            FunctionInvocation block_reduce_cub_RAKING_COMMUTATIVE_ONLY("block_reduce_cub_RAKING_COMMUTATIVE_ONLY");
            FunctionInvocation block_reduce_cub_WARP_REDUCTIONS("block_reduce_cub_WARP_REDUCTIONS");

            Scalar reduce_accum = impl_block_custom_reduce->get_reduce_accum();

            std::vector<FunctionInvocation> invocations = {
                block_custom_reduce, 
                block_reduce_cub_RAKING, 
                block_reduce_cub_RAKING_COMMUTATIVE_ONLY, 
                block_reduce_cub_WARP_REDUCTIONS
            };
            
            for (auto &reduce_invocation : invocations) {
                reduce_invocation.add_template_arg(input_spec.reduce_accum.get_type_string());
                reduce_invocation.add_template_arg(input_spec.reduction_functor_generator->type_name());
                reduce_invocation.add_template_arg(std::to_string(context->cuda_runtime_context.block_dim.x));
                reduce_invocation.add_template_arg("1");
                // bool vectorized_mem_access = input_spec.operands[0].get_dtype().is_vectorized();

                // if (vectorized_mem_access) {
                //     FunctionInvocation vector_to_scalar("to_scalar");

                //     vector_to_scalar.add_template_arg(input_spec.reduce_accum.get_type_string());
                //     vector_to_scalar.add_template_arg(std::to_string(input_spec.operands[0].get_dtype().get_elements_per_access()));
                //     vector_to_scalar.add_template_arg(input_spec.reduction_functor_generator->type_name());
                //     vector_to_scalar.add_arg(reduce_accum.get_name());

                //     reduce_invocation.add_arg(vector_to_scalar.to_string());
                // } else {
                reduce_invocation.add_arg(reduce_accum.get_name());
                // }

                reduce_invocation.add_arg(Config::get()->smem_reduction_cache_name);
            }

            impl_block_custom_reduce->set_invocation(invocations[0]);
            impl_block_custom_reduce->set_attribute("name", "block_custom_reduce");

            impl_block_reduce_cub_RAKING->set_invocation(invocations[1]);
            impl_block_reduce_cub_RAKING->set_attribute("name", "block_reduce_cub_RAKING");

            impl_block_reduce_cub_RAKING_COMMUTATIVE_ONLY->set_invocation(invocations[2]);
            impl_block_reduce_cub_RAKING_COMMUTATIVE_ONLY->set_attribute("name", "block_reduce_cub_RAKING_COMMUTATIVE_ONLY");
            
            impl_block_reduce_cub_WARP_REDUCTIONS->set_invocation(invocations[3]);
            impl_block_reduce_cub_WARP_REDUCTIONS->set_attribute("name", "block_reduce_cub_WARP_REDUCTIONS");

            return {
                std::static_pointer_cast<OpImplBase>(impl_block_custom_reduce), 
                std::static_pointer_cast<OpImplBase>(impl_block_reduce_cub_RAKING), 
                std::static_pointer_cast<OpImplBase>(impl_block_reduce_cub_RAKING_COMMUTATIVE_ONLY), 
                std::static_pointer_cast<OpImplBase>(impl_block_reduce_cub_WARP_REDUCTIONS)
            };
        }

        LOG(FATAL) << "Unexpected locality tier: " << tier.to_string();
    }
    
    ReduceImpl::Tier ReduceImpl::get_tier() const {
        return this->tier;
    }

    std::string ReduceImpl::generate_impl() const {
        std::stringstream code_stream;

        FunctionInvocation vector_reduce_invocation(this->input_spec.reduction_functor_generator->instance_name());
        vector_reduce_invocation.add_arg(this->reduce_accum.get_name());

        auto to_tuple_name_if_needed = [] (const std::vector<std::string> &name_list) -> std::string {
            if (name_list.size() == 1) {
                return name_list[0];
            }

            return "tuple_util::make_tuple(" + mononn_engine::helpers::join(", ", name_list) + ")";
        };

        if (this->input_spec.operands[0].get_dtype().is_vectorized() || this->is_instruction_parallelized()) {
            FunctionInvocation to_scalar("to_scalar");
            auto output_primitive_type = this->output.get_dtype().get_primitive_type();
            to_scalar.add_template_arg(this->reduce_accum.get_type_string());

            if (this->input_spec.operands[0].get_dtype().get_elements_per_access() != 1) {
                to_scalar.add_template_arg(std::to_string(this->input_spec.operands[0].get_dtype().get_elements_per_access()));
            }

            to_scalar.add_template_arg(this->input_spec.reduction_functor_generator->type_name());

            // if (this->input_spec.reducer == Reducer::Max) {
            //     to_scalar.add_template_arg(mononn_engine::helpers::string_format("cutlass::maximum<%s>", output_primitive_type.to_string().c_str()));
            // } else if (this->input_spec.reducer == Reducer::Min) {
            //     to_scalar.add_template_arg(mononn_engine::helpers::string_format("minimum::maximum<%s>", output_primitive_type.to_string().c_str()));
            // } else if (this->input_spec.reducer == Reducer::Sum) {
            //     to_scalar.add_template_arg(mononn_engine::helpers::string_format("cutlass::plus<%s>", output_primitive_type.to_string().c_str()));
            // } else {
            //     LOG(FATAL) << "Unsupported reducer " << (int)this->input_spec.reducer;
            // }

            if (this->is_instruction_parallelized()) {
                for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                    std::vector<std::string> name_list;
                    for (int operand_id = 0; operand_id < (int)this->input_spec.operands.size(); ++operand_id) {
                        name_list.push_back(mononn_engine::helpers::get_node_ilp_name(this->input_spec.operands[operand_id].get_name(), ilp_id));
                    }

                    to_scalar.add_arg(to_tuple_name_if_needed(name_list));
                }
            } else {
                std::vector<std::string> name_list;
                for (int operand_id = 0; operand_id < (int)this->input_spec.operands.size(); ++operand_id) {
                    name_list.push_back(this->input_spec.operands[operand_id].get_name());
                }

                to_scalar.add_arg(to_tuple_name_if_needed(name_list));
            }

            vector_reduce_invocation.add_arg(to_scalar.to_string());
        } else {

            std::vector<std::string> name_list;
            for (int operand_id = 0; operand_id < (int)this->input_spec.operands.size(); ++operand_id) {
                name_list.push_back(this->input_spec.operands[operand_id].get_name());
            }

            vector_reduce_invocation.add_arg(to_tuple_name_if_needed(name_list));
        }

        code_stream << this->reduce_accum.get_name() << " = " << vector_reduce_invocation.to_string() << ";\n";

        return code_stream.str();
    }

//    std::string ReduceImpl::generate_with_index_impl() const {
//        std::string reduction_function_name;
//        if (input_spec.reducer == Reducer::Sum) {
//            reduction_function_name = "atomicAdd";
//        } else  if (input_spec.reducer == Reducer::Min) {
//            reduction_function_name = "atomicMin";
//        } else if (input_spec.reducer == Reducer::Max) {
//            reduction_function_name = "atomicMax";
//        }
//
//        std::stringstream ss;
//
//        if (this->is_instruction_parallelized()) {
//            for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
//                std::string index = this->ilp_traced_index_list[ilp_id][0].index_before_trace;
//                TensorShape output_shape = this->output.get_shape();
//                std::string reduction_index = mononn_engine::helpers::string_format("(%s) / (%d)", index.c_str(), output_shape.get_shape(-1));
//                std::string output_buffer = BufferManager::get_buffer_name(this->output.get_name());
//                std::string ptr = mononn_engine::helpers::string_format("%s + (%s)", output_buffer.c_str(), reduction_index.c_str());
//
//                FunctionInvocation reduce_invocation(reduction_function_name);
//                reduce_invocation.add_arg(ptr);
//                reduce_invocation.add_arg(this->input_spec.operand.get_name() + "[" + std::to_string(ilp_id) + "]");
//                ss << reduce_invocation.to_string() << ";\n";
//            }
//        } else {
//            std::string index = this->traced_index_list[0].index_before_trace;
//            TensorShape output_shape = this->output.get_shape();
//            std::string reduction_index = mononn_engine::helpers::string_format("(%s) / (%d)", index.c_str(), output_shape.get_shape(-1));
//            std::string output_buffer = BufferManager::get_buffer_name(this->output.get_name());
//            std::string ptr = mononn_engine::helpers::string_format("%s + (%s)", output_buffer.c_str(), reduction_index.c_str());
//
//            FunctionInvocation reduce_invocation(reduction_function_name);
//            reduce_invocation.add_arg(ptr);
//            reduce_invocation.add_arg(this->input_spec.operand.get_name());
//            ss << reduce_invocation.to_string() << ";\n";
//        }
//
//        return ss.str();
//    }

    std::string ReduceImpl::generate_reduce() const {
        std::stringstream code_stream;

        // Use primitive type even for instruction level parallelism
        // Dtype type = this->output.get_dtype().get_primitive_type();
        std::string type_string = this->input_spec.reduce_accum.get_type_string();
        std::string node_name = this->output.get_name();
        code_stream << type_string << " " << node_name << " = ";
        code_stream << this->get_invocation().to_string() << ";\n";


        FunctionInvocation broadcast_invocation;

        if (this->tier == LocalityTier::kT1) {
            broadcast_invocation = FunctionInvocation("warp_broadcast");
            broadcast_invocation.add_arg(node_name);
        } else if (this->tier == LocalityTier::kT2) {
            broadcast_invocation = FunctionInvocation("block_broadcast");
            broadcast_invocation.add_arg(node_name);
            broadcast_invocation.add_arg(Config::get()->smem_reduction_cache_name);
        } else {
            LOG(FATAL) << "";
        }

        code_stream << node_name << " = " << broadcast_invocation.to_string() << ";\n";

        // if (this->is_instruction_parallelized()) {
        //     for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
        //         code_stream << "const "<< type.to_string() << " &" << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id) << " = " <<  node_name << ";\n";
        //     }
        // }

        return code_stream.str();
    }

    std::vector<Tensor> ReduceImpl::get_input_tensor() const {
        return this->input_spec.operands;
    }

    std::vector<Tensor> ReduceImpl::get_output_tensor() const {
        return std::vector<Tensor> { this->output };
    }

    int ReduceImpl::get_elements_per_access() const {
        return this->elements_per_access;
    }

    // std::shared_ptr<Schedule> ReduceImpl::get_schedule() const {
    //     return this->schedule;
    // }

    // void ReduceImpl::set_schedule(std::shared_ptr<Schedule> _schedule) {
    //     this->schedule = _schedule;
    // }

    std::string ReduceImpl::get_prerequisite_definition() {

        return std::string(R"(
template<typename Element, typename ReductionOp, int WarpSize, int N>
__device__ __forceinline__
Element warp_custom_reduce(Element my_val) {
    ReductionOp op;
    Element ret = my_val;

    #pragma unroll
    for (int offset = (WarpSize >> 1); offset > 0; offset >>= 1) {
        ret  = op(ret, __shfl_down_sync(0xffffffff, ret, offset));
    }

    return ret;
}

template<typename Element, typename ReductionOp, int WarpSize, int N>
__device__ __forceinline__
cutlass::AlignedArray<Element, N> warp_custom_reduce(cutlass::AlignedArray<Element, N> &my_val) {
    ReductionOp op;
    cutlass::AlignedArray<Element, N> ret = my_val;

    #pragma unroll
    for (int offset = (WarpSize >> 1); offset > 0; offset >>= 1) {
        #pragma unroll 
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = op(ret[idx], __shfl_down_sync(0xffffffff, ret[idx], offset));
        }
    }

    return ret;
}

template<typename Element, typename ReductionOp, int WarpSize, int N>
__device__ __forceinline__
Element warp_reduce_cub(Element my_val) {
    static_assert(WarpSize == 32, "Warp size should use 32"); // Power of 2 warp size will not use shared memory.
    extern __shared__ int8_t __cache[];
    Element *s_cache = reinterpret_cast<Element *>(__cache);

    using WarpReduce = cub::WarpReduce<Element, WarpSize, __CUDA_ARCH_GLOBAL__>;
    typename WarpReduce::TempStorage *temp_storage = reinterpret_cast<typename WarpReduce::TempStorage *>(s_cache);
    return WarpReduce(temp_storage[threadIdx.x / 32]).Reduce(my_val, ReductionOp());
}

template<typename Element, typename ReductionOp, int WarpSize, int N>
__device__ __forceinline__
cutlass::AlignedArray<Element, N> warp_reduce_cub(cutlass::AlignedArray<Element, N> &my_val) {
    static_assert(WarpSize == 32, "Warp size should use 32"); // Power of 2 warp size will not use shared memory.

    extern __shared__ int8_t __cache[];
    Element *s_cache = reinterpret_cast<Element *>(__cache);

    using WarpReduce = cub::WarpReduce<Element, WarpSize, __CUDA_ARCH_GLOBAL__>;
    typename WarpReduce::TempStorage *temp_storage = reinterpret_cast<typename WarpReduce::TempStorage *>(s_cache);

    cutlass::AlignedArray<Element, N> result;

    for (int idx = 0; idx < N; ++idx) {
        result[idx] = WarpReduce(temp_storage[threadIdx.x / 32]).Reduce(my_val[idx], ReductionOp());
    }

    return result;
}

template<typename Element, typename ReductionOp, int BlockSize, int N>
__device__ __forceinline__
Element block_custom_reduce(Element my_val, void *__cache) {
    const int lane_id = threadIdx.x & 0x1f;
    const int warp_id = threadIdx.x >> 5;

    Element *s_cache = reinterpret_cast<Element *>(__cache);

    Element ret = my_val;
    ret = warp_custom_reduce<Element, ReductionOp, 32, 1>(ret);

    if (lane_id == 0) 
        s_cache[warp_id] = ret;
    __syncthreads();
    
    ret = (threadIdx.x < blockDim.x / 32) ? s_cache[lane_id] : Element(0);
    ret = warp_custom_reduce<Element, ReductionOp, 32, 1>(ret);
    return ret;
}

template<typename Element, typename ReductionOp, int BlockSize, int N>
__device__ __forceinline__
Element block_reduce_cub_RAKING(Element my_val, void *__cache) {
    Element *s_cache = reinterpret_cast<Element *>(__cache);
    using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, 1, 1, __CUDA_ARCH_GLOBAL__>;
    typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);
    return BlockReduce(*temp_storage).Reduce(my_val, ReductionOp());
}

template<typename Element, typename ReductionOp, int BlockSize, int N>
__device__ __forceinline__
Element block_reduce_cub_RAKING_COMMUTATIVE_ONLY(Element my_val, void *__cache) {
    Element *s_cache = reinterpret_cast<Element *>(__cache);
    using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, __CUDA_ARCH_GLOBAL__>;
    typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);
    return BlockReduce(*temp_storage).Reduce(my_val, ReductionOp());
}

template<typename Element, typename ReductionOp, int BlockSize, int N>
__device__ __forceinline__
Element block_reduce_cub_WARP_REDUCTIONS(Element my_val, void *__cache) {
    Element *s_cache = reinterpret_cast<Element *>(__cache);
    using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, __CUDA_ARCH_GLOBAL__>;
    typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);
    return BlockReduce(*temp_storage).Reduce(my_val, ReductionOp());
}


template<typename T, int N, typename ReductionOp>
__device__ __forceinline__
T to_scalar(const cutlass::Array<T, N> &array) {
    ReductionOp op;
    T ret = array[0];

    #pragma unroll
    for (int iter = 1; iter < N; ++iter) {
        ret = op(ret, array[iter]);
    }

    return ret;
}

template<typename T, int N, typename ReductionOp>
__device__ __forceinline__
T to_scalar(const cutlass::AlignedArray<T, N> &array) {
    ReductionOp op;
    T ret = array[0];

#pragma unroll
    for (int iter = 1; iter < N; ++iter) {
        ret = op(ret, array[iter]);
    }

    return ret;
}

template<typename T, typename ReductionOp>
__device__ __forceinline__
T to_scalar(const T &val) {
    return val;
}

template<typename T, typename ReductionOp, typename ... TArgs>
__device__ __forceinline__
T to_scalar(const T &val, const TArgs &... args) {
    ReductionOp op;
    return op(val, to_scalar<T, ReductionOp>(args...));
}

template<typename T, int N, typename ReductionOp, typename ... TArgs>
__device__ __forceinline__
T to_scalar(const cutlass::Array<T, N> &array, const TArgs &... args) {
    ReductionOp op;
    T ret = array[0];

#pragma unroll
    for (int iter = 1; iter < N; ++iter) {
        ret = op(ret, array[iter]);
    }

    return op(ret, to_scalar<T, N, ReductionOp>(args...));
}

template<typename T, int N, typename ReductionOp, typename ... TArgs>
__device__ __forceinline__
T to_scalar(const cutlass::AlignedArray<T, N> &array, const TArgs &... args) {
    ReductionOp op;
    T ret = array[0];

#pragma unroll
    for (int iter = 1; iter < N; ++iter) {
        ret = op(ret, array[iter]);
    }

    return op(ret, to_scalar<T, N, ReductionOp>(args...));
}

template<typename T>
__device__ __forceinline__
T warp_broadcast(const T &val) {
    return __shfl_sync(0xffffffff, val, 0);
}

template<typename T>
__device__ __forceinline__
T block_broadcast(const T &val, void *__cache){
    T *s_cache = reinterpret_cast<T *>(__cache);
    if (threadIdx.x == 0) {
        s_cache[0] = val;
    }

    __syncthreads();

    return s_cache[0];
}
        )");
    }

    const Scalar &ReduceImpl::get_reduce_accum() const {
        return this->reduce_accum;
    }

    void ReduceImpl::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;

        for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
            auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
        }
    }

    // Functor ReduceImpl::get_reduce_functor(Reducer reducer, Dtype element_type) {
    //     if (reducer == Reducer::Sum) {
    //         return Functor("plus", element_type);
    //     } else if (reducer == Reducer::Max) {
    //         return Functor("maximum", element_type);
    //     } else if (reducer == Reducer::Min) {
    //         return Functor("minimum", element_type);
    //     }

    //     LOG(FATAL) << "Unsupported reducer " << (int)reducer;
    // }
}
}
}