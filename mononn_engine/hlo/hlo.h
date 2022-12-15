#include <string>
#include "mononn_engine/config/config.h"

namespace hlo {
    using Config = mononn_engine::config::Config;
    std::string hlo_text_mini = R"(
HloModule cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.566

%StatefulPartitionedCall_model_1_keras_layer_StatefulPartitionedCall_StatefulPartitionedCall_StatefulPartitionedCall_bert_encoder_StatefulPartitionedCall_embeddings_layer_norm_moments_mean-reduction.38 (x.39: f32[], y.40: f32[]) -> f32[] {
  %x.39 = f32[] parameter(0)
  %y.40 = f32[] parameter(1)
  ROOT %add.41 = f32[] add(f32[] %x.39, f32[] %y.40)
}

ENTRY %cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.566 (arg0.1: s32[1,128], arg1.2: s32[1,128]) -> f32[1,128] {
  %constant_491 = f32[128]{0} constant({...}), metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_1/output/add"}
  %broadcast.492 = f32[1,128,128]{2,1,0} broadcast(f32[128]{0} %constant_491), dimensions={2}, metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_1/output/add"}
  %constant_247 = f32[] constant(0.044715), metadata={op_type="Mul" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_0/activation/Gelu/mul_1"}
  %broadcast.467 = f32[1,128,512]{2,1,0} broadcast(f32[] %constant_247), dimensions={}, metadata={op_type="Mul" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_1/activation/Gelu/mul_1"}
  %constant_458 = f32[512]{0} constant({...}), metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_1/intermediate/add"}
  %broadcast.459 = f32[1,128,512]{2,1,0} broadcast(f32[512]{0} %constant_458), dimensions={2}, metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_1/intermediate/add"}
  %constant_272 = f32[128]{0} constant({...}), metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_0/output/add"}
  %broadcast.273 = f32[1,128,128]{2,1,0} broadcast(f32[128]{0} %constant_272), dimensions={2}, metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_0/output/add"}
  %constant_239 = f32[512]{0} constant({...}), metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_0/intermediate/add"}
  %broadcast.240 = f32[1,128,512]{2,1,0} broadcast(f32[512]{0} %constant_239), dimensions={2}, metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/transformer/layer_0/intermediate/add"}
  %constant_26 = f32[30522,128]{1,0} constant({...}), metadata={op_type="GatherV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/word_embeddings/Gather"}
  %arg0.1 = s32[1,128]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %bitcast.32 = s32[128]{0} bitcast(s32[1,128]{1,0} %arg0.1), metadata={op_type="Reshape" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/word_embeddings/Reshape"}
  %gather.28 = f32[128,128]{1,0} gather(f32[30522,128]{1,0} %constant_26, s32[128]{0} %bitcast.32), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,128}, metadata={op_type="GatherV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/word_embeddings/Gather"}
  %constant = f32[128,128]{1,0} constant({...})
  %add.6 = f32[128,128]{1,0} add(f32[128,128]{1,0} %gather.28, f32[128,128]{1,0} %constant), metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/add/add"}
  %arg1.2 = s32[1,128]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %bitcast.33 = s32[128]{0} bitcast(s32[1,128]{1,0} %arg1.2), metadata={op_type="Reshape" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/Reshape"}
  %broadcast.17 = s32[128,2]{1,0} broadcast(s32[128]{0} %bitcast.33), dimensions={0}, metadata={op_type="OneHot" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/one_hot"}
  %iota.16 = s32[128,2]{1,0} iota(), iota_dimension=1, metadata={op_type="OneHot" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/one_hot"}
  %compare.18 = pred[128,2]{1,0} compare(s32[128,2]{1,0} %broadcast.17, s32[128,2]{1,0} %iota.16), direction=EQ, metadata={op_type="OneHot" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/one_hot"}
  %constant_13 = f16[] constant(1), metadata={op_type="OneHot" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/one_hot"}
  %broadcast.15 = f16[128,2]{1,0} broadcast(f16[] %constant_13), dimensions={}, metadata={op_type="OneHot" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/one_hot"}
  %constant_12 = f16[] constant(0), metadata={op_type="OneHot" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/one_hot"}
  %broadcast.14 = f16[128,2]{1,0} broadcast(f16[] %constant_12), dimensions={}, metadata={op_type="OneHot" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/one_hot"}
  %select.19 = f16[128,2]{1,0} select(pred[128,2]{1,0} %compare.18, f16[128,2]{1,0} %broadcast.15, f16[128,2]{1,0} %broadcast.14), metadata={op_type="OneHot" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/one_hot"}
  %pad = f16[128,8]{1,0} pad(f16[128,2]{1,0} %select.19, f16[] %constant_12), padding=0_0x0_6, metadata={op_type="MatMul" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/MatMul"}
  %constant_20 = f16[2,128]{1,0} constant({...}), metadata={op_type="MatMul" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/MatMul"}
  %pad.1 = f16[8,128]{1,0} pad(f16[2,128]{1,0} %constant_20, f16[] %constant_12), padding=0_6x0_0, metadata={op_type="MatMul" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/MatMul"}
  %custom-call = f16[128,128]{1,0} custom-call(f16[128,8]{1,0} %pad, f16[8,128]{1,0} %pad.1), custom_call_target="__cublas$gemm", metadata={op_type="MatMul" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/MatMul"}, backend_config="{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"batch_size\":\"1\",\"activation_mode\":\"0\",\"selected_algorithm\":\"8\"}"
  %convert = f32[128,128]{1,0} convert(f16[128,128]{1,0} %custom-call), metadata={op_type="Cast" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/type_embeddings/Reshape_1-0-CastToFp32-AutoMixedPrecision"}
  %add.8 = f32[128,128]{1,0} add(f32[128,128]{1,0} %add.6, f32[128,128]{1,0} %convert), metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/add/add_1"}
  %bitcast.34 = f32[1,128,128]{2,1,0} bitcast(f32[128,128]{1,0} %add.8), metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/add/add_1"}
  %constant_68 = f32[] constant(1e-12), metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/embeddings/layer_norm/batchnorm/add"}
  %broadcast.29 = f32[1,128]{1,0} broadcast(f32[] %constant_68), dimensions={}, metadata={op_type="AddV2" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/embeddings/layer_norm/batchnorm/add"}
  %constant_36 = f32[] constant(0), metadata={op_type="Mean" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/embeddings/layer_norm/moments/mean"}
  %reduce.1 = f32[128]{0} reduce(f32[128,128]{1,0} %add.8, f32[] %constant_36), dimensions={1}, to_apply=%StatefulPartitionedCall_model_1_keras_layer_StatefulPartitionedCall_StatefulPartitionedCall_StatefulPartitionedCall_bert_encoder_StatefulPartitionedCall_embeddings_layer_norm_moments_mean-reduction.38
  ROOT %bitcast.1 = f32[1,128]{1,0} bitcast(f32[128]{0} %reduce.1), metadata={op_type="Mean" op_name="StatefulPartitionedCall/model_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/bert_encoder/StatefulPartitionedCall/embeddings/layer_norm/moments/mean"}
}
    )";

    std::string bert_tiny_bc4_fusion_42() {
        std::ifstream f("/apsarapangu/disk1/zhuangdonglin.zdl/tensorflow/tensorflow/onefuser/test/hlo/bert_tiny_fusion_42.txt");
        std::stringstream buffer;
        buffer << f.rdbuf();

        return buffer.str();
    }


    std::string bert_tiny_bc4_fusion_51() {
        std::ifstream f("/apsarapangu/disk1/zhuangdonglin.zdl/tensorflow/tensorflow/onefuser/test/hlo/bert_tiny_fusion_51.txt");
        std::stringstream buffer;
        buffer << f.rdbuf();

        return buffer.str();
    }

    std::string bert_tiny_bc4_test() {
        std::ifstream f("/apsarapangu/disk1/zhuangdonglin.zdl/tensorflow/tensorflow/onefuser/test/hlo/bert_tiny_test.txt");
        std::stringstream buffer;
        buffer << f.rdbuf();

        return buffer.str();
    }

    std::string bert_base_bc4() {
        std::ifstream f("/home/zhuangdonglin.zdl/workspace/models/bert_base/dump/text_bc4_with_weight/1646136118794537.module_4922.cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_589__XlaNumResourceArgs_0_.5620.sm_8.0_gpu_after_optimizations.txt");
        std::stringstream buffer;
        buffer << f.rdbuf();

        return buffer.str();
    }
}