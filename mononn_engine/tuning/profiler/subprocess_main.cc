#include <iostream>
#include "mononn_engine/tuning/profiler/subprocess.h"

using SubProcess = mononn_engine::tuning::profiler::SubProcess;

int main() {
    SubProcess process("/home/zhuangdonglin.zdl/workspace/tensorflow/bazel-bin/tensorflow/onefuser/codegen/graph_specification_codegen_main",
                       {"--graph_spec_file=/apsarapangu/disk1/zhuangdonglin.zdl/tensorflow/tensorflow/onefuser/build_generated/graph_spec_list/108_128/cublas_batch_gemm_1/0.pb", "--output_dir=/apsarapangu/disk1/zhuangdonglin.zdl/tensorflow/tensorflow/onefuser/build_generated/tmp", "2>&1"});

//    SubProcess process("pwd");
    process.start();
    process.wait();

    std::cout << "Exit code:: " << process.get_return_code() << std::endl;
    std::cout << process.get_output() << std::endl;
    return 0;
}