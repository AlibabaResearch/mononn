// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>

#include "mononn_engine/tuning/profiler/subprocess.h"

using SubProcess = mononn_engine::tuning::profiler::SubProcess;

int main() {
  SubProcess process(
      "/home/zhuangdonglin.zdl/workspace/tensorflow/bazel-bin/tensorflow/"
      "onefuser/codegen/graph_specification_codegen_main",
      {"--graph_spec_file=/apsarapangu/disk1/zhuangdonglin.zdl/tensorflow/"
       "tensorflow/onefuser/build_generated/graph_spec_list/108_128/"
       "cublas_batch_gemm_1/0.pb",
       "--output_dir=/apsarapangu/disk1/zhuangdonglin.zdl/tensorflow/"
       "tensorflow/onefuser/build_generated/tmp",
       "2>&1"});

  //    SubProcess process("pwd");
  process.start();
  process.wait();

  std::cout << "Exit code:: " << process.get_return_code() << std::endl;
  std::cout << process.get_output() << std::endl;
  return 0;
}