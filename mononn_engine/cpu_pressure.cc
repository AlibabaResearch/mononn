#include <chrono>
#include <iostream>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mononn_engine/tuning/profiler/thread_pool.h"

ABSL_FLAG(int, num_thread, -1, "Num thread");
struct Options {
  int num_thread;
};

int Main(Options const& options) {
  mononn_engine::tuning::profiler::ThreadPool thread_pool(options.num_thread);

  std::vector<std::future<float>> results;
  for (int idx = 0; idx < 10000 * options.num_thread; ++idx) {
    results.push_back(std::move(thread_pool.enqueue([]() -> float {
      int N = 500;
      std::vector<std::vector<float>> A, B, C;
      A.resize(N);
      B.resize(N);
      C.resize(N);

      for (int idx = 0; idx < N; ++idx) {
        A[idx].resize(N);
        B[idx].resize(N);
        C[idx].resize(N);
      }

      int a = 3;
      int b = 5;

      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          A[i][j] = a;
          B[i][j] = b;

          a = (a * 11) % 27;
          b = (b * 5) % 17;
        }
      }

      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          for (int k = 0; k < N; ++k) {
            C[i][j] += A[i][j] * B[i][j];
          }
        }
      }

      float res = 0;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          res += C[i][j] / (N * N);
        }
      }

      return res;
    })));
  }

  for (auto& result : results) {
    result.wait();
  }

  //    int alive_job = 0;
  //    std::mutex mtx;
  //    std::vector<std::thread> threads;
  //    while (true) {
  //        mtx.lock();
  //        int need = options.num_thread - alive_job;
  //        alive_job = options.num_thread;
  //        mtx.unlock();
  //
  //        if (need <= 0) {
  //            std::this_thread::sleep_for(std::chrono::milliseconds(20));
  //        }
  //
  //        for (int ii = 0; ii < need; ++ii) {
  //            threads.emplace_back(std::thread([&alive_job, &mtx]() -> float {
  //                int N = 500;
  //                std::vector<std::vector<float>> A, B, C;
  //                A.resize(N);
  //                B.resize(N);
  //                C.resize(N);
  //
  //                for (int idx = 0; idx < N; ++idx) {
  //                    A[idx].resize(N);
  //                    B[idx].resize(N);
  //                    C[idx].resize(N);
  //                }
  //
  //                int a = 3;
  //                int b = 5;
  //
  //                for (int i = 0; i < N; ++i) {
  //                    for (int j = 0; j < N; ++j) {
  //                        A[i][j] = a;
  //                        B[i][j] = b;
  //
  //                        a = (a * 11) % 27;
  //                        b = (b * 5) % 17;
  //                    }
  //                }
  //
  //                for (int i = 0; i < N; ++i) {
  //                    for (int j = 0; j < N; ++j) {
  //                        for (int k = 0; k < N; ++k) {
  //                            C[i][j] += A[i][j] * B[i][j];
  //                        }
  //                    }
  //                }
  //
  //                float res = 0;
  //                for (int i = 0; i < N; ++i) {
  //                    for (int j = 0; j < N; ++j) {
  //                        res += C[i][j] / (N * N);
  //                    }
  //                }
  //
  //                mtx.lock();
  //                alive_job--;
  //                mtx.unlock();
  //
  //                return res;
  //            }));
  //        }
  //    }

  return 0;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  Options options;

  options.num_thread = absl::GetFlag(FLAGS_num_thread);

  if (options.num_thread < 0 || options.num_thread > 110) {
    std::cerr << "Invalid thread number: " << options.num_thread << std::endl;
    return -1;
  }

  return Main(options);
}