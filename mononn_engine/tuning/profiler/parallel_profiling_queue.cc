#include <experimental/filesystem>

#include "mononn_engine/tuning/profiler/parallel_profiling_queue.h"
#include "mononn_engine/tuning/profiler/subprocess.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace tuning {
namespace profiler {
    using Config = mononn_engine::config::Config;
    using NCUResult = ParallelProfilingQueue::NCUResult;
    namespace fs = std::experimental::filesystem;

    std::future<ProfilingResult> ParallelProfilingQueue::post(const GraphSpecification *graph_spec, std::vector<std::string> host_codegen_disabled_pass, std::vector<std::string> optimization_disabled_pass) {
        return std::move(this->thread_pool.enqueue([this, graph_spec, host_codegen_disabled_pass, optimization_disabled_pass]() -> ProfilingResult {
            ProfilingResult profiling_result;
            profiling_result.codegen_success = false;
            profiling_result.build_success = false;
            profiling_result.profile_success = false;

            this->thread_mutex.lock();
            std::string tmp_directory = mononn_engine::helpers::Directory::get_mononn_root_temp_dir();

            tmp_directory = mononn_engine::helpers::Path::join(tmp_directory, mononn_engine::helpers::UUID::new_uuid());

            if (fs::exists(tmp_directory) && fs::is_directory(tmp_directory)) {
                LOG(WARNING) << "Directory " << tmp_directory << " already exists";

                fs::remove(tmp_directory);
            }

            if (!fs::create_directories(tmp_directory)) {
                LOG(FATAL) << "Create directory " << tmp_directory << " failed.";
            }

            profiling_result.profiling_directory = tmp_directory;

            this->thread_mutex.unlock();
            std::string graph_spec_pb_file = mononn_engine::helpers::Path::join(tmp_directory, "graph_spec.pb");
            std::string graph_spec_json_file = mononn_engine::helpers::Path::join(tmp_directory, "graph_spec.json");

            mononn_engine::helpers::save_proto_to_binary_file(graph_spec, graph_spec_pb_file);
            mononn_engine::helpers::save_proto_to_json_file(graph_spec, graph_spec_json_file);

            for (int codegen_cnt = 0; codegen_cnt < 5; ++codegen_cnt) {
                std::vector<std::string> params =  {"--graph_spec_file=" + graph_spec_pb_file,
                                                    "--output_dir=" + tmp_directory};
                if (!host_codegen_disabled_pass.empty()) {
                    // params.push_back("--host_codegen_disabled_pass=generate_memory_initialization,generate_parameter_initialization");
                    params.push_back("--host_codegen_disabled_pass=" + mononn_engine::helpers::join(",", host_codegen_disabled_pass));
                }

                if (!optimization_disabled_pass.empty()) {
                    params.push_back("--optimization_disabled_pass=" + mononn_engine::helpers::join(",", optimization_disabled_pass));
                }

                if (!Config::get()->optimization_disabled_pass.empty()) {
                    std::string disabled_passes = mononn_engine::helpers::join(",", Config::get()->optimization_disabled_pass);
                    disabled_passes = "--optimization_disabled_pass=" + disabled_passes;
                    params.push_back(disabled_passes);
                }

                params.push_back("2>&1");

                SubProcess codegen_process(Config::get()->graph_spec_compiler_path, params);
                codegen_process.start();
                codegen_process.wait();

                if (codegen_process.get_return_code() != 0) {
                    profiling_result.output += mononn_engine::helpers::string_format("Failed on codegen attempt %d===\n", codegen_cnt);
                    profiling_result.output += codegen_process.get_output();
                    this->thread_mutex.lock();
                    LOG(WARNING) << "Attempt "<< codegen_cnt << ". Codegen failed for sample " << tmp_directory;
                    this->thread_mutex.unlock();
                    std::this_thread::sleep_for(std::chrono::seconds(60 * codegen_cnt)); // wait and retry;
                    continue;
                } else {
                    profiling_result.codegen_success = true;
                    break;
                }
            }

            if (!profiling_result.codegen_success) { // codegen fail
                return profiling_result;
            }

            std::string build_cmd = "cd " + tmp_directory + " && make";

            SubProcess build_process(build_cmd, {"2>&1"});
            build_process.start();
            build_process.wait();

            if (build_process.get_return_code() != 0) {
                profiling_result.build_success = false;
                profiling_result.output = build_process.get_output();
                LOG(WARNING) << "Build failed in " << tmp_directory;
                LOG(WARNING) << profiling_result.output;

                return profiling_result;
            } else {
                profiling_result.build_success = true;
            }

            int gpu_id = -1;

            typedef std::chrono::high_resolution_clock Time;
            typedef std::chrono::milliseconds ms;
            typedef std::chrono::duration<float> fsec;
            auto t0 = Time::now();
            while (true) {
                for (int idx = 0; idx < Config::get()->gpus.size(); ++idx) {
                    if (this->profile_mutex[idx].try_lock()) {
                        gpu_id = idx;
                        break;
                    }
                }

                if (gpu_id != -1) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }

            for (int profiling_cnt = 0; profiling_cnt < 5; ++profiling_cnt) {
                auto t1 = Time::now();

                std::string profiling_cmd = " cd " + tmp_directory + " && ";
                profiling_cmd += "CUDA_VISIBLE_DEVICES=\"" + std::to_string(Config::get()->gpus[gpu_id]) + "\"";
//                profiling_cmd += " ncu --csv --app-replay-buffer memory --metrics gpu__time_duration.sum ./onefuser";
                profiling_cmd += " ./mononn";
                SubProcess profiling_process(profiling_cmd, {"2>&1"});

                auto t2 = Time::now();
                profiling_process.start();
                profiling_process.wait();
                auto t3 = Time::now();

                if (profiling_process.get_return_code() != 0) {
                    profiling_result.output += mononn_engine::helpers::string_format("Failed on profiling attempt %d===\n", profiling_cnt);
                    profiling_result.output += profiling_process.get_output();
                    this->thread_mutex.lock();
                    LOG(WARNING) << "Attempt "<< profiling_cnt << ". Profiling failed for sample " << tmp_directory << " on gpu " << Config::get()->gpus[gpu_id];
                    this->thread_mutex.unlock();
                    std::this_thread::sleep_for(std::chrono::seconds(60 * profiling_cnt * 10)); // wait and retry;
                    continue;
                }

                this->profile_mutex[gpu_id].unlock();

                profiling_result.profile_success = true;
//                NCUResult ncu_result = this->parse_ncu_result(profiling_process.get_output());
                NCUResult ncu_result = this->parse_console_result(profiling_process.get_output());
                profiling_result.time_in_us = ncu_result.time_in_us;

                this->thread_mutex.lock();
                fs::remove_all(tmp_directory);
                LOG(INFO) << "Removing directory: " << tmp_directory << " Wait latency " << std::chrono::duration_cast<ms>(t1 - t0).count()
                          << " execution latency " << std::chrono::duration_cast<ms>(t3 - t2).count();
                this->thread_mutex.unlock();

                return profiling_result;
            }

            this->profile_mutex[gpu_id].unlock();

            profiling_result.profile_success = false;
            return profiling_result;
        }));
    }

    NCUResult ParallelProfilingQueue::parse_ncu_result(std::string str) const {
        std::vector<std::string> lines = mononn_engine::helpers::string_split(str, '\n');
        NCUResult ncu_result;

        bool begin_section = false;
        for (auto &line : lines) {
            if (line == R"("ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value")") {
                begin_section = true;

                std::vector<std::string> cols = mononn_engine::helpers::string_split(line, ',');

                if (cols[9] != R"("Metric Name")") {
                    LOG(FATAL) << "Col 10 should be metric name: " << line;
                }

                if (cols[10] != R"("Metric Unit")") {
                    LOG(FATAL) << "Col 11 should be metric unit: " << line;
                }

                if (cols[11] != R"("Metric Value")") {
                    LOG(FATAL) << "Col 12 should be metric value: " << line;
                }

                continue;
            }

            if (begin_section) {
                std::vector<std::string> cols = mononn_engine::helpers::string_split(line, ',');

                if (cols[9] == R"("gpu__time_duration.sum")") {
                    std::string duration = cols[11].substr(1, cols[11].length() - 2);
                    ncu_result.time_in_us = std::atof(duration.c_str()) / 1000;

                    return ncu_result;
                }
            }
        }

        LOG(FATAL) << "Parse failed, begin section: " << begin_section << "\n" << mononn_engine::helpers::join("\n", lines);
    }

    NCUResult ParallelProfilingQueue::parse_console_result(std::string str) const {
        std::vector<std::string> lines = mononn_engine::helpers::string_split(str, '\n');
        NCUResult ncu_result;

        for (auto &line : lines) {
            if (line.substr(0, 11) == "Time in us:") {
                std::string time_str = line.substr(12);
                ncu_result.time_in_us = std::stof(time_str);
                return ncu_result;
            }
        }

        LOG(FATAL) << "Parse failed, " << str;
    }
}
}
}