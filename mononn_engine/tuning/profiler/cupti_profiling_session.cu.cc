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

#include "mononn_engine/tuning/profiler/cupti_profiling_session.h"

#include <cuda.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <nvperf_target.h>

#include "mononn_engine/helpers/macros.h"

#define CUPTI_API_CALL(apiFuncCall)                                        \
  do {                                                                     \
    CUptiResult _status = apiFuncCall;                                     \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char* errstr;                                                  \
      cuptiGetResultString(_status, &errstr);                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #apiFuncCall, errstr);                   \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define DRIVER_API_CALL(apiFuncCall)                                       \
  do {                                                                     \
    CUresult _status = apiFuncCall;                                        \
    if (_status != CUDA_SUCCESS) {                                         \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", \
              __FILE__, __LINE__, #apiFuncCall, _status);                  \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define NVPW_API_CALL(apiFuncCall)                                         \
  do {                                                                     \
    NVPA_Status _status = apiFuncCall;                                     \
    if (_status != NVPA_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", \
              __FILE__, __LINE__, #apiFuncCall, _status);                  \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

namespace mononn_engine {
namespace tuning {
namespace profiler {
inline void ParseMetricNameString(const std::string& metricName,
                                  std::string* reqName, bool* isolated,
                                  bool* keepInstances);

void GetRawMetricRequests(std::string chipName,
                          const std::vector<std::string>& metricNames,
                          std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
                          const uint8_t* pCounterAvailabilityImage);

void GetConfigImage(std::string chipName,
                    const std::vector<std::string>& metricNames,
                    std::vector<uint8_t>& configImage,
                    const uint8_t* pCounterAvailabilityImage);

void GetCounterDataPrefixImage(
    std::string chipName, const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& counterDataImagePrefix,
    const uint8_t* pCounterAvailabilityImage = nullptr);
void CreateCounterDataImage(std::vector<uint8_t>& counterDataImage,
                            std::vector<uint8_t>& counterDataScratchBuffer,
                            std::vector<uint8_t>& counterDataImagePrefix,
                            int numRanges);

void validate_device_supported_params(
    const CUpti_Profiler_DeviceSupported_Params& params, const int& deviceNum) {
  if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
    LOG(WARNING) << "Unable to profile on device " << deviceNum << ::std::endl;

    if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      LOG(FATAL) << "\tdevice architecture is not supported" << ::std::endl;
    }

    if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      LOG(FATAL) << "\tdevice sli configuration is not supported"
                 << ::std::endl;
    }

    if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      LOG(FATAL) << "\tdevice vgpu configuration is not supported"
                 << ::std::endl;
    } else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED) {
      LOG(FATAL) << "\tdevice vgpu configuration disabled profiling support"
                 << ::std::endl;
    }

    if (params.confidentialCompute ==
        CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      LOG(FATAL)
          << "\tdevice confidential compute configuration is not supported"
          << ::std::endl;
    }

    if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      LOG(FATAL) << "\tNVIDIA Crypto Mining Processors (CMP) are not supported"
                 << ::std::endl;
    }

    LOG(FATAL) << "\tcannot determine precise error";
  }
}

CuptiProfilingSession::CuptiProfilingSession(
    const std::vector<std::string>& _metricNames, int _numRanges)
    : metricNames(_metricNames), numRanges(_numRanges) {
  // We assump already have context
  // DRIVER_API_CALL(cuInit(0));
  // CUdevice cuDevice;
  CUcontext cuContext;
  DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));

  if (!cuContext) {
    LOG(FATAL) << "CUDA Context should be initialized.";
  }

  // DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
  // DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));

  this->cupti_initialize();
}

CuptiProfilingSession::ProfilingResult CuptiProfilingSession::profiling_context(
    std::function<void()> kernel_func_wrapper) {
  if (this->already_profiled) {
    LOG(FATAL)
        << "Repeated profiling_context run is not allowed at this moment";
  }

  this->already_profiled = true;

  CUpti_Profiler_BeginSession_Params beginSessionParams = {
      CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
  CUpti_Profiler_SetConfig_Params setConfigParams = {
      CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
  CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
      CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
  CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
      CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};

  beginSessionParams.ctx = NULL;
  beginSessionParams.counterDataImageSize = this->counterDataImage.size();
  beginSessionParams.pCounterDataImage = &(this->counterDataImage[0]);
  beginSessionParams.counterDataScratchBufferSize =
      this->counterDataScratchBuffer.size();
  beginSessionParams.pCounterDataScratchBuffer =
      &(this->counterDataScratchBuffer[0]);
  beginSessionParams.range = CUPTI_AutoRange;
  beginSessionParams.replayMode = CUPTI_KernelReplay;
  beginSessionParams.maxRangesPerPass = this->numRanges;
  beginSessionParams.maxLaunchesPerPass = this->numRanges;

  CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
  setConfigParams.pConfig = &configImage[0];
  setConfigParams.configSize = configImage.size();
  setConfigParams.passIndex = 0;

  CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
  CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));

  kernel_func_wrapper();

  CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
      CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
  CUpti_Profiler_EndSession_Params endSessionParams = {
      CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

  return this->decode_profiling_result();
}

CuptiProfilingSession::~CuptiProfilingSession() {
  CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
      CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
}

void CuptiProfilingSession::cupti_initialize() {
  CUpti_Profiler_Initialize_Params profilerInitializeParams = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
  CUpti_Profiler_DeviceSupported_Params params = {
      CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
  params.cuDevice = this->deviceNum;
  CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

  validate_device_supported_params(params, this->deviceNum);

  CUpti_Device_GetChipName_Params getChipNameParams = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  getChipNameParams.deviceIndex = this->deviceNum;
  CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
  this->chipName = getChipNameParams.pChipName;

  CUcontext cuContext;
  DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));

  CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  getCounterAvailabilityParams.ctx = cuContext;
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  this->counterAvailabilityImage.resize(
      getCounterAvailabilityParams.counterAvailabilityImageSize);
  getCounterAvailabilityParams.pCounterAvailabilityImage =
      this->counterAvailabilityImage.data();
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  /* Generate configuration for metrics, this can also be done offline*/
  NVPW_InitializeHost_Params initializeHostParams = {
      NVPW_InitializeHost_Params_STRUCT_SIZE};
  NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

  GetConfigImage(this->chipName, this->metricNames, this->configImage,
                 this->counterAvailabilityImage.data());
  GetCounterDataPrefixImage(this->chipName, this->metricNames,
                            this->counterDataImagePrefix);
  CreateCounterDataImage(this->counterDataImage, this->counterDataScratchBuffer,
                         this->counterDataImagePrefix, this->numRanges);
}

const std::string CuptiProfilingSession::Metrics::gpu__time_duration_sum =
    "gpu__time_duration.sum";

CuptiProfilingSession::ProfilingResult
CuptiProfilingSession::decode_profiling_result() const {
  ProfilingResult profiling_result;

  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params
      calculateScratchBufferSizeParam = {
          NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  calculateScratchBufferSizeParam.pChipName = this->chipName.c_str();
  calculateScratchBufferSizeParam.pCounterAvailabilityImage = nullptr;
  NVPW_API_CALL(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(
      &calculateScratchBufferSizeParam));

  std::vector<uint8_t> scratchBuffer(
      calculateScratchBufferSizeParam.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams =
      {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
  metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
  metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
  metricEvaluatorInitializeParams.pChipName = this->chipName.c_str();
  metricEvaluatorInitializeParams.pCounterAvailabilityImage = nullptr;
  metricEvaluatorInitializeParams.pCounterDataImage =
      this->counterDataImage.data();
  metricEvaluatorInitializeParams.counterDataImageSize =
      this->counterDataImage.size();
  NVPW_API_CALL(
      NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
  NVPW_MetricsEvaluator* metricEvaluator =
      metricEvaluatorInitializeParams.pMetricsEvaluator;

  NVPW_CounterData_GetNumRanges_Params getNumRangesParams = {
      NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE};
  getNumRangesParams.pCounterDataImage = this->counterDataImage.data();
  NVPW_API_CALL(NVPW_CounterData_GetNumRanges(&getNumRangesParams));

  std::string reqName;
  bool isolated = true;
  bool keepInstances = true;

  for (std::string metricName : this->metricNames) {
    ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
    NVPW_MetricEvalRequest metricEvalRequest;
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
        convertMetricToEvalRequest = {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
    convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
    convertMetricToEvalRequest.pMetricName = reqName.c_str();
    convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
    convertMetricToEvalRequest.metricEvalRequestStructSize =
        NVPW_MetricEvalRequest_STRUCT_SIZE;
    NVPW_API_CALL(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(
        &convertMetricToEvalRequest));

    if (getNumRangesParams.numRanges != this->numRanges) {
      LOG(FATAL) << "Got mismatch num range, this may indicate kernel "
                    "launch/execution failure."
                 << " Expected " << this->numRanges << " got "
                 << getNumRangesParams.numRanges;
    }

    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges;
         ++rangeIndex) {
      NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams =
          {NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE};
      getRangeDescParams.pCounterDataImage = this->counterDataImage.data();
      getRangeDescParams.rangeIndex = rangeIndex;
      NVPW_API_CALL(
          NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
      std::vector<const char*> descriptionPtrs(
          getRangeDescParams.numDescriptions);
      getRangeDescParams.ppDescriptions = descriptionPtrs.data();
      NVPW_API_CALL(
          NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

      std::string rangeName;
      for (size_t descriptionIndex = 0;
           descriptionIndex < getRangeDescParams.numDescriptions;
           ++descriptionIndex) {
        if (descriptionIndex) {
          rangeName += "/";
        }
        rangeName += descriptionPtrs[descriptionIndex];
      }

      NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = {
          NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE};
      setDeviceAttribParams.pMetricsEvaluator = metricEvaluator;
      setDeviceAttribParams.pCounterDataImage = this->counterDataImage.data();
      setDeviceAttribParams.counterDataImageSize =
          this->counterDataImage.size();
      NVPW_API_CALL(
          NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttribParams));

      double metricValue;
      NVPW_MetricsEvaluator_EvaluateToGpuValues_Params
          evaluateToGpuValuesParams = {
              NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE};
      evaluateToGpuValuesParams.pMetricsEvaluator = metricEvaluator;
      evaluateToGpuValuesParams.pMetricEvalRequests = &metricEvalRequest;
      evaluateToGpuValuesParams.numMetricEvalRequests = 1;
      evaluateToGpuValuesParams.metricEvalRequestStructSize =
          NVPW_MetricEvalRequest_STRUCT_SIZE;
      evaluateToGpuValuesParams.metricEvalRequestStrideSize =
          sizeof(NVPW_MetricEvalRequest);
      evaluateToGpuValuesParams.pCounterDataImage =
          this->counterDataImage.data();
      evaluateToGpuValuesParams.counterDataImageSize =
          this->counterDataImage.size();
      evaluateToGpuValuesParams.rangeIndex = rangeIndex;
      evaluateToGpuValuesParams.isolated = true;
      evaluateToGpuValuesParams.pMetricValues = &metricValue;
      NVPW_API_CALL(NVPW_MetricsEvaluator_EvaluateToGpuValues(
          &evaluateToGpuValuesParams));

      profiling_result.add_data(metricName, rangeName, metricValue);
    }
  }

  return profiling_result;
}

void GetRawMetricRequests(std::string chipName,
                          const std::vector<std::string>& metricNames,
                          std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
                          const uint8_t* pCounterAvailabilityImage) {
  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params
      calculateScratchBufferSizeParam = {
          NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  calculateScratchBufferSizeParam.pChipName = chipName.c_str();
  calculateScratchBufferSizeParam.pCounterAvailabilityImage =
      pCounterAvailabilityImage;
  NVPW_API_CALL(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(
      &calculateScratchBufferSizeParam));

  std::vector<uint8_t> scratchBuffer(
      calculateScratchBufferSizeParam.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams =
      {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
  metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
  metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
  metricEvaluatorInitializeParams.pChipName = chipName.c_str();
  metricEvaluatorInitializeParams.pCounterAvailabilityImage =
      pCounterAvailabilityImage;
  NVPW_API_CALL(
      NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
  NVPW_MetricsEvaluator* metricEvaluator =
      metricEvaluatorInitializeParams.pMetricsEvaluator;

  bool isolated = true;
  bool keepInstances = true;
  std::vector<const char*> rawMetricNames;
  for (auto& metricName : metricNames) {
    std::string reqName;
    ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
    keepInstances = true;
    NVPW_MetricEvalRequest metricEvalRequest;
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
        convertMetricToEvalRequest = {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
    convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
    convertMetricToEvalRequest.pMetricName = reqName.c_str();
    convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
    convertMetricToEvalRequest.metricEvalRequestStructSize =
        NVPW_MetricEvalRequest_STRUCT_SIZE;
    NVPW_API_CALL(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(
        &convertMetricToEvalRequest));

    std::vector<const char*> rawDependencies;
    NVPW_MetricsEvaluator_GetMetricRawDependencies_Params
        getMetricRawDependenciesParms = {
            NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
    getMetricRawDependenciesParms.pMetricsEvaluator = metricEvaluator;
    getMetricRawDependenciesParms.pMetricEvalRequests = &metricEvalRequest;
    getMetricRawDependenciesParms.numMetricEvalRequests = 1;
    getMetricRawDependenciesParms.metricEvalRequestStructSize =
        NVPW_MetricEvalRequest_STRUCT_SIZE;
    getMetricRawDependenciesParms.metricEvalRequestStrideSize =
        sizeof(NVPW_MetricEvalRequest);
    NVPW_API_CALL(NVPW_MetricsEvaluator_GetMetricRawDependencies(
        &getMetricRawDependenciesParms));
    rawDependencies.resize(getMetricRawDependenciesParms.numRawDependencies);
    getMetricRawDependenciesParms.ppRawDependencies = rawDependencies.data();
    NVPW_API_CALL(NVPW_MetricsEvaluator_GetMetricRawDependencies(
        &getMetricRawDependenciesParms));

    for (size_t i = 0; i < rawDependencies.size(); ++i) {
      rawMetricNames.push_back(rawDependencies[i]);
    }
  }

  for (auto& rawMetricName : rawMetricNames) {
    NVPA_RawMetricRequest metricRequest = {NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE};
    metricRequest.pMetricName = rawMetricName;
    metricRequest.isolated = isolated;
    metricRequest.keepInstances = keepInstances;
    rawMetricRequests.push_back(metricRequest);
  }

  NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = {
      NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
  metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
  NVPW_API_CALL(NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
}

void GetConfigImage(std::string chipName,
                    const std::vector<std::string>& metricNames,
                    std::vector<uint8_t>& configImage,
                    const uint8_t* pCounterAvailabilityImage) {
  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  GetRawMetricRequests(chipName, metricNames, rawMetricRequests,
                       pCounterAvailabilityImage);

  NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = {
      NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE};
  rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
  rawMetricsConfigCreateParams.pChipName = chipName.c_str();
  rawMetricsConfigCreateParams.pCounterAvailabilityImage =
      pCounterAvailabilityImage;
  NVPW_API_CALL(
      NVPW_CUDA_RawMetricsConfig_Create_V2(&rawMetricsConfigCreateParams));
  NVPA_RawMetricsConfig* pRawMetricsConfig =
      rawMetricsConfigCreateParams.pRawMetricsConfig;

  if (pCounterAvailabilityImage) {
    NVPW_RawMetricsConfig_SetCounterAvailability_Params
        setCounterAvailabilityParams = {
            NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
    setCounterAvailabilityParams.pRawMetricsConfig = pRawMetricsConfig;
    setCounterAvailabilityParams.pCounterAvailabilityImage =
        pCounterAvailabilityImage;
    NVPW_API_CALL(NVPW_RawMetricsConfig_SetCounterAvailability(
        &setCounterAvailabilityParams));
  }

  NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
      NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
  beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
  NVPW_API_CALL(NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

  NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
      NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
  addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
  addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  NVPW_API_CALL(NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

  NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
      NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
  endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
  NVPW_API_CALL(NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

  NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = {
      NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE};
  generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
  NVPW_API_CALL(
      NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams));

  NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = {
      NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE};
  getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
  getConfigImageParams.bytesAllocated = 0;
  getConfigImageParams.pBuffer = NULL;
  NVPW_API_CALL(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

  configImage.resize(getConfigImageParams.bytesCopied);
  getConfigImageParams.bytesAllocated = configImage.size();
  getConfigImageParams.pBuffer = configImage.data();
  NVPW_API_CALL(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

  NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
      NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
  rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
  NVPW_API_CALL(NVPW_RawMetricsConfig_Destroy(
      (NVPW_RawMetricsConfig_Destroy_Params*)&rawMetricsConfigDestroyParams));
}

void GetCounterDataPrefixImage(std::string chipName,
                               const std::vector<std::string>& metricNames,
                               std::vector<uint8_t>& counterDataImagePrefix,
                               const uint8_t* pCounterAvailabilityImage) {
  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  GetRawMetricRequests(chipName, metricNames, rawMetricRequests,
                       pCounterAvailabilityImage);

  NVPW_CUDA_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = {
      NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE};
  counterDataBuilderCreateParams.pChipName = chipName.c_str();
  counterDataBuilderCreateParams.pCounterAvailabilityImage =
      pCounterAvailabilityImage;
  NVPW_API_CALL(
      NVPW_CUDA_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

  NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = {
      NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE};
  addMetricsParams.pCounterDataBuilder =
      counterDataBuilderCreateParams.pCounterDataBuilder;
  addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  NVPW_API_CALL(NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams));

  size_t counterDataPrefixSize = 0;
  NVPW_CounterDataBuilder_GetCounterDataPrefix_Params
      getCounterDataPrefixParams = {
          NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE};
  getCounterDataPrefixParams.pCounterDataBuilder =
      counterDataBuilderCreateParams.pCounterDataBuilder;
  getCounterDataPrefixParams.bytesAllocated = 0;
  getCounterDataPrefixParams.pBuffer = NULL;
  NVPW_API_CALL(NVPW_CounterDataBuilder_GetCounterDataPrefix(
      &getCounterDataPrefixParams));

  counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);
  getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
  getCounterDataPrefixParams.pBuffer = counterDataImagePrefix.data();
  NVPW_API_CALL(NVPW_CounterDataBuilder_GetCounterDataPrefix(
      &getCounterDataPrefixParams));

  NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = {
      NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
  counterDataBuilderDestroyParams.pCounterDataBuilder =
      counterDataBuilderCreateParams.pCounterDataBuilder;
  NVPW_API_CALL(NVPW_CounterDataBuilder_Destroy((
      NVPW_CounterDataBuilder_Destroy_Params*)&counterDataBuilderDestroyParams));
}

void CreateCounterDataImage(std::vector<uint8_t>& counterDataImage,
                            std::vector<uint8_t>& counterDataScratchBuffer,
                            std::vector<uint8_t>& counterDataImagePrefix,
                            int numRanges) {
  CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
  counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
  counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
  counterDataImageOptions.maxNumRanges = numRanges;
  counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
  counterDataImageOptions.maxRangeNameLength = 64;

  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
      CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

  calculateSizeParams.pOptions = &counterDataImageOptions;
  calculateSizeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

  CUPTI_API_CALL(
      cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

  CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
      CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  initializeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  initializeParams.pOptions = &counterDataImageOptions;
  initializeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;

  counterDataImage.resize(calculateSizeParams.counterDataImageSize);
  initializeParams.pCounterDataImage = &counterDataImage[0];
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
      scratchBufferSizeParams = {
          CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  scratchBufferSizeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;
  scratchBufferSizeParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(
      &scratchBufferSizeParams));

  counterDataScratchBuffer.resize(
      scratchBufferSizeParams.counterDataScratchBufferSize);

  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
      initScratchBufferParams = {
          CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
  initScratchBufferParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;

  initScratchBufferParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  initScratchBufferParams.counterDataScratchBufferSize =
      scratchBufferSizeParams.counterDataScratchBufferSize;
  initScratchBufferParams.pCounterDataScratchBuffer =
      &counterDataScratchBuffer[0];

  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(
      &initScratchBufferParams));
}

inline void ParseMetricNameString(const std::string& metricName,
                                  std::string* reqName, bool* isolated,
                                  bool* keepInstances) {
  std::string& name = *reqName;
  name = metricName;
  if (name.empty()) {
    LOG(FATAL) << "name empty";
  }

  // boost program_options sometimes inserts a \n between the metric name and a
  // '&' at the end
  size_t pos = name.find('\n');
  if (pos != std::string::npos) {
    name.erase(pos, 1);
  }

  // trim whitespace
  while (name.back() == ' ') {
    name.pop_back();
    if (name.empty()) {
      LOG(FATAL) << "name empty";
    }
  }

  *keepInstances = false;
  if (name.back() == '+') {
    *keepInstances = true;
    name.pop_back();
    if (name.empty()) {
      LOG(FATAL) << "name empty";
    }
  }

  *isolated = true;
  if (name.back() == '$') {
    name.pop_back();
    if (name.empty()) {
      LOG(FATAL) << "name empty";
    }
  } else if (name.back() == '&') {
    *isolated = false;
    name.pop_back();
    if (name.empty()) {
      LOG(FATAL) << "name empty";
    }
  }
}

__global__ void power_kernel(float* in, float* out, int count) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count;
       idx += blockDim.x * gridDim.x) {
    out[idx] = in[idx] * in[idx];
  }
}

void launch_simple_cuda_kernel(int count) {
  float *in, *out;
  cudaMalloc(&in, count * sizeof(float));
  cudaMalloc(&out, count * sizeof(float));

  power_kernel<<<72 * 5, 128>>>(in, out, count);

  cudaFree(in);
  cudaFree(out);
}
}  // namespace profiler
}  // namespace tuning
}  // namespace mononn_engine