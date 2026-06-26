#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include "logger.h"

// ── Precision mode ────────────────────────────────────────────────────────────
enum class Precision { FP32, FP16, INT8 };

// ── TensorRT engine wrapper ───────────────────────────────────────────────────
class TRTEngine {
public:
    TRTEngine();
    ~TRTEngine();

    // Build from ONNX; serialise to engine_path
    bool buildFromONNX(const std::string& onnx_path,
                       const std::string& engine_path,
                       Precision prec,
                       size_t workspace_mb = 512);

    // Load pre-built serialised engine from disk
    bool loadEngine(const std::string& engine_path);

    // Run inference; input/output are device pointers (float32)
    bool infer(void* d_input, void* d_output, cudaStream_t stream = 0);

    int64_t inputSize()  const { return m_inputSize; }
    int64_t outputSize() const { return m_outputSize; }
    std::string name()   const { return m_name; }

private:
    Logger                                    m_logger;
    std::unique_ptr<nvinfer1::IRuntime>       m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>    m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    std::vector<void*>                        m_bindings;

    int64_t m_inputSize  = 0;
    int64_t m_outputSize = 0;
    std::string m_name;

    void setupBindings();
};
