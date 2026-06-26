// TensorRT engine: build from ONNX and run inference with async CUDA streams.
#include "trt_engine.h"
#include <NvOnnxParser.h>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <numeric>

TRTEngine::TRTEngine()
    : m_logger(nvinfer1::ILogger::Severity::kWARNING) {}

TRTEngine::~TRTEngine() {
    for (auto* p : m_bindings) {
        if (p) cudaFree(p);
    }
}

// ── Build from ONNX ───────────────────────────────────────────────────────────
bool TRTEngine::buildFromONNX(const std::string& onnx_path,
                               const std::string& engine_path,
                               Precision prec,
                               size_t workspace_mb)
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(m_logger));
    if (!builder) { LOG_ERR("createInferBuilder failed"); return false; }

    const auto flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(flags));

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, m_logger));

    if (!parser->parseFromFile(onnx_path.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        LOG_ERR("ONNX parse failed: " << onnx_path);
        for (int i = 0; i < parser->getNbErrors(); i++)
            LOG_ERR("  " << parser->getError(i)->desc());
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               workspace_mb * 1024 * 1024);

    if (prec == Precision::FP16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        LOG("FP16 mode enabled");
    } else if (prec == Precision::INT8 && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        LOG("INT8 mode enabled");
    } else if (prec != Precision::FP32) {
        LOG("Requested precision not supported, falling back to FP32");
    }

    LOG("Building TensorRT engine from " << onnx_path << " ...");
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized || serialized->size() == 0) {
        LOG_ERR("Engine serialization failed");
        return false;
    }

    // Save to disk
    std::ofstream fout(engine_path, std::ios::binary);
    fout.write(static_cast<const char*>(serialized->data()), serialized->size());
    LOG("Engine saved: " << engine_path << "  (" << serialized->size()/1024 << " KB)");

    return loadEngine(engine_path);
}

// ── Load serialised engine ────────────────────────────────────────────────────
bool TRTEngine::loadEngine(const std::string& engine_path)
{
    std::ifstream fin(engine_path, std::ios::binary | std::ios::ate);
    if (!fin) { LOG_ERR("Cannot open engine: " << engine_path); return false; }
    size_t sz = fin.tellg(); fin.seekg(0);
    std::vector<char> buf(sz);
    fin.read(buf.data(), sz);

    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    m_engine.reset(m_runtime->deserializeCudaEngine(buf.data(), sz));
    if (!m_engine) { LOG_ERR("deserializeCudaEngine failed"); return false; }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) { LOG_ERR("createExecutionContext failed"); return false; }

    m_name = engine_path;
    setupBindings();
    LOG("Engine loaded: " << engine_path);
    return true;
}

// ── Allocate device buffers for all I/O tensors ───────────────────────────────
void TRTEngine::setupBindings()
{
    int nb = m_engine->getNbIOTensors();
    m_bindings.resize(nb, nullptr);

    for (int i = 0; i < nb; i++) {
        const char* name = m_engine->getIOTensorName(i);
        auto dims   = m_engine->getTensorShape(name);
        int64_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) {
            if (dims.d[d] > 0) vol *= dims.d[d];
        }
        size_t bytes = vol * sizeof(float);

        cudaMalloc(&m_bindings[i], bytes);
        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            m_inputSize = vol;
        else
            m_outputSize = vol;
    }
}

// ── Async inference ───────────────────────────────────────────────────────────
bool TRTEngine::infer(void* d_input, void* d_output, cudaStream_t stream)
{
    // Copy caller's device pointers into binding slots
    int nb = m_engine->getNbIOTensors();
    for (int i = 0; i < nb; i++) {
        const char* name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            m_context->setTensorAddress(name, d_input);
        else
            m_context->setTensorAddress(name, d_output);
    }
    return m_context->enqueueV3(stream);
}
