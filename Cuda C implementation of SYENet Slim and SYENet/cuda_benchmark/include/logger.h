#pragma once
#include <NvInfer.h>
#include <iostream>
#include <string>

class Logger : public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity severity;
    Logger(nvinfer1::ILogger::Severity sev = nvinfer1::ILogger::Severity::kWARNING)
        : severity(sev) {}
    void log(nvinfer1::ILogger::Severity sev, const char* msg) noexcept override {
        if (sev <= severity) {
            const char* tag = "[TRT] ";
            switch (sev) {
                case nvinfer1::ILogger::Severity::kERROR:   tag = "[TRT ERROR] "; break;
                case nvinfer1::ILogger::Severity::kWARNING: tag = "[TRT WARN]  "; break;
                case nvinfer1::ILogger::Severity::kINFO:    tag = "[TRT INFO]  "; break;
                default: break;
            }
            std::cerr << tag << msg << "\n";
        }
    }
};

#define LOG(msg) std::cout << "[BENCH] " << msg << "\n"
#define LOG_ERR(msg) std::cerr << "[ERROR] " << msg << "\n"
