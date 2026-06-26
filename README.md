<div align="center">
  <h1>Implementation of ISPs on Edge Devices</h1>
  <p><strong>A Full-Stack Hardware and Software Engineering Research Project</strong></p>

  <!-- Badges -->
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" />
  <img src="https://img.shields.io/badge/FPGA-Verilog-blue?style=for-the-badge" />
</div>

<br>

> **Information Technology University (ITU)**  
> **Department of Computer & Software Engineering**  
> **Team:** Muhammad Arham, Muhammad Moiz Ahmad, Alishba Zahid  
> **Supervisors:** Dr. Rehan Hafiz, Dr. Rehan Ahmed  

---

## 📖 Project Overview

This repository contains the source code, custom hardware designs, and comprehensive benchmarking suites for deploying advanced **Image Signal Processors (ISPs)** onto resource-constrained edge devices. 

Our research systematically evaluates the strict technical trade-offs between execution speed, hardware power constraints, and visual image quality. We bridge the gap between high-level Python AI training and pure silicon hardware design, comparing Classical (rule-based) ISPs against Learned (Neural Network-driven) ISPs.

### 🌟 Novelty & Academic Contribution
While deploying Neural ISPs on edge hardware is an active research field, the vast majority of existing solutions rely on high-level synthesis (HLS) tools to auto-compile models into hardware. **Our project takes a highly specialized, full-stack approach:**
1. **Custom Silicon Logic:** We bypassed auto-compilers to manually engineer a custom FSM-driven datapath operating at the INT8 MAC level via a single DSP48E1 unit on the FPGA. 
2. **The Full-Stack Ecosystem:** We built an end-to-end pipeline that spans from low-level silicon FPGA design, to manual CUDA C GPU acceleration (641+ FPS), all the way up to engineering a mathematical zero-shot domain adaptation pipeline for a modern consumer smartphone (Pixel 7a).

---

## 🚀 Key Achievements

- **Real-Time AI Processing:** Successfully translated Python-based neural ISPs into highly optimized CUDA C, achieving up to **641 FPS** on an NVIDIA Quadro GPU.
- **Hardware-Level Execution:** Engineered a custom FPGA datapath architecture (Nexys-4 Artix-7) to execute Neural ISPs without standard memory bottlenecks.
- **Zero-Shot Domain Adaptation:** Built a mathematical pipeline to adapt a Mediatek-trained AI model to process raw 14-bit data from a **Google Pixel 7a** smartphone with zero network retraining, achieving **16.32 dB PSNR**.

---

## 🏛️ Repository Architecture & Project Pillars

This project is divided into four major technical domains. Please navigate to the respective folders for source code, deployment instructions, and detailed metric logs.

### 1. [Edge Hardware Benchmarking](./Benchmarking)
We deployed and benchmarked four different ISPs (FastOpenISP, InfiniteISP, SYENet, and SlimEYE) across multiple edge platforms.
* **Hardware Evaluated:** Raspberry Pi 3, Raspberry Pi 5, Intel Neural Compute Stick 2 (NCS2).
* **Metrics Analyzed:** Precise latency (FPS) vs visual reconstruction quality (PSNR/SSIM).

### 2. [High-Performance CUDA GPU Acceleration](./Cuda%20C%20implementation%20of%20SYENet%20Slim%20and%20SYENet)
To address the heavy computational overhead of neural ISPs, we manually translated the lightweight SYENet-Slim architecture from Python into highly optimized **CUDA C**.
* **Deployment:** Natively running on an NVIDIA Quadro T1000 GPU (896 CUDA cores).
* **Performance:** Massive acceleration (641 FPS) while maintaining pristine image quality (24.85 dB PSNR).

### 3. [Custom FPGA Hardware Design](./Hardware%20Implementation%20of%20SYENet%20Slim%20on%20FPGA(NEXYS%204%20-Artrix%207))
We engineered a custom hardware architecture for the SYENet Slim pipeline designed to operate within strict embedded constraints.
* **Device:** Nexys-4 / XC7A100T (Artix-7 FPGA).
* **Datapath:** FSM-driven logic executing one INT8 MAC per clock cycle via a single DSP48E1 unit.

### 4. [Cross-Sensor Domain Adaptation](./Cross%20Sensor%20Testing)
AI models are typically highly sensor-specific. We proved the generalizability of SYENet by capturing a proprietary, real-world raw dataset using a **Google Pixel 7a** smartphone.
* **The Challenge:** Pixel 7a uses a 14-bit depth and 0 black-level, whereas the model expects 12-bit depth and a 344 black-level.
* **The Solution:** We engineered a mathematical zero-shot preprocessing pipeline (bit-scaling, black-level injection, and color nudges) to seamlessly translate the foreign hardware physics into the model's native profile.

---

## 💾 Model Architectures Evaluated
* **Classical ISPs:** Fast-OpenISP, InfiniteISP
* **Learned ISPs:** SYENet, SYENet-Slim, SlimEYE (VISPRO Lab, ITU)

---

## 🔗 Additional Resources
* **Datasets & Output Images:** High-resolution benchmarking results and the full Google Pixel 7a custom mobile dataset are hosted externally due to size constraints. You can access them here: 
  * [🔗 Google Drive: SYENET_Pixel7a_Custom_Dataset](https://drive.google.com/drive/folders/1XHF59HnqhQiPCTLFdz6z6k-wkiPBaQ70?usp=sharing)
  * [🔗 Google Drive: Benchmarking Full Results](https://drive.google.com/drive/folders/1DdzcnSLQPj7ml81z7FJ1bOuxpQtP6J1o?usp=drive_link) 
* **Base Architectures:** The original ISP frameworks evaluated in this research can be found at their official repositories:
  * [SYENet (Learned ISP)](https://github.com/sanechips-multimedia/syenet)
  * [Fast-OpenISP (Classical ISP)](https://github.com/QiuJueqin/fast-openISP)
  * [Infinite-ISP (Classical ISP)](https://github.com/10x-Engineers/Infinite-ISP)

<br>

*Special thanks to our supervisors Sir Rehan Hafiz and Sir Rehan Ahmad, as well as Engr. Nauman Latif (Ph.D. fellow), for their continuous guidance and support throughout this Final Year Project.*
