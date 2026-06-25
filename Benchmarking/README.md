# Edge ISP Benchmarking & Performance Comparison

This folder contains the benchmarking suite, results, and visualization assets comparing various Image Signal Processor (ISP) architectures on different edge hardware platforms. 

The evaluation compares two primary aspects of the ISPs:
1. **Execution Efficiency (Inference Time, Latency, & FPS)**: How quickly the ISP executes on edge hardware.
2. **Reconstruction Quality (PSNR & SSIM)**: The visual fidelity of the ISP's output RGB compared to the Ground Truth (GT) RGB.

---

## Supported ISP Architectures

1. **Fast-OpenISP**: A lightweight and fast open-source ISP pipeline.
   * Repository: [QiuJueqin/fast-openISP](https://github.com/QiuJueqin/fast-openISP)
2. **Infinite-ISP**: A comprehensive, high-quality, parameter-configurable software ISP.
   * Repository: [10x-Engineers/Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP/tree/main)
3. **SYENet (and SYENet-Slim)**: A neural-network-based lightweight ISP designed for edge platforms.
   * Repository: [sanechips-multimedia/syenet](https://github.com/sanechips-multimedia/syenet)
4. **SlimEYE**: A neural networks-based ISP model developed by the **VISPRO Lab** at the **Information Technology University (ITU)**.

---

## Hardware Platforms Evaluated
* **Raspberry Pi 3**
* **Raspberry Pi 5**
* **Intel Neural Compute Stick 2 (NCS2)**

---

## How to Run the Benchmarks

To replicate the results, you must clone the official repositories of the respective ISPs, copy our custom analysis scripts into them, and execute the tests as detailed below:

### 1. Fast-OpenISP Benchmark
1. Clone the repository:
   ```bash
   git clone https://github.com/QiuJueqin/fast-openISP.git
   cd fast-openISP
   ```
2. Copy the benchmark script [`scripts/fastopen_analysis.py`](scripts/fastopen_analysis.py) from this Benchmarking folder into the root directory of the cloned `fast-openISP` repository.
3. Configure your RAW input images and ground truth dataset paths inside the script, then execute:
   ```bash
   python fastopen_analysis.py
   ```
4. Collect the output `benchmark_metrics.csv` containing image names, latency, PSNR, and SSIM.

### 2. Infinite-ISP Benchmark
1. Clone the repository:
   ```bash
   git clone https://github.com/10x-Engineers/Infinite-ISP.git
   cd Infinite-ISP
   ```
2. Copy the benchmark script [`scripts/infinite_analysis.py`](scripts/infinite_analysis.py) from this Benchmarking folder into the root directory of the cloned `Infinite-ISP` repository.
3. Run the script:
   ```bash
   python infinite_analysis.py
   ```
4. Collect the output metrics and logs.

### 3. SYENet Benchmark
1. Clone the repository:
   ```bash
   git clone https://github.com/sanechips-multimedia/syenet.git
   cd syenet
   ```
2. Copy the benchmark script [`scripts/syenet_main.py`](scripts/syenet_main.py) from this Benchmarking folder into the root directory of the cloned `syenet` repository, renaming it to `main.py` (overwriting the default `main.py`).
3. Run the code using:
   ```bash
   python main.py -task test -model_task isp -device cpu
   ```
4. Collect the metrics output.

### 4. Intel Neural Compute Stick 2 (NCS2) Setup & Model Deployment

To deploy models (such as `SYENet-Slim`) onto the **Intel Neural Compute Stick 2 (NCS2)** (which runs on the Myriad X VPU), you must compile the model into the OpenVINO Intermediate Representation (IR) format and configure the system drivers to detect the USB hardware.

#### Model Conversion Pipeline
1. **Convert to ONNX:** Export your trained model (e.g., from PyTorch `.pth` weights) to standard ONNX format (`.onnx`).
2. **Convert to TensorFlow:** Convert the ONNX model to a TensorFlow model representation (if required for downstream compatibility).
3. **Compile to OpenVINO IR (.bin & .xml):** Use OpenVINO's Model Optimizer tool (`mo`) to convert the model into OpenVINO Intermediate Representation files:
   * **`.xml`**: Defines the neural network topology.
   * **`.bin`**: Contains the binary weights and biases of the model.

#### Environment & USB Driver Configuration (Windows)
For OpenVINO to run inference on the NCS2, it must detect the compute stick as a `MYRIAD` target device:

1. **Download OpenVINO:**
   Download and install OpenVINO version 2026.2.1 for Windows from the official package directory:
   * [🔗 OpenVINO 2026.2.1 Windows Package](https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.2.1/windows)
2. **Install Movidius USB Drivers:**
   * Plug in the Intel NCS2 into a USB 3.0 port.
   * Locate the drivers in your OpenVINO installation path (usually under `<openvino_install_dir>\redist\x64\drivers` or inside the package's drivers directory).
   * Right-click `Movidius_VSC.inf` and select **Install**.
   * *Troubleshooting:* If Windows does not recognize the device or it disconnects during load, download the **Zadig** utility and install the `WinUSB` driver for the device when it appears as `Movidius MyriadX` or `Loopback`.
3. **Verify Device Detection:**
   Run the following Python snippet to verify if OpenVINO detects the compute stick:
   ```bash
   python -c "import openvino as ov; core = ov.Core(); print(core.available_devices)"
   ```
   Ensure `'MYRIAD'` appears in the printed list of available devices.

---

## Benchmark Results

The aggregated metrics across different model-device configurations are summarized in the table below:

| Device | Model | Avg Latency (ms) | Avg PSNR (dB) | Avg SSIM | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Intel Neural Compute stick2** | SYENet-Slim | 21.8 | 28.12 | 0.8727 | 45.9 |
| **raspberry-pi3** | Fastopen | 228.9 | 18.02 | 0.7169 | 4.4 |
| **raspberry-pi3** | Infinite | 5147.1 | 20.85 | 0.7659 | 0.2 |
| **raspberry-pi3** | SYENet | 3518.7 | 25.10 | 0.8592 | 0.3 |
| **raspberry-pi3** | SYENet-Slim | 98.8 | 25.10 | 0.8592 | 10.1 |
| **raspberry-pi5** | Fastopen | 21.3 | 18.02 | 0.7169 | 46.9 |
| **raspberry-pi5** | Infinite | 456.8 | 20.85 | 0.7659 | 2.2 |
| **raspberry-pi5** | SYENet | 520.1 | 25.10 | 0.8592 | 1.9 |
| **raspberry-pi5** | SYENet-Slim | 24.0 | 25.10 | 0.8592 | 41.6 |

---

## Performance Visualizations

### 1. General Metrics (Image Quality)
Since PSNR and SSIM depend entirely on the model's algorithms and not the execution hardware, they are identical across all devices.

#### PSNR Comparison (Higher is Better)
![PSNR Comparison](visualization/psnr_comparison.png)

#### SSIM Comparison (Higher is Better)
![SSIM Comparison](visualization/ssim_comparison.png)

### 2. Device-Specific Metrics (FPS Speed)
Frame rate is highly dependent on hardware capabilities. Below are the speed comparisons for each target platform:

#### Raspberry Pi 3 FPS Comparison
![Raspberry Pi 3 FPS](visualization/fps_pi3.png)

#### Raspberry Pi 5 FPS Comparison
![Raspberry Pi 5 FPS](visualization/fps_pi5.png)

#### Intel NCS2 FPS Comparison
![Intel NCS2 FPS](visualization/fps_ncs2.png)

---

## Additional Resources

* **Complete Results & Output Sample Images:** You can access high-resolution output sample comparisons, full log files, and intermediate charts at our shared drive folder:
  [🔗 Google Drive: Benchmarking Full Results](https://drive.google.com/drive/folders/YOUR_GOOGLE_DRIVE_FOLDER_LINK_HERE) *(Replace this link once available)*