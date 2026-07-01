# 🚀 Computer Vision Production Inference Engine

Welcome to the **Computer Vision Production Inference Engine** project! 

This repository implements a containerized, high-performance inference stack using **NVIDIA Triton Inference Server**. It serves four model variations for **ResNet50** (Image Classification) and **YOLO** (Object Detection) using both ONNX and PyTorch backends. The setup includes dynamic batching optimizations and a complete Prometheus/Grafana monitoring dashboard.

---

## 📋 System Overview

Before starting, it is helpful to understand the architecture:
*   **Inference Server (Triton)**: Hosts the model files and serves HTTP/gRPC requests. Models are loaded/unloaded dynamically via API to optimize VRAM.
*   **Observability Stack**: **Prometheus** scrapes inference performance metrics from Triton; **Grafana** visualizes real-time metrics (throughput, latency, GPU utilization).
*   **Client**: A Python script that preprocesses sample images, queries the Triton server, post-processes the outputs, and logs inference results.

---

## 🛠️ Prerequisites

To run this project, make sure your machine has:
1.  **Docker** installed and running.
2.  **Windows Subsystem for Linux (WSL)** (if on Windows) or a Linux/macOS environment.
3.  **Python 3.8+** installed.
4.  **`make`** command utility installed.
5.  *Optional but Recommended*: **NVIDIA GPU** with CUDA drivers. If no GPU is found, the system will automatically fall back to CPU mode.

---

## 🚀 Step-by-Step Execution Guide

If you are running this project for the first time, follow these steps:

### 1. Set Up the Environment
Create a localized Python virtual environment and install the required dependencies:
```bash
make setup
```
*What this does:* Creates a virtual environment in `/opt/ml-dev-ops-venv` (or locally) and installs deep learning tools (`torch`, `onnx`, `ultralytics`, etc.) for exporting models and running the client.

#### Configure Environment (Optional)
You can customize container names, ports, and model directories by copying `.env.example` to `.env` and editing the values:
```bash
cp .env.example .env
```
Both the `Makefile` and `deployment/run_triton.sh` script will automatically load and apply these environment variables.

### 2. Export the Deep Learning Models
Download pretrained weights and export them into formats optimized for Triton (ONNX and TorchScript):
```bash
make export-models
```
*What this does:* Downloads ResNet50 and YOLO11n weights, exports them with matching input/output layers and dynamic batching configurations, and places them into the versioned folders inside `models/`. It also automatically handles ONNX version compatibility.

### 3. Build the Triton Server Image
Build a thin, customized Triton Docker image wrapping the official NVIDIA runtime:
```bash
make build
```

### 4. Start the Service Stack
Launch the containers (Triton server, Prometheus, Grafana) in the background:
```bash
make up
```

### 5. Load the Models into Memory
Trigger Triton to load the models:
```bash
make load-models
```
*Why this is needed:* Triton runs in `explicit` control mode. It starts with zero models in VRAM to prevent Out-Of-Memory (OOM) crashes. Models must be explicitly loaded to active memory.

### 6. Run Inference and Verify
Run the test suite to execute batch predictions on sample images:
```bash
make infer
```

#### Run Unit Tests
To run the client unit tests (verifying preprocessing, postprocessing, and logging):
```bash
make test
```

---

## 💡 Quick Start (All-in-One)
Alternatively, you can run the entire pipeline from scratch with a single command:
```bash
make quickstart
```

---

## 🔍 How to Check and Verify the Project

Once you have started the project, here is how you check if it is running correctly:

### A. Check Container States
Run the status check to see active containers, ports, and health:
```bash
make status
```
You should see three running containers:
*   `triton-cv` (running on ports `8000`, `8001`, `8002`)
*   `prometheus-mon` (running on port `9090`)
*   `grafana-viz` (running on port `3000`)

### B. Verify Model States
To check if the models loaded successfully, run:
```bash
curl -s -X POST localhost:8000/v2/repository/index
```
You should get a JSON response indicating all 4 models are in the `"READY"` state:
```json
[
  {"name":"resnet50_notonnx","version":"1","state":"READY"},
  {"name":"resnet50_onnx","version":"1","state":"READY"},
  {"name":"yolo_notonnx","version":"1","state":"READY"},
  {"name":"yolo_onnx","version":"1","state":"READY"}
]
```

### C. Verify Inference Logs
The test outputs are saved in your workspace folder. Open the file to verify results:
```bash
cat logs/inference_history.csv
```
You should see rows indicating the model name, sample image tested, model classification/detection, confidence score, and latency.

### D. Check Real-Time Observability
1.  **Prometheus**: Open [http://localhost:9090](http://localhost:9090) in your browser. Go to **Status > Targets** and verify `triton-metrics` is showing status **UP**.
2.  **Grafana**: Open [http://localhost:3000](http://localhost:3000) (Login using `admin` / `admin`).
    *   Go to **Data Sources > Add Data Source > Prometheus**.
    *   Set the Prometheus server URL to: `http://host.docker.internal:9090` (or `http://localhost:9090`).
    *   Go to **Dashboards > Import** and import `monitoring/grafana-dashboard.json`.
    *   This will display real-time graphs showing inference throughput (req/sec), queue latency, and resource utilization.

---

## 🕹️ Model Control (Managing Memory)

You can load and unload individual models manually to free up hardware resources:
*   **Unload Model** (frees up VRAM/RAM immediately):
    ```bash
    make unload-resnet50-onnx
    ```
*   **Load Model**:
    ```bash
    make load-resnet50-onnx
    ```

---

## 🧹 Cleanup
To stop the services and clean up generated model binary files when you are done:
```bash
make clean
```
This stops the containers, deletes them, and removes exported model weights.
