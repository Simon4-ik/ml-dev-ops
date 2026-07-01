# =============================================================================
# Computer Vision Production Inference Engine - Makefile
# =============================================================================
# Usage:
#   make help           Show all available commands
#   make setup          Install Python dependencies
#   make pull-base      Pull official NVIDIA Triton base image
#   make export-models  Export YOLO + ResNet50 models to ONNX / TorchScript
#   make build          Build the Triton Docker image
#   make up             Launch the full stack (Triton + Prometheus + Grafana)
#   make load-models    Load all 4 models via Triton REST API
#   make infer          Run inference / regression test suite
#   make down           Stop and remove all containers
#   make logs           Tail Triton container logs
#   make status         Show running containers and endpoints
#   make clean          Remove containers + generated model files
#   make quickstart     Full end-to-end: setup + export + build + up + load + infer
# =============================================================================

-include .env

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_NAME     ?= triton-cv
TRITON_NAME    ?= triton-cv
PROM_NAME      ?= prometheus-mon
GRAF_NAME      ?= grafana-viz

MODELS_DIR     ?= $(shell pwd)/models
MONITORING_DIR ?= $(shell pwd)/monitoring

HTTP_PORT      ?= 8000
GRPC_PORT      ?= 8001
METRICS_PORT   ?= 8002
PROM_PORT      ?= 9090
GRAF_PORT      ?= 3000

TRITON_URL     ?= localhost:$(HTTP_PORT)
export TRITON_URL

# Virtual environment stored in WSL native filesystem (avoids /mnt/c space limits)
# Override with: make setup VENV=/path/to/venv
VENV           := /opt/ml-dev-ops-venv
VENV_PYTHON    := $(VENV)/bin/python3
VENV_PIP       := $(VENV)/bin/pip

# Auto-detect NVIDIA GPU (Linux only; CPU mode on Windows / Mac)
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)

ifeq ($(UNAME_S),Linux)
  GPU_AVAILABLE := $(shell nvidia-smi -L > /dev/null 2>&1 && echo yes || echo no)
else
  GPU_AVAILABLE := no
endif

ifeq ($(GPU_AVAILABLE),yes)
  GPU_FLAG := --gpus all
  $(info [INFO] NVIDIA GPU detected -- running in GPU mode)
else
  GPU_FLAG :=
  $(info [INFO] No NVIDIA GPU -- running in CPU mode)
endif

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: help setup pull-base export-models export-yolo export-resnet \
        build up down \
        load-models unload-models \
        load-resnet50-onnx load-resnet50-notonnx \
        load-yolo-onnx load-yolo-notonnx \
        unload-resnet50-onnx unload-resnet50-notonnx \
        unload-yolo-onnx unload-yolo-notonnx \
        infer benchmark test logs status clean quickstart

# ===========================================================================
# HELP
# ===========================================================================
help:
	@echo ""
	@echo "====================================================================="
	@echo "  CV Inference Engine -- Make Commands"
	@echo "====================================================================="
	@echo ""
	@echo "  Setup and Preparation:"
	@echo "    make setup              Install Python dependencies"
	@echo "    make pull-base          Pull official NVIDIA Triton base image"
	@echo "    make export-models      Export all models (YOLO + ResNet50)"
	@echo "    make export-yolo        Export YOLO11n to ONNX + TorchScript"
	@echo "    make export-resnet      Export ResNet50 via export_resnet.py"
	@echo ""
	@echo "  Docker and Stack:"
	@echo "    make build              Build the triton-cv Docker image"
	@echo "    make up                 Launch Triton + Prometheus + Grafana"
	@echo "    make down               Stop and remove all containers"
	@echo ""
	@echo "  Model Management (requires Triton running):"
	@echo "    make load-models           Load all 4 models"
	@echo "    make unload-models         Unload all 4 models (frees VRAM)"
	@echo "    make load-resnet50-onnx    Load resnet50_onnx"
	@echo "    make load-resnet50-notonnx Load resnet50_notonnx"
	@echo "    make load-yolo-onnx        Load yolo_onnx"
	@echo "    make load-yolo-notonnx     Load yolo_notonnx"
	@echo ""
	@echo "  Inference and Testing:"
	@echo "    make infer              Run regression test suite + 50-iter benchmark"
	@echo ""
	@echo "  Observability:"
	@echo "    make logs               Tail Triton container logs"
	@echo "    make status             Show running containers and endpoints"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean              Remove containers + exported model files"
	@echo ""
	@echo "  Quick Start (end-to-end):"
	@echo "    make quickstart         setup + export + build + up + load + infer"
	@echo ""

# ===========================================================================
# SETUP
# ===========================================================================
venv:
	@echo "--- Creating virtual environment at $(VENV) ---"
	python3 -m venv $(VENV)
	@echo "Venv created."

setup: venv
	@echo "--- Installing Python dependencies (CPU-only torch to save space) ---"
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(VENV_PIP) install onnx onnxscript ultralytics tritonclient[http] Pillow numpy requests opencv-python
	@echo "Done.  Venv at $(VENV)"

pull-base:
	@echo "--- Pulling NVIDIA Triton base image ---"
	docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# ===========================================================================
# MODEL EXPORT
# ===========================================================================
export-yolo:
	@echo "--- Exporting YOLO11n to ONNX (dynamic batching) ---"
	$(VENV_PYTHON) -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); m.export(format='onnx', dynamic=True, opset=17)"
	mkdir -p models/yolo_onnx/1
	mv yolo11n.onnx models/yolo_onnx/1/model.onnx
	@echo "--- Exporting YOLO11n to TorchScript ---"
	$(VENV_PYTHON) -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); m.export(format='torchscript')"
	mkdir -p models/yolo_notonnx/1
	mv yolo11n.torchscript models/yolo_notonnx/1/model.pt
	@echo "YOLO models exported."

export-resnet:
	@echo "--- Exporting ResNet50 ---"
	$(VENV_PYTHON) deployment/export_resnet.py
	@echo "ResNet50 models exported."

export-models: export-yolo export-resnet
	@echo "All models exported."

# ===========================================================================
# DOCKER / STACK
# ===========================================================================
build:
	@echo "--- Building Docker image: $(IMAGE_NAME) ---"
	docker build -t $(IMAGE_NAME) -f deployment/docker/Dockerfile .
	@echo "Image built."

up:
	@echo "--- Cleaning up old containers ---"
	-docker stop $(TRITON_NAME) $(PROM_NAME) $(GRAF_NAME) 2>/dev/null
	-docker rm   $(TRITON_NAME) $(PROM_NAME) $(GRAF_NAME) 2>/dev/null
	@echo "--- Starting Prometheus ---"
	docker run -d \
	  --name $(PROM_NAME) \
	  -p $(PROM_PORT):9090 \
	  -v "$(MONITORING_DIR):/etc/prometheus" \
	  prom/prometheus
	@echo "Prometheus -> http://localhost:$(PROM_PORT)"
	@echo "--- Starting Grafana ---"
	docker run -d \
	  --name $(GRAF_NAME) \
	  -p $(GRAF_PORT):3000 \
	  grafana/grafana
	@echo "Grafana -> http://localhost:$(GRAF_PORT)  (login: admin / admin)"
	@echo "--- Starting Triton Inference Server ---"
	docker run -d \
	  --name $(TRITON_NAME) \
	  $(GPU_FLAG) \
	  -p $(HTTP_PORT):8000 \
	  -p $(GRPC_PORT):8001 \
	  -p $(METRICS_PORT):8002 \
	  -v "$(MODELS_DIR):/models" \
	  $(IMAGE_NAME) \
	  --model-repository=/models \
	  --model-control-mode=explicit \
	  --allow-gpu-metrics=false \
	  --exit-on-error=false
	@echo "Triton HTTP  -> http://localhost:$(HTTP_PORT)"
	@echo "Triton gRPC  -> localhost:$(GRPC_PORT)"
	@echo "Metrics      -> http://localhost:$(METRICS_PORT)/metrics"
	@echo ""
	@echo "Stack is up!  Next:  make load-models   then:  make infer"

down:
	@echo "--- Stopping and removing containers ---"
	-docker stop $(TRITON_NAME) $(PROM_NAME) $(GRAF_NAME) 2>/dev/null
	-docker rm   $(TRITON_NAME) $(PROM_NAME) $(GRAF_NAME) 2>/dev/null
	@echo "Done."

# ===========================================================================
# MODEL MANAGEMENT  (explicit load / unload via Triton REST API)
# ===========================================================================
load-resnet50-onnx:
	@echo "Loading resnet50_onnx..."
	curl -s -X POST $(TRITON_URL)/v2/repository/models/resnet50_onnx/load
	@echo ""

load-resnet50-notonnx:
	@echo "Loading resnet50_notonnx..."
	curl -s -X POST $(TRITON_URL)/v2/repository/models/resnet50_notonnx/load
	@echo ""

load-yolo-onnx:
	@echo "Loading yolo_onnx..."
	curl -s -X POST $(TRITON_URL)/v2/repository/models/yolo_onnx/load
	@echo ""

load-yolo-notonnx:
	@echo "Loading yolo_notonnx..."
	curl -s -X POST $(TRITON_URL)/v2/repository/models/yolo_notonnx/load
	@echo ""

load-models: load-resnet50-onnx load-resnet50-notonnx load-yolo-onnx load-yolo-notonnx
	@echo "All 4 models loaded."

unload-resnet50-onnx:
	@echo "Unloading resnet50_onnx..."
	curl -s -X POST $(TRITON_URL)/v2/repository/models/resnet50_onnx/unload
	@echo ""

unload-resnet50-notonnx:
	@echo "Unloading resnet50_notonnx..."
	curl -s -X POST $(TRITON_URL)/v2/repository/models/resnet50_notonnx/unload
	@echo ""

unload-yolo-onnx:
	@echo "Unloading yolo_onnx..."
	curl -s -X POST $(TRITON_URL)/v2/repository/models/yolo_onnx/unload
	@echo ""

unload-yolo-notonnx:
	@echo "Unloading yolo_notonnx..."
	curl -s -X POST $(TRITON_URL)/v2/repository/models/yolo_notonnx/unload
	@echo ""

unload-models: unload-resnet50-onnx unload-resnet50-notonnx unload-yolo-onnx unload-yolo-notonnx
	@echo "All 4 models unloaded."

# ===========================================================================
# INFERENCE
# ===========================================================================
infer:
	@echo "--- Running inference regression test + benchmark ---"
	$(VENV_PYTHON) client/inference_client.py
	@echo "Done. Results saved to logs/inference_history.csv"

benchmark: infer

test:
	@echo "--- Running unit tests ---"
	$(VENV_PYTHON) -m unittest discover -s client -p "test_*.py"

# ===========================================================================
# OBSERVABILITY
# ===========================================================================
logs:
	docker logs -f $(TRITON_NAME)

status:
	@echo ""
	@echo "=== Running Containers ==="
	@docker ps --filter name=$(TRITON_NAME) \
	           --filter name=$(PROM_NAME) \
	           --filter name=$(GRAF_NAME) \
	           --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "(none running)"
	@echo ""
	@echo "=== Service Endpoints ==="
	@echo "  Triton HTTP    -> http://localhost:$(HTTP_PORT)"
	@echo "  Triton gRPC    -> localhost:$(GRPC_PORT)"
	@echo "  Triton Metrics -> http://localhost:$(METRICS_PORT)/metrics"
	@echo "  Prometheus     -> http://localhost:$(PROM_PORT)"
	@echo "  Grafana        -> http://localhost:$(GRAF_PORT)  (admin/admin)"
	@echo ""

# ===========================================================================
# CLEANUP
# ===========================================================================
clean: down
	@echo "--- Removing exported model artifacts ---"
	-rm -f models/yolo_onnx/1/model.onnx
	-rm -f models/yolo_notonnx/1/model.pt
	-rm -f models/resnet50_onnx/1/model.onnx
	-rm -f models/resnet50_notonnx/1/model.pt
	-rm -rf $(VENV)
	@echo "Clean complete."

# ===========================================================================
# QUICK START  (full end-to-end convenience target)
# ===========================================================================
quickstart: setup export-models build up
	@echo "Waiting 10 seconds for Triton to initialise..."
	sleep 10
	$(MAKE) load-models
	$(MAKE) infer
	@echo "Quick start complete! Check logs/inference_history.csv for results."

