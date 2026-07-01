#!/bin/bash
set -e

# Load environment variables from .env if present
if [ -f .env ]; then
  echo "--- Loading .env file ---"
  export $(grep -v '^#' .env | xargs)
fi

NAME="${NAME:-triton-cv}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-$(pwd)/models}"
PROM_CONTAINER="${PROM_CONTAINER:-prometheus-mon}"
GRAF_CONTAINER="${GRAF_CONTAINER:-grafana-viz}"

HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"
PROM_PORT="${PROM_PORT:-9090}"
GRAF_PORT="${GRAF_PORT:-3000}"

echo "================================================================"
echo "🛑 STEP 0: Cleaning up old containers..."
echo "================================================================"
docker stop $NAME $PROM_CONTAINER $GRAF_CONTAINER 2>/dev/null || true
docker rm $NAME $PROM_CONTAINER $GRAF_CONTAINER 2>/dev/null || true


echo "================================================================"
echo "🛠️ STEP 1: Pulling Official NVIDIA Triton Image"
echo "================================================================"
docker build -t $NAME -f deployment/docker/Dockerfile .

echo "================================================================"
echo "🔥 STEP 2: Launching Prometheus & Grafana"
echo "================================================================"


# Launch Prometheus 
docker run -d \
  --name $PROM_CONTAINER \
  -p $PROM_PORT:9090 \
  -v "$(pwd)/monitoring:/etc/prometheus" \
  prom/prometheus

# Launch Grafana
docker run -d \
  --name $GRAF_CONTAINER \
  -p $GRAF_PORT:3000 \
  grafana/grafana

echo "✅ Monitoring stack is up at http://localhost:$PROM_PORT and http://localhost:$GRAF_PORT"

echo "================================================================"
echo "🔥 STEP 3: Launching Triton Inference Server"
echo "================================================================"


# Check if nvidia-smi command is available (indicates NVIDIA drivers are installed)
if command -v nvidia-smi &> /dev/null; then
  # Check if a GPU is actually detected by the driver
  if nvidia-smi -L &> /dev/null; then
    echo "🚀 NVIDIA GPU detected! Enabling GPU mode."
    GPU_FLAG="--gpus all"
  else
    echo "⚠️ NVIDIA driver found, but no GPU detected. Falling back to CPU."
    GPU_FLAG=""
  fi
else
  echo "ℹ️ No NVIDIA drivers found. Running in CPU mode."
  GPU_FLAG=""
fi

docker run \
  --name $NAME \
  $GPU_FLAG \
  -p $HTTP_PORT:8000 \
  -p $GRPC_PORT:8001 \
  -p $METRICS_PORT:8002 \
  -v "$MODEL_REPOSITORY:/models" \
  $NAME \
  --model-repository=/models \
  --model-control-mode=explicit \
  --allow-gpu-metrics=false \
  --exit-on-error=false