#!/bin/bash
set -e

NAME="triton-cv"
MODEL_REPOSITORY="$(pwd)/models"
PROM_CONTAINER="prometheus-mon"
GRAF_CONTAINER="grafana-viz"

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
  -p 9090:9090 \
  -v "$(pwd)/monitoring:/etc/prometheus" \
  prom/prometheus

# Launch Grafana
docker run -d \
  --name $GRAF_CONTAINER \
  -p 3000:3000 \
  grafana/grafana

echo "✅ Monitoring stack is up at http://localhost:9090 and http://localhost:3000"

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
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$MODEL_REPOSITORY:/models" \
  $NAME \
  --model-repository=/models \
  --model-control-mode=explicit \
  --allow-gpu-metrics=false \
  --exit-on-error=false