# Use the official NVIDIA Triton Inference Server image directly.
# This includes all backends (ONNX Runtime, PyTorch/LibTorch, TensorRT)
# and all required CUDA/cuDNN/NCCL libraries out of the box.
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Expose Triton ports: 8000 (HTTP), 8001 (gRPC), 8002 (Metrics)
EXPOSE 8000 8001 8002

# The ENTRYPOINT is the main command that starts the server
ENTRYPOINT ["tritonserver"]

# Default arguments:
# --model-repository: folder where models are located
# --allow-gpu-metrics=false: disabled to prevent errors on systems without NVIDIA GPUs (like Mac)
CMD ["--model-repository=/models", "--allow-gpu-metrics=false", "--exit-on-error=false"]