#!/bin/bash
echo "===== SYSTEM INFO ====="
uname -a

echo "===== CUDA INFO ====="
nvcc --version || echo "CUDA not found"

echo "===== GPU INFO ====="
nvidia-smi || echo "GPU not available"
