# AI Model Regression & Performance Test Suite

## Overview

This project is an automated test framework designed to validate AI model
accuracy, performance, and regression across CPU and GPU environments.

## Features

- Performance benchmarking (latency, FPS, memory)
- Regression detection using baseline comparison
- GPU-aware testing (CUDA)
- Automated reporting (JSON)
- CI-ready architecture

## Supported Models

- ResNet50
- BERT

## How to Run

```bash
pip install -r requirements.txt
bash scripts/run_all_tests.sh
```
