# AI Model Regression & Performance Test Suite

## Overview

AI Model Regression & Performance Test Suite is a lightweight framework for
benchmarking and detecting performance regressions in deep learning models
across CPU and GPU environments.

The framework is designed for Linux-based systems with CUDA support, and is
suitable for CI/CD and model validation pipelines.

## Features

- Performance benchmarking (latency, FPS, CPU/GPU memory)
- Baseline-driven regression detection
- CPU and GPU (CUDA) support
- Automated JSON reports
- CI-friendly architecture

## Supported Models

- ResNet50 (Vision)
- BERT (NLP)

## Requirements

- Python 3.9+
- PyTorch
- NVIDIA GPU + CUDA (optional, for GPU tests)
- Linux (recommended)

## Installation

```bash
pip install -r requirements.txt
