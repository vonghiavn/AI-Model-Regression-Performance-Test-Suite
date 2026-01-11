#!/bin/bash
set -e

python framework/test_runner.py --model resnet50 --device gpu
python framework/test_runner.py --model bert --device gpu

pytest test_cases/
