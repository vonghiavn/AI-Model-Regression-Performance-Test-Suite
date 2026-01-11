import argparse
import json
import torch

from framework.metrics import measure_performance
from framework.compare import compare_metrics
from framework.reporter import write_report

from models.resnet50 import load_model as load_resnet, generate_input as resnet_input
from models.bert import load_model as load_bert, generate_input as bert_input

MODEL_REGISTRY = {
    "resnet50": "resnet50",
    "bert": "bert"
}

def run_test(model_name: str, device: str):
    device = "cuda" if device == "gpu" and torch.cuda.is_available() else "cpu"

    if model_name == "resnet50":
        model = load_resnet(device)
        input_data = resnet_input(device)

    elif model_name == "bert":
        model, tokenizer = load_bert(device)
        input_data = bert_input(tokenizer, device)

    else:
        raise ValueError("Unsupported model")

    metrics = measure_performance(model, device, input_data)

    with open(f"baselines/{model_name}_gpu.json") as f:
        baseline = json.load(f)

    regressions = compare_metrics(metrics, baseline)

    report = write_report(model_name, device, metrics, regressions)

    print(f"[{report['status']}] {model_name} on {device}")
    if regressions:
        for r in regressions:
            print(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()

    run_test(args.model, args.device)
