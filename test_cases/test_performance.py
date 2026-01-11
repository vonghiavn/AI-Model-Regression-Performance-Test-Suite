import torch
from models.resnet50 import load_model, generate_input
from framework.metrics import measure_performance

def test_resnet50_latency_gpu():
    if not torch.cuda.is_available():
        return

    model = load_model("cuda")
    input_data = generate_input("cuda")

    metrics = measure_performance(model, "cuda", input_data)
    assert metrics["latency_ms"] < 50
