import torch
from models.resnet50 import load_model, generate_input

def test_resnet50_output_consistency():
    model = load_model("cpu")
    input_data = generate_input("cpu")

    with torch.no_grad():
        out1 = model(input_data)
        out2 = model(input_data)

    diff = torch.abs(out1 - out2).mean().item()
    assert diff < 1e-6
