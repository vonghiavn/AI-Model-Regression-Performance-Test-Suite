import torch
import torchvision.models as models

def load_model(device: str):
    """
    Load ResNet50 model for inference testing.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model

def generate_input(device: str):
    """
    Generate dummy input for ResNet50.
    """
    return torch.randn(1, 3, 224, 224).to(device)
