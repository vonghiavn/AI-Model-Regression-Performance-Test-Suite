import torch
from transformers import BertTokenizer, BertModel

def load_model(device: str):
    """
    Load BERT model for inference testing.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    model.to(device)
    return model, tokenizer

def generate_input(tokenizer, device: str):
    """
    Generate dummy input for BERT.
    """
    encoded = tokenizer(
        "NVIDIA builds world-class AI platforms.",
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in encoded.items()}
