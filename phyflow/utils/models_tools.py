
import torch

def check_model(model: torch.nn.Module):
    """Print the model architecture and the number of trainable parameters.
    Args:
        model (torch.nn.Module): The model to check.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Total number of trainable parameters: {trainable_params/1e6:.2f} M")
