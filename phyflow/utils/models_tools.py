import os
from typing import Tuple, Dict, Any

import copy
import torch
from torch import nn, Tensor
from thop import profile
from contextlib import redirect_stdout

def count_params_and_flops(module: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, float]:
    """Count the number of parameters and FLOPs of a PyTorch module.

    This function constructs a dummy input tensor with batch size 1,
    computes the total number of parameters in the module, and profiles
    the module to determine its floating-point operations (FLOPs).

    Args:
        module (nn.Module): The PyTorch module to profile.
        input_shape (Tuple[int, ...]): Shape of the input tensor excluding
            the batch dimension (e.g., (3, 224, 224)).

    Returns:
        Tuple[int, float]:
            params_total: Total number of parameters in the module.
            flops_total: Total number of FLOPs required by the module.
    """
    # Create a dummy input with batch size = 1
    dummy_input = torch.randn((1, *input_shape), device=next(module.parameters()).device)

    # Calculate total number of parameters
    params_total = sum(p.numel() for p in module.parameters())

    # Profile FLOPs and suppress thop output
    with redirect_stdout(open(os.devnull, "w")):
        flops_total, _ = profile(module, inputs=(dummy_input,))

    return params_total, flops_total


def profile_model(
    model: nn.Module,
    input_dict: Dict[str, Any]
) -> None:
    """Profile a RoDitUnet model: compute parameter count, FLOPs, and estimate memory usage.

    This function moves the model and dummy inputs to the same device,
    uses THOP to compute total parameters and FLOPs, then prints:
      - parameter count in millions (M)
      - FLOPs in billions (G)
      - estimated memory usage in megabytes (MB), assuming 4 bytes per parameter

    Args:
        model (nn.Module): Initialized RoDitUnet (or any nn.Module) to profile.
        input_dict (Dict[str, Any]):
            A dict containing at least:
              - 'x': Tensor of shape (B, C, H, W)
              - 'time': Tensor of shape (B,)
            Optionally:
              - 'conditions': Sequence[Tensor] of shape (B,) each

    Raises:
        KeyError: If required keys ('x', 'time') are missing in input_dict.
    """
    # Ensure required inputs are present
    if 'x' not in input_dict or 'time' not in input_dict:
        raise KeyError("input_dict must contain 'x' and 'time' keys")

    # Move model to evaluation mode and get device
    model.eval()
    device = next(model.parameters()).device

    # Move inputs to model device
    x: Tensor = input_dict['x'].to(device)
    time: Tensor = input_dict['time'].to(device)
    conditions = input_dict.get('conditions', None)
    if conditions is not None:
        # Move each condition tensor to device
        conditions = [c.to(device) for c in conditions]

    # Build input tuple for THOP
    if conditions is not None:
        inputs = (x, time, conditions)
    else:
        inputs = (x, time)

    profiling_model = copy.deepcopy(model)
    # Profile parameters and FLOPs, suppressing THOP stdout
    with redirect_stdout(open(os.devnull, 'w')):
        flops, params = profile(profiling_model, inputs=inputs, verbose=False)

    # Convert to human-readable units
    params_m = params / 1e3
    flops_m = flops / 1e9
    mem_bytes = params * 4  # float32 = 4 bytes
    mem_mb = mem_bytes / (1024 ** 2)

    # Print results
    print(f"Parameters:                           {params_m:.2f} K")
    print(f"FLOPs (floating point operations):    {flops_m:.4f} G")
    print(f"Estimated memory usage (params only): {mem_mb:.2f} MB")


def check_model(model: torch.nn.Module):
    """Print the model architecture and the number of trainable parameters.
    Args:
        model (torch.nn.Module): The model to check.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Total number of trainable parameters: {trainable_params/1e6:.2f} M")


def lock_all_convolutional_layers(model: nn.Module) -> None:
    """Freezes all convolutional layers in a given model for fine-tuning."""
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            for param in module.parameters():
                param.requires_grad = False


def lock_all_non_convolutional_layers(model: nn.Module) -> None:
    """Freezes all non-convolutional layers in a model for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            for param in module.parameters():
                param.requires_grad = True


def unlock_all_layers(model: nn.Module) -> None:
    """Unfreezes all layers in a model."""
    for param in model.parameters():
        param.requires_grad = True