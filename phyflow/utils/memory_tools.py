import torch
import torch.nn as nn
from typing import Optional, Any, List
import gc


def _measure_activation_bytes(
    model: nn.Module,
    channels: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device
) -> int:
    """Measure approximate activation memory (in bytes) per sample for the given model.

    Registers forward hooks to accumulate the output tensor sizes (activations) for a dummy input of shape (1, C, H, W).

    Args:
        model: The neural network model.
        channels: Number of input channels C.
        height: Input height H.
        width: Input width W.
        dtype: Data type of input tensor.
        device: Device to run the dummy forward pass on.

    Returns:
        Total activation memory per sample in bytes.
    """
    hooks = []
    activation_bytes = 0

    def _hook(module, inp, out):
        nonlocal activation_bytes
        outputs = out if isinstance(out, (list, tuple)) else [out]
        for o in outputs:
            if torch.is_tensor(o):
                activation_bytes += o.numel() * o.element_size()

    for module in model.modules():
        if not list(module.children()):
            hooks.append(module.register_forward_hook(_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros((1, channels, height, width), dtype=dtype, device=device)
        dummy_time = torch.zeros((1, 1), dtype=dtype, device=device)
        dummy_condition = torch.zeros((1, 1), dtype=dtype, device=device)
        _ = model(dummy, dummy_time, (dummy_condition,))

    for h in hooks:
        h.remove()
    return activation_bytes


def estimate_max_batch_size(
    model: nn.Module,
    num_data: int,
    channels: int,
    height: int,
    width: int,
    time_steps: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    num_evals: Optional[int] = None,
    use_adjoint: bool = False,
    available_memory: Optional[int] = None,
    overhead: int = 0,
    safety_factor: float = 0.7
) -> int:
    """Estimate optimal batch size conservatively including model activations and torchdiffeq stacking.

    This function accounts for:
      1. Model activation memory per sample (forward activations).
      2. ODE output stacking memory (time_steps, B, C, H, W).
      3. Autograd cache for ODE (adjoint vs non-adjoint).
      4. Model parameter and gradient memory as overhead.
      5. Applies a safety factor to avoid overestimation.

    Args:
        model: Neural network module applied before ODE stacking.
        num_data: Total number of data samples available.
        channels: Input channel count C.
        height: Input height H.
        width: Input width W.
        time_steps: Number of time points T.
        dtype: Tensor data type (default torch.float32).
        device: Device to run the model on (default None, auto-detected).
        num_evals: Number of ODE function evaluations N_eval (required if non-adjoint).
        use_adjoint: If True, assume constant autograd cache for ODE.
        available_memory: Total free GPU memory in bytes; auto-detected if None.
        overhead: Additional fixed memory overhead in bytes.
        safety_factor: Fraction of available memory to actually use (0 < safety_factor <= 1).

    Returns:
        Maximum feasible batch size (<= num_data) that fits in memory conservatively.

    Raises:
        ValueError: If num_evals is None when use_adjoint is False.
        RuntimeError: If CUDA unavailable and available_memory not provided.
    """
    # Detect available GPU memory if not provided
    if available_memory is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; please specify available_memory manually."
            )
        if device is None:
            device = torch.device("cuda")
        else:
            device = torch.device(device)
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        available_memory = total - max(reserved, allocated)
    else:
        device = next(model.parameters()).device

    # Apply safety factor
    usable_memory = int(0.1 * available_memory * safety_factor)

    # Bytes per element
    bytes_elem = torch.tensor([], dtype=dtype).element_size()

    # Model parameter + gradient overhead
    param_elems = sum(p.numel() for p in model.parameters())
    param_bytes = param_elems * bytes_elem * 2  # params + grads

    # Activation memory per sample
    activation_bytes = _measure_activation_bytes(
        model, channels, height, width, dtype, device
    )

    # ODE stacking memory per sample
    state_elems = channels * height * width
    ode_bytes = (1 + time_steps) * state_elems * bytes_elem

    # Autograd cache for ODE per sample
    if use_adjoint:
        ode_cache = state_elems * bytes_elem
    else:
        if num_evals is None:
            raise ValueError("num_evals required when use_adjoint=False")
        ode_cache = num_evals * state_elems * bytes_elem

    per_sample_bytes = activation_bytes + ode_bytes + ode_cache

    # Total fixed overhead = model params + user overhead
    fixed_overhead = param_bytes + overhead

    # Compute max batch conservatively
    max_batch = (usable_memory - fixed_overhead) // per_sample_bytes
    max_batch = int(max(0, max_batch))
    return min(max_batch, num_data)



def clear_cuda_cache(del_vars: Optional[List[Any]] = None) -> None:
    """Forcefully clear PyTorch CUDA cache to free GPU memory.

    This function will:
    1. Optionally delete Python references to tensors/models.
    2. Perform Python garbage collection.
    3. Empty the CUDA caching allocator.
    4. Trigger IPC memory collection (for multiprocessing scenarios).

    Args:
        del_vars (Optional[List[Any]]): List of Python variables (tensors, models, optimizers)
            to delete before clearing the cache. If None, no deletion is performed.

    Raises:
        RuntimeError: If CUDA is not available on this system.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")
    
    if del_vars:
        for var in del_vars:
            try:
                del var
            except NameError:
                pass
    gc.collect()
    torch.cuda.empty_cache()
    
    if hasattr(torch.cuda, 'ipc_collect'):
        torch.cuda.ipc_collect()

