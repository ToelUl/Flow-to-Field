import torch
import torch.nn as nn
from typing import Optional, Any, List
import gc


def _measure_activation_bytes(
        model: nn.Module,
        channels: int,
        height: int,
        width: int,
        num_conditions: int,
        dtype: torch.dtype,
        device: torch.device
) -> int:
    """
    Measures the approximate activation memory (in bytes) for a single sample.

    This function registers forward hooks on the model's leaf modules to accumulate
    the size of their output tensors (activations) for a dummy input of shape
    (1, C, H, W).

    Note: This assumes the model's forward signature is model(x, t, context_list).
    If your model has a different signature, you may need to adjust the dummy
    input creation.

    Args:
        model: The neural network model.
        channels: Number of input channels C.
        height: Input height H.
        width: Input width W.
        num_conditions: Number of conditions in the context list.
        dtype: The data type of the input tensor.
        device: The device to run the dummy forward pass on.

    Returns:
        The total activation memory per sample in bytes.
    """
    # Clean up cache for a more accurate measurement
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hooks = []
    activation_bytes = 0

    def _hook(_module, _inp, out):
        nonlocal activation_bytes
        # Ensure multi-output modules are handled correctly
        outputs = out if isinstance(out, (list, tuple)) else [out]
        for o in outputs:
            if torch.is_tensor(o):
                activation_bytes += o.numel() * o.element_size()

    # Register hooks only on leaf modules (those with no children)
    for module in model.modules():
        if not list(module.children()):
            hooks.append(module.register_forward_hook(_hook))

    model.eval()
    with torch.no_grad():
        # Create dummy tensors that match the model's expected input signature
        dummy_x = torch.zeros((1, channels, height, width), dtype=dtype, device=device)
        dummy_time = torch.zeros((1,), dtype=dtype, device=device)
        dummy_condition = torch.zeros((1,), dtype=dtype, device=device)
        _ = model(dummy_x, dummy_time, [dummy_condition for _ in range(num_conditions)])

    # Clean up by removing the hooks
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
        num_conditions: int = 1,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        num_evals: Optional[int] = None,
        use_adjoint: bool = False,
        available_memory: Optional[int] = None,
        overhead: int = 0,
        safety_factor: float = 0.8
) -> int:
    """
    Conservatively estimates the maximum feasible batch size for an ODE solve.

    This function accounts for the primary sources of memory consumption:
      1. Model Parameters: The memory to store the model weights.
      2. ODE Output Stack: The memory for the final tensor storing the trajectory
         of shape (time_steps, B, C, H, W).
      3. Peak Computational Memory: The transient memory used during the solve,
         which differs significantly between adjoint and non-adjoint methods.
      4. A user-defined overhead and a safety factor.

    Args:
        model: The neural network to be evaluated in the ODE solver.
        num_data: The total number of samples in the dataset.
        channels: Input channel count C.
        height: Input height H.
        width: Input width W.
        time_steps: Number of time points T to be stored in the output.
        num_conditions: Number of conditions in the context list (default: 1).
        dtype: Tensor data type (default: torch.float32).
        device: Device to run the model on (default: None, auto-detected).
        num_evals: Number of function evaluations (NFE) for the ODE solver.
                   Required if `use_adjoint` is False.
        use_adjoint: If True, assumes the memory-efficient adjoint method is used.
        available_memory: Total free GPU memory in bytes. If None, it is auto-detected.
        overhead: Additional fixed memory overhead in bytes (e.g., for other variables).
        safety_factor: Fraction of available memory to use (0 < safety_factor <= 1).

    Returns:
        The maximum feasible batch size (<= num_data) that should fit in memory.

    Raises:
        ValueError: If `num_evals` is None when `use_adjoint` is False.
        RuntimeError: If CUDA is unavailable and `available_memory` is not provided.
    """
    # 1. Detect device and available memory
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if available_memory is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; please specify `available_memory` manually."
            )
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = total_memory - max(reserved_memory, allocated_memory)

    # Apply safety factor to prevent out-of-memory errors
    usable_memory = int(available_memory * safety_factor)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    # 2. Calculate fixed memory overhead (independent of batch size)
    # Model parameters (for inference, we don't need gradients, so no * 2)
    param_elems = sum(p.numel() for p in model.parameters())
    param_bytes = param_elems * bytes_per_elem
    fixed_overhead = param_bytes + overhead

    # 3. Calculate per-sample memory cost
    # (a) Measure activation memory for a single model forward pass
    activation_bytes_per_sample = _measure_activation_bytes(
        model, channels, height, width, num_conditions, dtype, device
    )

    state_elems = channels * height * width

    # (b) Calculate memory for storing the final ODE output trajectory
    output_stack_bytes_per_sample = time_steps * state_elems * bytes_per_elem

    # (c) Estimate the peak computational memory during the ODE solve
    if use_adjoint:
        # Adjoint method has low, ~constant memory. Peak is dominated by a single
        # forward pass's activations.
        compute_bytes_per_sample = activation_bytes_per_sample
    else:
        # Non-adjoint methods store the entire computation graph. Memory scales
        # linearly with the number of function evaluations (NFE).
        if num_evals is None:
            raise ValueError("`num_evals` is required when `use_adjoint` is False.")
        # Peak memory is roughly the sum of activations from all NFEs.
        compute_bytes_per_sample = num_evals * activation_bytes_per_sample

    per_sample_bytes = output_stack_bytes_per_sample + compute_bytes_per_sample

    if per_sample_bytes <= 0:
        return num_data  # Avoid division by zero

    # 4. Calculate the maximum batch size
    available_for_batches = usable_memory - fixed_overhead
    max_batch = available_for_batches // per_sample_bytes

    # Ensure the result is a non-negative integer and does not exceed the dataset size
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

