"""An efficient Heun's 3rd order ODE solver for flow matching inference."""

import time
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn


class Heun3Solver(nn.Module):
    """A highly efficient Heun's 3rd order (RK3) ODE solver for inference.

    This solver is optimized for flow matching inference tasks, prioritizing
    speed and memory efficiency. It operates without building a computation
    graph and utilizes in-place operations where possible.

    The implementation is designed to be compatible with models that require
    additional conditional inputs, making it versatile for various generative
    modeling scenarios.

    The core of the solver implements the following steps for an ODE
    dy/dt = v(t, y) with a step size h:
        1. k1 = v(t, y)
        2. k2 = v(t + h/3, y + (h/3) * k1)
        3. k3 = v(t + 2h/3, y + (2h/3) * k2)
        4. y_next = y + (h/4) * (k1 + 3*k3)
    """

    def __init__(self):
        """Initializes the Heun3Solver module."""
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module,
        t_span: Union[List[float], torch.Tensor],
        conditions: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Executes the ODE solving process from t_span[0] to t_span[-1].

        Args:
            x: The initial state tensor, typically noise sampled from a
                distribution. Shape: (B, C, H, W).
            model: The neural network that models the velocity field v(t, x).
                Its forward method should accept `x`, `t`, and `conditions`.
            t_span: A sequence of time steps for the integration. Can be a
                Python list or a PyTorch tensor. Non-uniform steps are supported.
            conditions: An optional sequence of conditional tensors to be passed
                to the model. Defaults to None.

        Returns:
            The solution tensor at the final time step specified in `t_span`.
        """
        if not isinstance(t_span, torch.Tensor):
            t_span = torch.tensor(t_span, device=x.device, dtype=x.dtype)

        # Clone the input tensor to avoid modifying it in place.
        x_current = x.clone()

        # A temporary tensor for intermediate calculations, reused to save memory.
        x_intermediate = torch.empty_like(x_current)

        for i in range(len(t_span) - 1):
            t_current = t_span[i]
            t_next = t_span[i + 1]

            # Create a batch-wise time tensor for the model.
            t_batch = torch.full((x_current.shape[0],),
                                 t_current,
                                 device=x_current.device,
                                 dtype=x_current.dtype)

            h = t_next - t_current

            # Step 1: Calculate k1.
            k1 = model(x_current, t_batch, conditions=conditions)

            # Step 2: Calculate k2.
            # Reuse x_intermediate to calculate y + (h/3) * k1.
            torch.add(x_current, k1, alpha=h / 3.0, out=x_intermediate)
            t_k2 = t_current + h / 3.0
            t_batch.fill_(t_k2)
            k2 = model(x_intermediate, t_batch, conditions=conditions)

            # Step 3: Calculate k3.
            # Reuse x_intermediate to calculate y + (2h/3) * k2.
            torch.add(x_current, k2, alpha=2.0 * h / 3.0, out=x_intermediate)
            t_k3 = t_current + 2.0 * h / 3.0
            t_batch.fill_(t_k3)
            k3 = model(x_intermediate, t_batch, conditions=conditions)

            # Step 4: Final update using an in-place operation.
            # x_current += (h/4) * (k1 + 3*k3)
            # Combine update term into k3 to save memory.
            k3.mul_(3.0).add_(k1)
            x_current.add_(k3, alpha=h / 4.0)

        return x_current


class _DummyVectorField(nn.Module):
    """A dummy model for demonstrating the solver.

    This model simulates a trained neural network (e.g., a U-Net) that
    predicts a velocity field.

    Args:
        channels: The number of input and output channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        # A simple convolutional layer to simulate a network operation.
        self.layer = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Simulates the forward pass of a velocity prediction model.

        Note: In a real model, `t` and `conditions` would be used to modulate
        the network's behavior, for instance, through embeddings. Here, they
        are ignored for simplicity, but the signature is kept compatible.

        Args:
            x: The input state tensor.
            t: The time step tensor.
            conditions: The conditional inputs (ignored).

        Returns:
            A tensor representing the predicted velocity field.
        """
        return self.layer(x)


def main():
    """Main function to demonstrate and validate the Heun3Solver."""
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define Parameters
    batch_size = 8
    channels = 3
    height, width = 64, 64
    num_steps = 101

    # 3. Prepare Model and Inputs
    model = _DummyVectorField(channels).to(device)
    model.eval()

    initial_noise = torch.randn(batch_size,
                                channels,
                                height,
                                width,
                                device=device)

    # Define a non-uniform time span for integration from t=1 to t=0.
    # A power-law spacing concentrates steps near t=0.
    t_span = torch.linspace(1.0, 0.0, num_steps, device=device).pow(2)

    print(f"\nModel: {_DummyVectorField.__name__}")
    print(f"Input shape: {initial_noise.shape}")
    print(f"Time span shape: {t_span.shape}, from {t_span[0]:.2f} to {t_span[-1]:.2f}")

    # 4. Initialize and Run Solver
    solver = Heun3Solver()

    print("\nStarting ODE integration...")
    start_time = time.perf_counter()

    generated_data = solver(
        x=initial_noise,
        model=model,
        t_span=t_span,
        conditions=None  # Explicitly pass None for clarity.
    )

    end_time = time.perf_counter()
    print(f"Integration finished in {end_time - start_time:.4f} seconds.")
    print(f"Output shape: {generated_data.shape}")

    # 5. Validation
    assert generated_data.shape == initial_noise.shape
    # Check that gradients were not computed.
    assert not generated_data.requires_grad
    # Check that the original input tensor was not modified.
    assert generated_data.data_ptr() != initial_noise.data_ptr()
    # Check that the output is different from the input.
    assert not torch.allclose(generated_data, initial_noise)

    print("\nAll checks passed successfully!")


if __name__ == '__main__':
    main()