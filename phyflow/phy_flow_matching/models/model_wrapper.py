from typing import Optional, Sequence
import torch
from flow_matching.utils import ModelWrapper


class CFMWrapper(ModelWrapper):
    """Wraps the UNet, handling time tensor broadcasting and passing arguments.

    This wrapper ensures the time tensor `t` is correctly broadcasted to the
    batch size before being passed to the underlying model. It now accepts
    a sequence of condition tensors.
    """
    def __init__(self, model):
        """Initializes the wrapper.

        Args:
            model: The UNet instance to wrap.
        """
        super().__init__(model)
        self.model = model # Keep direct reference if needed

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Optional[Sequence[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Performs the forward pass after ensuring time tensor compatibility.

        Args:
            x: Input tensor (e.g., noisy image) of shape (B, C_in, H, W).
            t: Time step tensor, potentially shape (,), (1,), (B,), or (B, 1).
               Will be expanded to (B,).
            conditions: A sequence (list or tuple) of conditional input tensors,
                each of shape (B,) or (B, 1), or None if no conditions are used.
                Passed directly to the wrapped model. Defaults to None.

        Returns:
            Output tensor from the wrapped model, typically of shape (B, C_out, H, W).

        Raises:
            ValueError: If the time tensor `t` cannot be correctly broadcast
                to the batch size `B` derived from `x`.
        """
        batch_size = x.shape[0]

        # --- Time Tensor Broadcasting Logic ---
        original_t_shape = t.shape
        if t.ndim == 0:
            # Expand scalar time tensor to batch size
            t = t.expand(batch_size)
        elif t.ndim == 1 and t.shape[0] == 1:
             # Expand single-element time tensor to batch size
             t = t.expand(batch_size)
        elif t.ndim == 2 and t.shape[0] == batch_size and t.shape[1] == 1:
            t = t.squeeze(1)

        # Final check after potential expansion
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"Internal Error: Time tensor shape after potential expansion ({t.shape}) "
                f"from original shape {original_t_shape} is still incompatible with batch size {batch_size}"
            )

        # --- Call the wrapped model's forward method ---
        # return self.model.forward(x, t, conditions=conditions)
        x_theta = self.model.forward(x, t, conditions=conditions)
        return (x_theta - x) / (torch.ones_like(x_theta) - t.view(t.shape[0],1,1,1).expand(x_theta.shape))


