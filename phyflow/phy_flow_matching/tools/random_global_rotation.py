import torch
from torch import Tensor, nn


class RandomGlobalRotation(nn.Module):
    """
    Applies a random global rotation to a tensor.

    This transform is robust to both single 3D samples (C, H, W) and batches of
    4D samples (B, C, H, W).
    """

    def __init__(self, is_rad: bool = True) -> None:
        """
        Initializes the RandomGlobalRotation transform.

        Args:
            is_rad (bool): If True, treat input as radians. If False, treat
                           input as (cos, sin) pairs. Defaults to True.
        """
        super().__init__()
        self.is_rad = is_rad

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Performs the random global rotation on the input tensor.
        Handles both 3D and 4D tensors.
        """
        # Check if the input is a single sample (3D) and add a batch dimension
        is_single_sample = tensor.dim() == 3
        if is_single_sample:
            tensor = tensor.unsqueeze(0)  # Add batch dimension: (C, H, W) -> (1, C, H, W)

        # Generate a single random rotation angle alpha
        random_rotation_alpha = torch.rand(1, device=tensor.device, dtype=tensor.dtype) * 2 * torch.pi

        if self.is_rad:
            # Mode 1: Operating on radians
            if tensor.shape[1] != 1:
                raise ValueError(f"Input tensor must have 1 channel for is_rad=True, but got {tensor.shape[1]}")

            rotated_tensor = tensor + random_rotation_alpha
            wrapped_tensor = torch.remainder(rotated_tensor, 2 * torch.pi)
        else:
            # Mode 2: Operating on (cos, sin) components
            if tensor.shape[1] != 2:
                raise ValueError(f"Input tensor must have 2 channels for is_rad=False, but got {tensor.shape[1]}")

            cos_alpha = torch.cos(random_rotation_alpha)
            sin_alpha = torch.sin(random_rotation_alpha)

            cos_theta = tensor[:, 0:1, :, :]
            sin_theta = tensor[:, 1:2, :, :]

            new_cos = cos_theta * cos_alpha - sin_theta * sin_alpha
            new_sin = sin_theta * cos_alpha + cos_theta * sin_alpha

            wrapped_tensor = torch.cat([new_cos, new_sin], dim=1)

        # If we added a batch dimension, remove it before returning
        if is_single_sample:
            wrapped_tensor = wrapped_tensor.squeeze(0)  # Remove batch dimension: (1, C, H, W) -> (C, H, W)

        return wrapped_tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(is_rad={self.is_rad})"