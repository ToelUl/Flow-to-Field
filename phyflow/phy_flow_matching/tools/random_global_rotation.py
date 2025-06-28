import torch
from torch import Tensor, nn

class RandomGlobalRotation(nn.Module):
    """
    Applies a random global rotation to a tensor of angles.

    This transform generates a single random rotation value in the range [0, 2 * pi)
    and adds it to all elements of the input tensor. The resulting values are
    then wrapped to stay within the [0, 2 * pi) range.

    The input tensor is expected to contain values representing angles in radians.
    """

    def __init__(self) -> None:
        """
        Initializes the RandomGlobalRotation transform.
        """
        super().__init__()

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Performs the random global rotation on the input tensor.

        Args:
            tensor (Tensor): The input tensor with angular values in the
                           range [0, 2 * pi).

        Returns:
            Tensor: The tensor with the random global rotation applied, with
                    values wrapped in the [0, 2 * pi) range.
        """
        random_rotation = torch.rand(1, device=tensor.device, dtype=tensor.dtype) * 2 * torch.pi

        rotated_tensor = tensor + random_rotation

        wrapped_tensor = torch.remainder(rotated_tensor, 2 * torch.pi)

        return wrapped_tensor

    def __repr__(self) -> str:
        """
        Provides a string representation of the transform.
        """
        return f"{self.__class__.__name__}()"