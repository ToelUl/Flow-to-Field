import torch
import random
from torch import Tensor
import torch.nn as nn
from typing import Union, Tuple


class RandomRoll(nn.Module):
    """Randomly roll an image tensor along height and width axes.

    This module applies a circular shift (torch.roll) to each input tensor,
    simulating periodic boundary translation. Works on both CPU and CUDA tensors.

    Args:
        max_shift (int or tuple[int, int]):
            Maximum absolute shift in pixels. If int, uses same bound for
            height and width; if tuple, should be (max_shift_y, max_shift_x).
        p (float):
            Probability of applying the roll. Must be in [0, 1].

    Shape:
        - Input: (C, H, W) or (N, C, H, W)
        - Output: same as input.

    Example:
        >>> aug = RandomRoll(max_shift=(5, 10), p=0.5)
        >>> x = torch.randn(8, 3, 224, 224, device='cuda')
        >>> y = aug(x)  # 50% 機率對 batch 中每張圖做隨機平移
    """

    def __init__(
        self,
        max_shift: Union[int, Tuple[int, int]] = 10,
        p: float = 1.0
    ) -> None:
        super().__init__()
        if isinstance(max_shift, int):
            self.max_shift_y = self.max_shift_x = max_shift
        else:
            self.max_shift_y, self.max_shift_x = max_shift
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0,1], got {p}")
        self.p = p

    def forward(self, img: Tensor) -> Tensor:
        """Apply random circular roll to image tensor.

        Args:
            img (Tensor): shape (C, H, W) or (N, C, H, W). If batch, same shift
                applied to all images in the batch.

        Returns:
            Tensor: shifted tensor with identical shape.
        """
        if not torch.rand(1, device=img.device).item() < self.p:
            return img

        dy = random.randint(-self.max_shift_y, self.max_shift_y)
        dx = random.randint(-self.max_shift_x, self.max_shift_x)
        # dims -2, -1 對應 H, W
        return torch.roll(img, shifts=(dy, dx), dims=(-2, -1))


def _validate_random_roll() -> None:
    """Basic unit tests for RandomRoll module."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aug = RandomRoll(max_shift=(2, 3), p=1.0).to(device)
    x1 = torch.arange(0, 2*4*5, device=device).reshape(2, 4, 5)
    y1 = aug(x1)
    assert y1.shape == x1.shape, "Shape must be preserved"
    x2 = torch.arange(0, 2*3*4*5, device=device).reshape(2, 2, 4, 5)
    y2 = aug(x2)
    assert y2.shape == x2.shape, "Batch shape must be preserved"
    inv = torch.roll(y1, shifts=(- (y1.shape[1]-1), -2), dims=(1, 2))
    assert torch.allclose(inv, x1), "Inverse roll failed"
    print("✔ RandomRoll validation passed.")


if __name__ == "__main__":
    _validate_random_roll()
