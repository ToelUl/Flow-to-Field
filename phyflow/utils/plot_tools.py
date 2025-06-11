from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from PIL import Image


def load_image_pil(path: Union[str, Path]) -> Image.Image:
    """Load an image from disk using Pillow.

    Args:
        path (str | Path): Path to the image file.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    img = Image.open(path).convert("RGB")
    return img


def display_image_pil(img: Image.Image, figsize: tuple = (6, 6)) -> None:
    """Display a PIL image using matplotlib.

    Args:
        img (PIL.Image.Image): Image to display.
        figsize (tuple, optional): Figure size. Defaults to (6, 6).
    """
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(img)
    plt.show()
