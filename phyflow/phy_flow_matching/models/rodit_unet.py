import math
from typing import Optional, Sequence, Tuple
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ==============================================================================
# Helper Function
# ==============================================================================

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Applies a learned affine transformation to the input tensor.

    This function is used for Adaptive LayerNorm (adaLN), where the input
    tensor `x` is modulated by a shift and a scale vector derived from an
    embedding (e.g., a time embedding).

    Args:
        x: The input tensor. Can be 2D (B, C) or 4D (B, C, H, W).
        shift: The shift vector, broadcastable to `x`.
        scale: The scale vector, broadcastable to `x`.

    Returns:
        The modulated tensor.

    Raises:
        ValueError: If the input tensor has an unsupported dimension.
    """
    if x.dim() == 4:
        # For 4D tensors, scale and shift are applied channel-wise.
        return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
    elif x.dim() == 2:
        # For 2D tensors, scale and shift are applied feature-wise.
        return x * (1 + scale) + shift
    else:
        raise ValueError(f"Unsupported input dimension {x.dim()}. Expected 2 or 4.")


# ==============================================================================
# Core Modules
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Creates sinusoidal positional embeddings.

    This module generates fixed sinusoidal positional embeddings for a given
    1D input tensor, often used for time or conditional embeddings in
    diffusion models.

    Attributes:
        embedding_dim: The dimension of the embedding vector. It's enforced
            to be an even number.
        max_period: The maximum period of the sinusoidal functions.
    """

    def __init__(self, embedding_dim: int, max_period: int = 10000):
        """Initializes the SinusoidalPosEmb module.

        Args:
            embedding_dim: The desired dimension of the embedding. If odd, it
                will be incremented by 1 to make it even.
            max_period: The maximum period for the sine and cosine functions.
        """
        super().__init__()
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be a positive integer, got {embedding_dim}")
        # Ensure the embedding dimension is even
        if embedding_dim % 2 != 0:
            embedding_dim += 1
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, x: Tensor) -> Tensor:
        """Generates the positional embeddings for the input tensor.

        Args:
            x: A 1D tensor of positions (e.g., timesteps). If not 1D, it will
               be flattened.

        Returns:
            A 2D tensor of shape (B, embedding_dim) containing the
            positional embeddings.
        """
        if x.ndim != 1:
            x = x.flatten()
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(x.device)
        args = x.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ECA(nn.Module):
    """Efficient Channel Attention (ECA) block.

    An efficient channel attention mechanism that uses a 1D convolution to
    generate channel-wise attention weights.

    Attributes:
        use_eca: A boolean flag indicating if ECA is active.
        avg_pool: Adaptive average pooling layer.
        conv: 1D convolution layer for generating attention weights.
        sigmoid: Sigmoid activation to normalize weights.
    """

    def __init__(self, channels: int, b: int = 1, gamma: int = 2):
        """Initializes the ECA module.

        Args:
            channels: The number of input channels. If less than or equal to 0,
                the block will act as an identity function.
            b: A parameter to adjust the kernel size calculation. Defaults to 1.
            gamma: A parameter to adjust the kernel size calculation. Defaults to 2.
        """
        super().__init__()
        self.use_eca = channels > 0
        if not self.use_eca:
            return

        # Calculate kernel size dynamically based on channel number
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the ECA block."""
        if not self.use_eca:
            return x

        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # (B, C, 1, 1) -> (B, C, 1) -> (B, 1, C)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)    # (B, 1, C) -> (B, C, 1) -> (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class RotaryEmbedding1D(nn.Module):
    """Implements 1D Rotary Positional Embedding (RoPE).

    This module applies rotary embeddings to the query and key tensors, encoding
    positional information by rotating feature vectors based on their 1D
    sequence position. It includes a caching mechanism to avoid recomputing
    sinusoidal values for inputs of the same sequence length.

    Attributes:
        dim (int): The feature dimension. Must be an even number.
        base (int): The base for the geometric progression of frequencies.
        inv_freq (torch.Tensor): The inverse frequencies for the positional
            encoding. This is a registered buffer.
    """

    def __init__(self, dim: int, base: int = 10000):
        """Initializes the RotaryEmbedding1D module.

        Args:
            dim (int): The feature dimension for the embeddings. Must be an
                even number.
            base (int): The base value for the frequency calculation.

        Raises:
            ValueError: If `dim` is not an even number.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(
                f"Dimension must be an even number for 1D RoPE, but got {dim}.")

        self.dim = dim
        self.base = base

        # Calculate inverse frequencies.
        # This corresponds to theta_i = 1.0 / (base^(2i / dim))
        freqs = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (self.base**freqs)
        self.register_buffer("inv_freq", inv_freq)

        # Caching attributes.
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None
        self._cached_seq_len: int = -1

    def _update_cache(self, seq_len: int, device: torch.device):
        """Updates the cached sinusoidal embeddings if seq_len changes.

        Args:
            seq_len (int): The length of the input sequence.
            device (torch.device): The device of the input tensor.
        """
        if seq_len == self._cached_seq_len:
            return

        self._cached_seq_len = seq_len
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Calculate frequency components for each position.
        # Shape: (seq_len, dim / 2)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)

        # Interleave for sine and cosine application.
        # Shape: (seq_len, dim)
        embedded_freqs = freqs.repeat_interleave(2, dim=-1)

        # Reshape for broadcasting: (1, seq_len, dim)
        embedded_freqs = embedded_freqs.unsqueeze(0)

        self._cached_cos = embedded_freqs.cos()
        self._cached_sin = embedded_freqs.sin()

    def _apply_rotary_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the pre-computed rotary embeddings to the input tensor.

        This function performs the rotation on pairs of features.

        Args:
            x (torch.Tensor): Input tensor (query or key) of shape
                (Batch, SeqLen, Dim).

        Returns:
            torch.Tensor: The tensor with rotary positional embeddings applied.
        """
        # Split features into pairs: (x_1, x_2), (x_3, x_4), ...
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        if self._cached_cos is None or self._cached_sin is None:
            raise RuntimeError(
                "Cache is not initialized. Call _update_cache first.")

        # Apply rotation using broadcasted sin/cos values.
        # This is equivalent to complex multiplication:
        # (x1 + i*x2) * (cos + i*sin)
        cos = self._cached_cos[..., 0::2]
        sin = self._cached_sin[..., 0::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Reassemble the tensor.
        rotated_x = torch.empty_like(x)
        rotated_x[..., 0::2] = rotated_x1
        rotated_x[..., 1::2] = rotated_x2

        return rotated_x

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies 1D RoPE to query and key tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (Batch, SeqLen, Dim).
            k (torch.Tensor): Key tensor of shape (Batch, SeqLen, Dim).

        Returns:
            A tuple containing the rotated query and key tensors, both of shape
            (Batch, SeqLen, Dim).
        """
        seq_len = q.shape[1]
        self._update_cache(seq_len, q.device)
        rotated_q = self._apply_rotary_embedding(q)
        rotated_k = self._apply_rotary_embedding(k)
        return rotated_q, rotated_k


class RotaryEmbedding2D(nn.Module):
    """Implements 2D Rotary Positional Embedding (RoPE).

    This module applies rotary embeddings to the query and key tensors, encoding
    positional information by rotating feature vectors based on their 2D
    coordinates. The embedding dimension is split to encode x and y coordinates
    independently. It includes a caching mechanism to avoid recomputing sinusoidal
    values for inputs of the same spatial dimensions.

    Attributes:
        dim (int): The feature dimension of each attention head. Must be divisible
            by 4.
        base (int): The base for the geometric progression of frequencies.
        alpha (float): The NTK interpolation scaling factor. Defaults to 1.0 (no scaling).
        inv_freq_x (torch.Tensor): The inverse frequencies for the x-dimension.
        inv_freq_y (torch.Tensor): The inverse frequencies for the y-dimension.
    """

    def __init__(self, dim: int, base: int = 10000, alpha: float = 1.0):
        """Initializes the RotaryEmbedding2D module.

        Args:
            dim (int): The feature dimension per head. Must be an even number and
                divisible by 4.
            base (int): The base value for the frequency calculation.
            alpha (float): The NTK interpolation scaling factor. Defaults to 1.0 (no scaling).

        Raises:
            ValueError: If `dim` is not divisible by 4.
        """
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(
                f"Dimension must be divisible by 4 for 2D RoPE, but got {dim}.")

        self.dim = dim
        self.base = base
        self.alpha = alpha

        # Calculate inverse frequencies for x and y dimensions.
        # The dimension is split in half for x and y, and each half is further
        # split for sine and cosine components.
        dim_half = self.dim // 2
        freqs_for_half = torch.arange(0, dim_half, 2).float() / dim_half

        # Apply NTK scaling to the base if alpha is not 1.0
        # A larger alpha leads to a larger base, which results in smaller frequencies (longer wavelengths),
        # allowing the model to handle larger positions.
        scaled_base = self.base * (
                self.alpha ** (dim_half / (dim_half - 2.0 + 1e-6))
        ) if self.alpha != 1.0 and dim_half != 2.0 else self.base

        inv_freq = 1.0 / (scaled_base**freqs_for_half)

        self.register_buffer("inv_freq_x", inv_freq)
        self.register_buffer("inv_freq_y", inv_freq)

        # Caching attributes
        self._cached_cos: Tensor | None = None
        self._cached_sin: Tensor | None = None
        self._cached_seq_shape: Tuple[int, int] | None = None

    def _update_cache(self, height: int, width: int, device: torch.device):
        """Updates the cached sinusoidal embeddings if the sequence shape changes.

        Args:
            height (int): The height of the input feature map.
            width (int): The width of the input feature map.
            device (torch.device): The device of the input tensor.
        """
        seq_shape = (height, width)
        if seq_shape == self._cached_seq_shape:
            return

        self._cached_seq_shape = seq_shape

        pos_y = torch.arange(height, device=device, dtype=torch.float32)
        pos_x = torch.arange(width, device=device, dtype=torch.float32)

        # Calculate frequency components for each position.
        freqs_y = torch.einsum("i,j->ij", pos_y, self.inv_freq_y)
        freqs_x = torch.einsum("i,j->ij", pos_x, self.inv_freq_x)

        # Interleave for sine and cosine application.
        freqs_y = freqs_y.repeat_interleave(2, dim=-1)
        freqs_x = freqs_x.repeat_interleave(2, dim=-1)

        # Combine y and x frequencies. Shape: (H, W, dim).
        freqs = torch.cat([
            freqs_y.unsqueeze(1).expand(-1, width, -1),
            freqs_x.unsqueeze(0).expand(height, -1, -1)
        ], dim=-1)

        # Flatten and reshape for broadcasting. Shape: (1, H*W, 1, dim).
        freqs = freqs.flatten(0, 1).unsqueeze(0).unsqueeze(2)

        self._cached_cos = freqs.cos()
        self._cached_sin = freqs.sin()

    def _apply_rotary_embedding(self, x: Tensor) -> Tensor:
        """Applies the pre-computed rotary embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor (query or key) of shape
                (Batch, SeqLen, NumHeads, HeadDim).

        Returns:
            torch.Tensor: The tensor with rotary positional embeddings applied.
        """
        # x_rotated = (-x2, x1) * sin + (x1, x2) * cos
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # Ensure cached tensors are available.
        if self._cached_cos is None or self._cached_sin is None:
            raise RuntimeError(
                "Cache is not initialized. Call _update_cache first.")

        # Apply rotation using broadcasted sin/cos values.
        rotated_x1 = x1 * self._cached_cos[..., 0::2] - x2 * self._cached_sin[..., 0::2]
        rotated_x2 = x1 * self._cached_sin[..., 0::2] + x2 * self._cached_cos[..., 0::2]

        # Reassemble the tensor.
        rotated_x = torch.empty_like(x)
        rotated_x[..., 0::2] = rotated_x1
        rotated_x[..., 1::2] = rotated_x2

        return rotated_x

    def forward(self, q: Tensor, k: Tensor, height: int,
                width: int) -> Tuple[Tensor, Tensor]:
        """Applies 2D RoPE to query and key tensors.

        Args:
            q (torch.Tensor): Query tensor of shape
                (Batch, SeqLen, NumHeads, HeadDim).
            k (torch.Tensor): Key tensor of shape
                (Batch, SeqLen, NumHeads, HeadDim).
            height (int): The height of the original spatial feature map.
            width (int): The width of the original spatial feature map.

        Returns:
            A tuple containing the rotated query and key tensors.
        """
        self._update_cache(height, width, q.device)
        rotated_q = self._apply_rotary_embedding(q)
        rotated_k = self._apply_rotary_embedding(k)
        return rotated_q, rotated_k


class ConditionalSAWithRoPE(nn.Module):
    """A Single-Head Self-Attention block for fusing conditional embeddings.

    This module takes a sequence of embeddings (e.g., time, and other
    conditions), treats them as a sequence, and applies self-attention with
    Rotary Positional Embeddings (RoPE). This allows the embeddings to
    interact and create a context-aware fusion.

    Attributes:
        dim (int): The feature dimension of the embeddings.
        input-projection (nn.Linear): A linear layer to project the input
        norm (nn.RMSNorm): Pre-attention layer normalization.
        qkv_projection (nn.Linear): The layer to project the input sequence
            to Q, K, V.
        rope (RotaryEmbedding1D): The 1D rotary embedding module to encode the
            order of conditions.
        scale (float): The scaling factor for the dot product (1/sqrt(dim)).
    """
    def __init__(self, dim: int, qkv_bias: bool = False):
        """Initializes the ConditionalSAWithRoPE module.

        Args:
            dim (int): The feature dimension of the input and output embeddings.
                Must be an even number for RoPE compatibility.
            qkv_bias (bool): If True, adds a learnable bias to the QKV
                projection.
        """
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.input_projection = nn.Linear(dim, dim)
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.qkv_projection = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rope = RotaryEmbedding1D(dim=dim)

    def forward(self, embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the attention block.

        Args:
            embeddings (Sequence[torch.Tensor]): A sequence of embedding
                tensors. The first tensor is expected to be the primary
                embedding (e.g., time). Each tensor should have a shape of
                (Batch, Dim).

        Returns:
            torch.Tensor: An aggregated embedding tensor of shape (Batch, Dim),
                representing the processed first token of the sequence.
        """
        # Stack embeddings to form a sequence: (B, S, D)
        # where S is the number of conditions + 1 (for time).
        x = torch.stack(embeddings, dim=1)
        residual = x

        # Project the input sequence to a common dimension.
        x = self.input_projection(x)

        # Normalize the sequence.
        x = self.norm(x)

        # Project to Q, K, V and split.
        q, k, v = self.qkv_projection(x).chunk(3, dim=-1)

        # Apply RoPE to Query and Key to encode position of each condition.
        q, k = self.rope(q, k)

        # Compute scaled dot-product attention.
        # Add a head dimension for compatibility: (B, S, D) -> (B, 1, S, D)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=False
        ).squeeze(1)

        # Add residual connection.
        processed_sequence = residual + out

        return processed_sequence.mean(dim=1)


class MSAWithRoPE(nn.Module):
    """A Multi-Head Self-Attention module integrated with 2D RoPE.

    This module computes self-attention on a 2D feature map. It uses a
    depthwise separable convolution for efficient QKV projection and integrates
    2D Rotary Positional Embedding (RoPE) to provide positional awareness to the
    attention mechanism.

    Attributes:
        is_identity (bool): If True, the module acts as an identity function.
            This is true when the input channel count is zero.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        scale (float): The scaling factor for the dot product.
        qkv_projection (nn.Module): The layer to project input to Q, K, V.
        rope (RotaryEmbedding2D): The 2D rotary embedding module.
    """
    def __init__(self,
                 channels: int,
                 num_heads: int = 4,
                 qkv_bias: bool = False,
                 padding_mode: str = "circular",
                 alpha: float = 1.0
                 ):
        """Initializes the MSAWithRoPE module.

        Args:
            channels (int): The number of input and output channels. If 0,
                the module becomes an identity function.
            num_heads (int): The number of attention heads.
            qkv_bias (bool): If True, adds a learnable bias to the QKV
                projection.
            padding_mode (str): The padding mode for the QKV projection's
                convolution. Typically 'zeros' or 'circular'.
            alpha (float): The NTK interpolation scaling factor for RoPE.

        Raises:
            AssertionError: If `channels` is not divisible by `num_heads` or if
                the resulting `head_dim` is not divisible by 4 (a RoPE requirement).
        """
        super().__init__()
        if channels <= 0:
            self.is_identity = True
            return

        self.is_identity = False
        assert channels % num_heads == 0, (
            f"Channels ({channels}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.scale = self.head_dim**-0.5

        assert self.head_dim % 4 == 0, (
            f"Head dimension ({self.head_dim}) must be divisible by 4 for 2D RoPE.")

        self.qkv_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels * 3,
            kernel_size=3,
            padding=1,
            bias=qkv_bias,
            padding_mode=padding_mode,
            groups=channels  # Depthwise convolution
        )

        self.rope = RotaryEmbedding2D(dim=self.head_dim, base=channels, alpha=alpha)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the forward pass for the attention module.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: The output tensor with the same shape as the input,
                after applying attention and adding a residual connection.
        """
        if self.is_identity:
            return x

        B, C, H, W = x.shape
        N = H * W  # Sequence length

        # 1. Project to Q, K, V and reshape for multi-head attention.
        qkv = self.qkv_projection(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2),
            qkv
        ) # Shape of q, k, v: (B, num_heads, N, head_dim)

        # 2. Reshape and apply 2D Rotary Positional Embedding to Q and K.
        # Permute to (B, N, num_heads, head_dim) for RoPE.
        q_for_rope = q.permute(0, 2, 1, 3)
        k_for_rope = k.permute(0, 2, 1, 3)

        q_rotated, k_rotated = self.rope(q_for_rope, k_for_rope, height=H, width=W)

        # Permute back to (B, num_heads, N, head_dim) for attention.
        q = q_rotated.permute(0, 2, 1, 3)
        k = k_rotated.permute(0, 2, 1, 3)

        # 3. Compute scaled dot-product attention.
        out = F.scaled_dot_product_attention(q, k, v)

        return x + out.transpose(-1, -2).reshape(B, C, H, W)


class GRN(nn.Module):
    """Global Response Normalization (GRN) layer.

    As proposed in the ConvNeXt V2 paper, GRN enhances feature competition
    by normalizing feature maps based on their global response.

    Attributes:
        gamma: A learnable scaling parameter.
        beta: A learnable shifting parameter.
        eps: A small value to prevent division by zero.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        """Initializes the GRN module.

        Args:
            channels: The number of input channels.
            eps: A small epsilon value for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the GRN module."""
        # Calculate global feature descriptor (L2 norm)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        # Normalize the global descriptor
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        # Apply the GRN transformation
        return self.gamma * (x * nx) + self.beta + x


class GRN1D(nn.Module):
    """Global Response Normalization (GRN) layer for 1D inputs.

    This is a variant of GRN designed for 1D inputs, such as time series or
    embeddings. It normalizes the input based on its global response.

    Attributes:
        gamma: A learnable scaling parameter.
        beta: A learnable shifting parameter.
        eps: A small value to prevent division by zero.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        """Initializes the GRN1D module.

        Args:
            channels: The number of input channels.
            eps: A small epsilon value for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels))
        self.beta = nn.Parameter(torch.zeros(1, channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the GRN1D module."""
        gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class Mlp(nn.Module):
    """Multi-Layer Perceptron for spatial features.

    This module implements a simple MLP with two pointwise convolutions,
    an activation function, and a normalization layer. It is typically used
    in the U-Net architecture to process spatial features.

    Attributes:
        pw_conv1: The first pointwise convolution layer.
        act: The activation function applied after the first convolution.
        norm2: The normalization layer applied after the activation.
        pw_conv2: The second pointwise convolution layer.
        drop: The dropout layer applied after the second convolution.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ):
        """Initializes the Mlp module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels in the hidden layer. Defaults to
                `in_channels * 4`.
            out_channels: Number of output channels. Defaults to `in_channels`.
            act_layer: The activation function to use. Defaults to nn.GELU.
            bias: If True, adds a learnable bias to convolutions. Defaults to True.
            drop: Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels * 4

        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=bias)
        self.act = act_layer()
        self.norm2 = GRN(hidden_channels)
        self.pw_conv2 = nn.Conv2d(hidden_channels, out_channels, 1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the Mlp module."""
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.pw_conv2(x)
        x = self.drop(x)
        return x


class RoDitBlock(nn.Module):
    """A Diffusion transformer like convolutional block with rotary embeddings.

    This block forms the main building block of the U-Net at deeper levels. It
    contains two sub-blocks: a self-attention block and an MLP block. Both are
    modulated by an input embedding.

    Attributes:
        channels: Number of input and output channels. If 0, the block acts as
            an identity function.
        is_identity: If True, the block acts as an identity function.
        norm1: Group normalization before the attention block.
        attn: The multi-head self-attention module.
        eca: Efficient Channel Attention for the attention output.
        norm2: Group normalization before the MLP block.
        mlp: The MLP module.
        emb_mlp: A small MLP to process the input embedding.
        adaLN_modulation: A linear layer to generate shift and scale parameters
            from the input embedding.
    """

    def __init__(
        self,
        channels: int,
        emb_dim: int,
        num_heads: int = 8,
        num_groups: int = 4,
        dropout: float = 0.0,
        padding_mode: str = "circular",
        alpha: float = 1.0,
    ):
        """Initializes the RoDitBlock.

        Args:
            channels: Number of input and output channels.
            emb_dim: Dimension of the input embedding (e.g., time embedding).
            num_heads: Number of attention heads.
            num_groups: Number of groups for GroupNorm. Will be adjusted to be a
                divisor of `channels`.
            dropout: Dropout rate for the MLP block.
            padding_mode: Padding mode for convolutions.
            alpha: NTK interpolation scaling factor for RoPE. Defaults to 1.0 (no scaling).
        """
        super().__init__()
        self.channels = channels
        self.is_identity = channels <= 0
        if self.is_identity:
            return

        # Ensure num_groups is a valid divisor of channels
        if channels % num_groups != 0:
            valid_divisors = [g for g in range(num_groups, 0, -1) if channels % g == 0]
            num_groups = valid_divisors[0] if valid_divisors else 1

        self.norm1 = nn.GroupNorm(num_groups, channels, affine=False)
        self.attn = MSAWithRoPE(
            channels=self.channels,
            num_heads=num_heads,
            padding_mode=padding_mode,
            alpha=alpha,
        )
        self.eca = ECA(channels)
        self.norm2 = nn.GroupNorm(num_groups, channels, affine=False)
        self.mlp = Mlp(
            channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )
        self.emb_mlp = nn.Sequential(
            nn.RMSNorm(emb_dim, eps=1e-6),
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(approximate="tanh"),
            GRN1D(4 * emb_dim),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )
        modulation_dim = channels * 6  # 6 for: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, modulation_dim, bias=True)
        )
        # Initialize the final modulation projection to be zero
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Forward pass for the RoDitBlock.

        Args:
            x: Input tensor of shape (B, C, H, W).
            emb: Embedding tensor of shape (B, emb_dim).

        Returns:
            Output tensor of the same shape as `x`.
        """
        if self.is_identity:
            return x

        emb = self.emb_mlp(emb)

        # Generate modulation parameters from the embedding
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(emb).chunk(6, dim=1)

        # First residual sub-block (Attention)
        h_attn = self.norm1(x)
        h_attn = modulate(h_attn, shift_msa, scale_msa)
        h_attn = self.attn(h_attn)
        h_attn = self.eca(h_attn)
        h = x + gate_msa.unsqueeze(-1).unsqueeze(-1) * h_attn

        # Second residual sub-block (MLP)
        h_mlp = self.norm2(h)
        h_mlp = modulate(h_mlp, shift_mlp, scale_mlp)
        h_mlp = self.mlp(h_mlp)
        return h + gate_mlp.unsqueeze(-1).unsqueeze(-1) * h_mlp


class ResnetBlock(nn.Module):
    """A ResNet-like block with depthwise convolution and ECA.

    This block is designed for use in diffusion models, combining depthwise
    convolution, Efficient Channel Attention (ECA), and a multi-layer perceptron
    (MLP) for feature processing. It supports modulation based on an input
    embedding, allowing dynamic adjustment of the block's behavior.

    Attributes:
        channels: Number of input and output channels. If 0, the block acts as
            an identity function.
        is_identity: If True, the block acts as an identity function.
        norm1: Group normalization before the depthwise convolution.
        dw_conv: Depthwise convolution layer.
        eca: Efficient Channel Attention for the depthwise convolution output.
        norm2: Group normalization before the MLP block.
        mlp: The MLP module.
        emb_mlp: A small MLP to process the input embedding.
        adaLN_modulation: A linear layer to generate modulation parameters
            from the input embedding.
    """
    def __init__(
            self,
            channels: int,
            emb_dim: int,
            num_groups: int = 4,
            dropout: float = 0.0,
            padding_mode: str = "circular",
    ):
        """Initializes the ResnetBlock.

        Args:
            channels: Number of input and output channels.
            emb_dim: Dimension of the input embedding (e.g., time embedding).
            num_groups: Number of groups for GroupNorm. Will be adjusted to be a
                divisor of `channels`.
            dropout: Dropout rate for the MLP block.
            padding_mode: Padding mode for convolutions.
        """
        super().__init__()
        self.channels = channels
        self.is_identity = channels <= 0
        if self.is_identity:
            return

        # Ensure num_groups is a valid divisor of channels
        if channels % num_groups != 0:
            valid_divisors = [g for g in range(num_groups, 0, -1) if channels % g == 0]
            num_groups = valid_divisors[0] if valid_divisors else 1

        self.norm1 = nn.GroupNorm(num_groups, channels, affine=False)
        self.dw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            groups=channels  # Depthwise convolution
        )
        self.eca = ECA(channels)
        self.norm2 = nn.GroupNorm(num_groups, channels, affine=False)
        self.mlp = Mlp(
            channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )
        self.emb_mlp = nn.Sequential(
            nn.RMSNorm(emb_dim, eps=1e-6),
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(approximate="tanh"),
            GRN1D(4 * emb_dim),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )
        modulation_dim = channels * 6  # 6 for: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, modulation_dim, bias=True)
        )
        # Initialize the final modulation projection to be zero
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Forward pass for the RoDitBlock.

        Args:
            x: Input tensor of shape (B, C, H, W).
            emb: Embedding tensor of shape (B, emb_dim).

        Returns:
            Output tensor of the same shape as `x`.
        """
        if self.is_identity:
            return x

        emb = self.emb_mlp(emb)

        # Generate modulation parameters from the embedding
        shift_dwc, scale_dwc, gate_dwc, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(emb).chunk(6, dim=1)

        # First residual sub-block (depthwise convolution)
        h_dwc = self.norm1(x)
        h_dwc = modulate(h_dwc, shift_dwc, scale_dwc)
        h_dwc = self.dw_conv(h_dwc)
        h_dwc = self.eca(h_dwc)
        h = x + gate_dwc.unsqueeze(-1).unsqueeze(-1) * h_dwc

        # Second residual sub-block (MLP)
        h_mlp = self.norm2(h)
        h_mlp = modulate(h_mlp, shift_mlp, scale_mlp)
        h_mlp = self.mlp(h_mlp)
        return h + gate_mlp.unsqueeze(-1).unsqueeze(-1) * h_mlp


# ==============================================================================
# Downsampling & Upsampling Modules
# ==============================================================================

class Downsample(nn.Module):
    """Downsampling layer using a mix of pooling and strided convolution.

    This layer combines features from a strided convolution, average pooling,
    and max pooling to create a robust downsampled representation.

    If the input or output channels are zero, it acts as an identity function,
    returning an empty tensor with the appropriate spatial dimensions.

    Attributes:
        out_channels: Number of output channels.
        is_identity: If True, the module acts as an identity function.
        conv_in: A depthwise convolution layer to process the input.
        avg_pool: Average pooling layer for downsampling.
        max_pool: Max pooling layer for downsampling.
        conv_out: A 1x1 convolution to merge the feature maps.
    """

    def __init__(self, in_channels: int, out_channels: int, padding_mode: str = "circular"):
        """Initializes the Downsample module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            padding_mode: Padding mode for the strided convolution.
        """
        super().__init__()
        self.out_channels = out_channels
        self.is_identity = in_channels == 0 or out_channels == 0
        if self.is_identity:
            return

        self.conv_in = nn.Conv2d(
            in_channels, in_channels,
            3,
            stride=2,
            padding=1,
            padding_mode=padding_mode,
            groups=in_channels
        )
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        # 1x1 convolution to merge the three feature maps
        self.conv_out = nn.Conv2d(
            in_channels * 3, out_channels,
            1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the Downsample module."""
        if self.is_identity:
            B, _, H, W = x.shape
            new_H, new_W = H // 2, W // 2
            return torch.zeros((B, 0, new_H, new_W), dtype=x.dtype, device=x.device)

        features = torch.cat([self.conv_in(x), self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.conv_out(features)


class Upsample(nn.Module):
    """Upsampling layer using PixelShuffle.

    This layer first increases the number of channels and then rearranges
    them into a higher resolution feature map using PixelShuffle.
    The final convolution can refine the features.

    If the input or output channels are zero, it acts as an identity function,
    returning an empty tensor with the appropriate spatial dimensions.

    Attributes:
        out_channels: Number of output channels.
        scale_factor: The factor by which to increase spatial resolution.
        is_identity: If True, the module acts as an identity function.
        conv1: A depthwise convolution layer to increase channels.
        pixel_shuffle: PixelShuffle layer to rearrange channels into spatial dimensions.
        conv2: A 1x1 convolution to refine the output features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        padding_mode: str = "circular"
    ):
        """Initializes the Upsample module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            scale_factor: The factor by which to increase spatial resolution.
            padding_mode: Padding mode for convolutions.
        """
        super().__init__()
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.is_identity = in_channels == 0 or out_channels == 0

        self.conv1 = nn.Conv2d(
            in_channels, out_channels * (scale_factor ** 2),
            3,
            padding=1,
            padding_mode=padding_mode,
            groups=in_channels
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the Upsample module."""
        if self.is_identity:
            B, _, H, W = x.shape
            new_H, new_W = H * self.scale_factor, W * self.scale_factor
            return torch.zeros((B, 0, new_H, new_W), dtype=x.dtype, device=x.device)

        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        return x


# ==============================================================================
# Main Model: RoDitUnet
# ==============================================================================
class RoDitUnet(nn.Module):
    """Rotary Embedding Diffusion Transformer U-Net (RoDitUnet).

    This model architecture follows the standard U-Net pattern with an encoder,
    a bottleneck, and a decoder with skip connections. It uses RoDitBlock as
    the main building block and supports time and conditional embeddings.

    The `start_attn_level` parameter allows using ResnetBlocks for shallower
    layers and switching to RoDitBlocks (with self-attention) at deeper levels
    to balance performance and computational cost.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        model_channels: Base number of channels for the first level of the U-Net.
        channel_mult: A sequence of multipliers for the channel count at each
            level of the U-Net.
        start_attn_level: The level index (0-based) at which to start using
            RoDitBlocks. Layers before this level will use ResnetBlocks.
        num_blocks: The number of Backbone Block at each level.
        dropout: The dropout rate used in the RoDitBlock.
        num_heads: The number of attention heads in the RoDitBlock.
        num_groups: The number of groups for GroupNorm in the RoDitBlock.
        num_conditions: The number of conditions (e.g., time embeddings).
        emb_dim: The dimension of the time and conditional embeddings passed to
            the RoDitBlock. Defaults to `model_channels`.
        padding_mode: The padding mode for all convolutions.
        alpha: The NTK interpolation scaling factor for RoPE. Defaults to 1.0 (no scaling).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4,),
        start_attn_level: int = 0,
        num_blocks: int = 1,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_groups: int = 8,
        num_conditions: int = 0,
        emb_dim: Optional[int] = None,
        padding_mode: str = "circular",
        alpha: float = 1.0
    ):
        """Initializes the RoDitUnet model.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels in the output tensor.
            model_channels: The base number of channels for the first level of the U-Net.
            channel_mult: A sequence of multipliers for the channel count at each
                level of the U-Net.
            start_attn_level (int): The level index (0-based) at which to start
                using RoDitBlocks. Layers before this level will use ResnetBlocks.
                Defaults to 0 (all levels use attention).
            num_blocks: The number of Backbone Block at each level.
            dropout: The dropout rate used in the RoDitBlock.
            num_heads: The number of attention heads in the RoDitBlock.
            num_groups: The number of groups for GroupNorm in the RoDitBlock.
            num_conditions: The number of conditions.
            emb_dim: The dimension of the time and conditional embeddings passed to the RoDitBlock.
                Defaults to `model_channels`.
            padding_mode: The padding mode for all convolutions.
            alpha: The NTK interpolation scaling factor for RoPE. Defaults to 1.0 (no scaling).
        """
        super().__init__()
        if not channel_mult:
            raise ValueError("channel_mult cannot be empty.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = len(channel_mult)
        self.start_attn_level = start_attn_level

        # --- Embedding Setup ---
        emb_dim = emb_dim or model_channels
        if emb_dim % 2 != 0:
            emb_dim += 1
        self.emb_dim = emb_dim

        self.time_embedder = SinusoidalPosEmb(self.emb_dim)

        self.num_conditions = num_conditions
        if self.num_conditions > 0:
            self.cond_embedders = nn.ModuleList([SinusoidalPosEmb(self.emb_dim) for _ in range(num_conditions)])
            self.embedding_attention = ConditionalSAWithRoPE(dim=self.emb_dim)

        # --- U-Net Backbone Construction ---
        resnet_block_args = {
            "emb_dim": self.emb_dim,
            "num_groups": num_groups,
            "dropout": dropout,
            "padding_mode": padding_mode,
        }
        rodit_block_args = {
            **resnet_block_args,
            "num_heads": num_heads,
            "alpha": alpha,
        }

        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1, padding_mode=padding_mode)

        # === Encoder ===
        self.down_blocks = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        skip_channels = []
        current_ch = model_channels
        for i in range(self.num_levels):
            if i < self.start_attn_level:
                Block = ResnetBlock
                block_args = resnet_block_args
            else:
                Block = RoDitBlock
                block_args = rodit_block_args

            self.down_blocks.append(
                nn.ModuleList([Block(channels=current_ch, **block_args) for _ in range(num_blocks)]))

            skip_channels.append(current_ch)
            if i < self.num_levels - 1:
                next_ch = model_channels * channel_mult[i+1]
                self.down_samplers.append(Downsample(in_channels=current_ch, out_channels=next_ch))
                current_ch = next_ch

        # === Bottleneck ===
        ch = current_ch  # Channel count at the bottom of the U-Net

        self.middle_blocks = nn.ModuleList([
            RoDitBlock(channels=ch, **rodit_block_args),
            RoDitBlock(channels=ch, **rodit_block_args)
        ])

        # === Decoder ===
        self.up_blocks = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        self.up_proj_convs = nn.ModuleList()

        # Build decoder from the bottom up (deepest to shallowest)
        for i in reversed(range(self.num_levels)):
            target_ch = skip_channels.pop()

            # Create an upsampler to transition from the deeper layer's channel count (`ch`)
            # to the current level's channel count (`target_ch`).
            # No upsampler is needed for the first (deepest) decoder stage.
            if i < self.num_levels - 1:
                self.up_samplers.append(Upsample(in_channels=ch, out_channels=target_ch))

            # The projection convolution handles the channel merging after concatenation.
            # The input will be the concatenation of the upsampled feature map (`target_ch`)
            # and the skip connection (`target_ch`), so the input channel is `target_ch * 2`.
            proj_in_ch = target_ch * 2
            self.up_proj_convs.append(nn.Conv2d(proj_in_ch, target_ch, 1))

            # Add the main blocks for this level.
            if i < self.start_attn_level:
                Block = ResnetBlock
                block_args = resnet_block_args
            else:
                Block = RoDitBlock
                block_args = rodit_block_args

            self.up_blocks.append(
                nn.ModuleList([Block(channels=target_ch, **block_args) for _ in range(num_blocks)]))

            # Update `ch` for the next iteration (the level above).
            ch = target_ch

        # --- Final Output Layers ---
        if num_groups > ch > 0:
            num_groups = ch
        self.out_norm = nn.GroupNorm(num_groups, ch) if ch > 0 and ch % num_groups == 0 else nn.Identity()
        self.out_act = nn.SiLU()
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x: Tensor, time: Tensor, conditions: Optional[Sequence[Tensor]] = None, **keywords) -> Tensor:
        """The forward pass of the RoDitUnet model.

        Args:
            x: The input tensor (e.g., noisy image) of shape (B, C, H, W).
            time: A tensor of timesteps, shape (B,).
            conditions: An optional sequence of conditional tensors, each of shape (B,).

        Returns:
            The output tensor of the model (e.g., predicted noise), shape (B, out_C, H, W).

        Raises:
            ValueError: If input spatial dimensions are not divisible by the
                downsampling factor, or if the number of conditions is incorrect.
        """
        B, C, H, W = x.shape
        min_size = 2 ** (self.num_levels - 1)
        if H % min_size != 0 or W % min_size != 0:
            raise ValueError(f"Input H/W ({H}/{W}) must be divisible by the total downsampling factor {min_size}")

        # --- Process Embeddings ---
        # Ensure time tensor has the correct shape (B,)
        if time.ndim != 1 or time.shape[0] != B:
            time = time.flatten().expand(B)

        t_emb = self.time_embedder(time.to(x.device))
        final_emb = t_emb

        if self.num_conditions > 0:
            if conditions is None or len(conditions) != self.num_conditions:
                raise ValueError(f"Expected {self.num_conditions} conditions, but got {len(conditions) if conditions else 0}")

            embedding_sequence = [t_emb]

            for cond_index, cond_tensor in enumerate(conditions):
                if cond_tensor.ndim != 1 or cond_tensor.shape[0] != B:
                    cond_tensor = cond_tensor.flatten().expand(B)
                c_emb = self.cond_embedders[cond_index](cond_tensor.to(x.device))
                embedding_sequence.append(c_emb)

            final_emb = self.embedding_attention(embedding_sequence)

        # --- U-Net Forward Pass ---
        skips = []
        h = self.conv_in(x)

        # === Encoder ===
        for i in range(self.num_levels):
            for block in self.down_blocks[i]:
                h = block(h, final_emb)
            skips.append(h)
            if i < self.num_levels - 1:
                h = self.down_samplers[i](h)

        # === Bottleneck ===
        for block in self.middle_blocks:
            h = block(h, final_emb)

        # === Decoder ===
        # The decoder modules were built from deep to shallow, so we iterate through them directly.
        for i in range(self.num_levels):
            # Retrieve the corresponding skip connection from the encoder.
            skip_h = skips.pop()

            # The channel counts of `h` (from the deeper layer) and `skip_h` are now identical.
            # Concatenate them along the channel dimension.
            h = torch.cat([h, skip_h], dim=1)

            # Project the concatenated features to merge them and restore the channel count.
            h = self.up_proj_convs[i](h)

            # Perform feature refinement using the RoDitBlock for this level.
            for block in self.up_blocks[i]:
                h = block(h, final_emb)

            # Upsample the feature map for the next (shallower) level.
            # This is skipped for the last (shallowest) level of the decoder.
            if i < self.num_levels - 1:
                h = self.up_samplers[i](h)

        # === Output ===
        h = self.out_norm(h)
        h = self.out_act(h)
        return self.conv_out(h)


# ==============================================================================
# Test Suite
# ==============================================================================

class TestHelperFunctions(unittest.TestCase):
    """Test helper functions like modulate."""

    def test_modulate_4d(self):
        """Test modulate with 4D tensor."""
        x = torch.randn(2, 16, 8, 8)
        shift = torch.randn(2, 16)
        scale = torch.randn(2, 16)
        output = modulate(x, shift, scale)
        self.assertEqual(output.shape, x.shape)
        # Check if modulation was applied (output should not be equal to input)
        self.assertFalse(torch.allclose(output, x))

    def test_modulate_2d(self):
        """Test modulate with 2D tensor."""
        x = torch.randn(2, 16)
        shift = torch.randn(2, 16)
        scale = torch.randn(2, 16)
        output = modulate(x, shift, scale)
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.allclose(output, x))

    def test_modulate_unsupported_dim(self):
        """Test modulate with unsupported tensor dimension."""
        with self.assertRaises(ValueError):
            modulate(torch.randn(2, 16, 8), torch.randn(2, 16), torch.randn(2, 16))


class TestCoreModules(unittest.TestCase):
    """Tests for core neural network modules."""

    def setUp(self):
        """Set up common variables for tests."""
        self.batch_size = 2
        self.channels = 32
        self.height = 16
        self.width = 16
        self.emb_dim = 64
        self.seq_len = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_sinusoidal_pos_emb(self):
        """Test SinusoidalPosEmb module."""
        emb = SinusoidalPosEmb(self.emb_dim - 1)  # Test odd dimension correction
        self.assertEqual(emb.embedding_dim, self.emb_dim)
        x = torch.randint(0, 100, (self.batch_size,)).to(self.device)
        output = emb(x)
        self.assertEqual(output.shape, (self.batch_size, self.emb_dim))
        with self.assertRaises(ValueError):
            SinusoidalPosEmb(0)

    def test_eca(self):
        """Test ECA module."""
        eca = ECA(self.channels).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = eca(x)
        self.assertEqual(output.shape, x.shape)

        # Test identity case
        eca_identity = ECA(0)
        output_identity = eca_identity(x)
        self.assertTrue(torch.allclose(x, output_identity))

    def test_rotary_embedding_1d(self):
        """Test RotaryEmbedding1D module."""
        rope = RotaryEmbedding1D(dim=self.emb_dim).to(self.device)
        q = torch.randn(self.batch_size, self.seq_len, self.emb_dim).to(self.device)
        k = torch.randn(self.batch_size, self.seq_len, self.emb_dim).to(self.device)
        q_rot, k_rot = rope(q, k)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        self.assertFalse(torch.allclose(q_rot, q))
        with self.assertRaises(ValueError):
            RotaryEmbedding1D(dim=31)  # Odd dimension

    def test_rotary_embedding_2d(self):
        """Test RotaryEmbedding2D module."""
        dim = 32  # Must be divisible by 4
        num_heads = 4
        head_dim = dim // num_heads
        rope = RotaryEmbedding2D(dim=head_dim, alpha=1.5).to(self.device)
        q = torch.randn(self.batch_size, self.height * self.width, num_heads, head_dim).to(self.device)
        k = torch.randn(self.batch_size, self.height * self.width, num_heads, head_dim).to(self.device)
        q_rot, k_rot = rope(q, k, self.height, self.width)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        self.assertFalse(torch.allclose(q_rot, q))
        with self.assertRaises(ValueError):
            RotaryEmbedding2D(dim=10)  # Not divisible by 4

    def test_conditional_sa_with_rope(self):
        """Test ConditionalSAWithRoPE module."""
        cond_sa = ConditionalSAWithRoPE(dim=self.emb_dim).to(self.device)
        emb1 = torch.randn(self.batch_size, self.emb_dim).to(self.device)
        emb2 = torch.randn(self.batch_size, self.emb_dim).to(self.device)
        output = cond_sa([emb1, emb2])
        self.assertEqual(output.shape, (self.batch_size, self.emb_dim))

    def test_msa_with_rope(self):
        """Test MSAWithRoPE module."""
        num_heads = 4
        msa = MSAWithRoPE(channels=self.channels, num_heads=num_heads).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = msa(x)
        self.assertEqual(output.shape, x.shape)

        # Test assertion errors
        with self.assertRaises(AssertionError):
            MSAWithRoPE(channels=30, num_heads=4)  # Channels not divisible by heads
        with self.assertRaises(AssertionError):
            MSAWithRoPE(channels=32, num_heads=5)  # Head dim not div by 4

    def test_grn(self):
        """Test GRN module."""
        grn = GRN(self.channels).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)

        # Verify initial state: When gamma and beta are 0, the output should be equal to the input (identity mapping).
        # This is a check for the correctness of the initialization behavior.
        initial_output = grn(x)
        self.assertTrue(torch.allclose(initial_output, x),
                        "GRN's initial output should be identical to its input due to zero initialization.")

        with torch.no_grad():
            # 將 gamma 和 beta 的值都設為 1
            grn.gamma.fill_(1.0)
            grn.beta.fill_(1.0)

        transformed_output = grn(x)
        self.assertEqual(transformed_output.shape, x.shape)
        self.assertFalse(torch.allclose(transformed_output, x),
                         "After setting parameters to non-zero, GRN should transform the input.")

    def test_grn1d(self):
        """Test GRN1D module."""
        grn1d = GRN1D(self.channels).to(self.device)
        x = torch.randn(self.batch_size, self.channels).to(self.device)

        # Verify initial state: When gamma and beta are 0, the output should be equal to the input (identity mapping).
        initial_output = grn1d(x)
        self.assertTrue(torch.allclose(initial_output, x),
                        "GRN1D's initial output should be identical to its input due to zero initialization.")

        with torch.no_grad():
            # Set gamma and beta to 1
            grn1d.gamma.fill_(1.0)
            grn1d.beta.fill_(1.0)

        transformed_output = grn1d(x)
        self.assertEqual(transformed_output.shape, x.shape)
        self.assertFalse(torch.allclose(transformed_output, x),
                         "After setting parameters to non-zero, GRN1D should transform the input.")

    def test_mlp(self):
        """Test Mlp module."""
        mlp = Mlp(in_channels=self.channels, out_channels=self.channels * 2).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = mlp(x)
        self.assertEqual(output.shape, (self.batch_size, self.channels * 2, self.height, self.width))


class TestBuildingBlocks(unittest.TestCase):
    """Tests for composite blocks like RoDitBlock, Upsample, and Downsample."""

    def setUp(self):
        self.batch_size = 2
        self.channels = 32
        self.height = 16
        self.width = 16
        self.emb_dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_rodit_block(self):
        """Test RoDitBlock module."""
        block = RoDitBlock(channels=self.channels, emb_dim=self.emb_dim, num_heads=4).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        emb = torch.randn(self.batch_size, self.emb_dim).to(self.device)
        output = block(x, emb)
        self.assertEqual(output.shape, x.shape)

        # Test identity case
        block_identity = RoDitBlock(0, 0)
        output_identity = block_identity(x, emb)
        self.assertTrue(torch.allclose(x, output_identity))

    def test_resnet_block(self):
        """Test the ResnetBlock module for correct output shape and identity case."""
        block = ResnetBlock(channels=self.channels, emb_dim=self.emb_dim).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        emb = torch.randn(self.batch_size, self.emb_dim).to(self.device)
        output = block(x, emb)
        self.assertEqual(output.shape, x.shape)

        # Test identity case when channels are 0
        block_identity = ResnetBlock(0, 0)
        output_identity = block_identity(x, emb)
        self.assertTrue(torch.allclose(x, output_identity))

    def test_upsample(self):
        """Test Upsample module."""
        scale_factor = 2
        up = Upsample(in_channels=self.channels, out_channels=self.channels // 2, scale_factor=scale_factor).to(
            self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = up(x)
        expected_shape = (self.batch_size, self.channels // 2, self.height * scale_factor, self.width * scale_factor)
        self.assertEqual(output.shape, expected_shape)

    def test_downsample(self):
        """Test Downsample module."""
        down = Downsample(in_channels=self.channels, out_channels=self.channels * 2).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = down(x)
        expected_shape = (self.batch_size, self.channels * 2, self.height // 2, self.width // 2)
        self.assertEqual(output.shape, expected_shape)


class TestRoDitUnet(unittest.TestCase):
    """Comprehensive tests for the full RoDitUnet model."""

    def setUp(self):
        """Set up common variables for the U-Net tests."""
        self.batch_size = 2
        self.in_channels = 3
        self.out_channels = 3
        self.height = 32  # Must be divisible by 2^(num_levels-1)
        self.width = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_config = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "model_channels": 32,
            "channel_mult": (1, 2, 4),  # 3 levels (0, 1, 2)
            "num_blocks": 2,
            "num_heads": 4,
        }

    def test_forward_pass_without_conditions(self):
        """Test a standard forward pass without any conditions."""
        model = RoDitUnet(**self.base_config, num_conditions=0).to(self.device)
        x = torch.randn(self.batch_size, self.in_channels, self.height, self.width).to(self.device)
        time = torch.randint(0, 100, (self.batch_size,)).to(self.device)
        output = model(x, time)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.height, self.width))

    def test_forward_pass_with_conditions(self):
        """Test a forward pass with conditional inputs and check error handling."""
        num_cond = 2
        model = RoDitUnet(**self.base_config, num_conditions=num_cond).to(self.device)
        x = torch.randn(self.batch_size, self.in_channels, self.height, self.width).to(self.device)
        time = torch.randint(0, 100, (self.batch_size,)).to(self.device)
        conditions = [torch.rand(self.batch_size).to(self.device) for _ in range(num_cond)]

        # Test successful forward pass
        output = model(x, time, conditions=conditions)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.height, self.width))

        # Test ValueError with the wrong number of conditions
        with self.assertRaisesRegex(ValueError, "Expected 2 conditions, but got 1"):
            model(x, time, conditions=[conditions[0]])

    def test_architecture_based_on_start_attn_level(self):
        """Verify that `start_attn_level` correctly configures the block types."""
        num_levels = len(self.base_config["channel_mult"])

        # Define test scenarios: start_level and expected block types for encoder/decoder
        test_cases = [
            {
                "desc": "All levels use attention",
                "start_attn_level": 0,
                "expected_block": RoDitBlock,
            },
            {
                "desc": "Attention starts at level 1",
                "start_attn_level": 1,
                "expected_block": [ResnetBlock, RoDitBlock, RoDitBlock],
            },
            {
                "desc": "Attention starts at the last level",
                "start_attn_level": num_levels - 1,
                "expected_block": [ResnetBlock, ResnetBlock, RoDitBlock],
            },
            {
                "desc": "Only bottleneck uses attention",
                "start_attn_level": num_levels,
                "expected_block": ResnetBlock,
            },
        ]

        for case in test_cases:
            with self.subTest(desc=case["desc"]):
                model = RoDitUnet(**self.base_config, start_attn_level=case["start_attn_level"])

                # 1. Verify Encoder (down_blocks)
                for level_idx, level_blocks in enumerate(model.down_blocks):
                    expected = case["expected_block"]
                    BlockType = expected[level_idx] if isinstance(expected, list) else expected
                    for block in level_blocks:
                        self.assertIsInstance(block, BlockType,
                                              f"Down Block at level {level_idx} should be {BlockType.__name__}")

                # 2. Verify Bottleneck (middle_blocks) - should always be RoDitBlock
                for block in model.middle_blocks:
                    self.assertIsInstance(block, RoDitBlock,
                                          "Bottleneck block should always be RoDitBlock")

                # 3. Verify Decoder (up_blocks)
                # up_blocks are ordered from deep to shallow (level 2, 1, 0)
                for i, level_blocks in enumerate(model.up_blocks):
                    level_idx = num_levels - 1 - i  # Convert up_block index to level index
                    expected = case["expected_block"]
                    BlockType = expected[level_idx] if isinstance(expected, list) else expected
                    for block in level_blocks:
                        self.assertIsInstance(block, BlockType,
                                              f"Up Block at level {level_idx} should be {BlockType.__name__}")

    def test_input_size_error_handling(self):
        """Test for ValueError when input spatial dimensions are not valid."""
        model = RoDitUnet(**self.base_config).to(self.device)
        # Total downsampling is 2^(num_levels-1) = 2^2 = 4. Input must be divisible by 4.
        invalid_size = self.height - 1  # 31 is not divisible by 4
        x = torch.randn(self.batch_size, self.in_channels, invalid_size, invalid_size).to(self.device)
        time = torch.rand(self.batch_size).to(self.device)

        with self.assertRaisesRegex(ValueError, "must be divisible by the total downsampling factor"):
            model(x, time)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
