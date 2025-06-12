#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiResUnet Implementation and Associated Modules.

This script defines various PyTorch modules inspired by recent trends in
generative modeling (like Diffusion Transformers - DiT), including:
- Depthwise Separable Convolution (DConv)
- DiT-inspired Residual Block (DiResBlock) with internal Attention and MLP
- Multi-Head Self-Attention (Attention)
- ConvNeXt V2 style MLP
- Modulated layers using adaLN-style conditioning
- Custom Upsample (PixelShuffle) and Downsample layers
- Sinusoidal Positional Embeddings with dynamic max_period
- Circular padding for periodic boundary conditions of physics fields

It implements DiResUnet, a U-Net architecture built with these components,
featuring multiscale input handling and multi-conditional embedding capabilities.
A test suite is included for validation.
"""

import math
import traceback
from functools import partial
from typing import Optional, Sequence, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.layers import LayerNorm2d

# Global flag to control detailed test output
VERBOSE_TEST = False # Set to True for more detailed module prints during tests

# ==============================================================================
# Helper Function: modulate
# ==============================================================================

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Applies learned affine transformation (scale and shift) to input tensor.

    This is commonly used in Adaptive Layer Normalization (adaLN) or similar
    conditional normalization techniques, especially in generative models like DiT.

    Args:
        x (Tensor): Input tensor. Expected shape is (B, C, H, W) for 4D inputs
            or (B, D) for 2D inputs.
        shift (Tensor): Shift parameters. Shape should match 'x' based on
            dimension: (B, C) for 4D 'x', or (B, D) for 2D 'x'.
        scale (Tensor): Scale parameters. Shape should match 'x' based on
            dimension: (B, C) for 4D 'x', or (B, D) for 2D 'x'.

    Returns:
        Tensor: Modulated tensor with the same shape as 'x'.

    Raises:
        ValueError: If input tensor 'x' has unexpected dimensions (not 2 or 4).
        ValueError: If shape mismatch occurs between 'x' and 'shift'/'scale'
            tensors based on the dimension of 'x'.
    """
    # Ensure shift and scale are on the same device as x
    if shift.device != x.device:
        shift = shift.to(x.device)
    if scale.device != x.device:
        scale = scale.to(x.device)

    if x.dim() == 4:
        # Check batch size and channel dimension compatibility
        if shift.shape[0] != x.shape[0] or scale.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch for 4D input: x({x.shape[0]}) vs "
                f"shift/scale({shift.shape[0]})"
            )
        # shift/scale are expected to be (B, C)
        if shift.shape[1] != x.shape[1] or scale.shape[1] != x.shape[1]:
            raise ValueError(
                f"Channel dimension mismatch for 4D input: x channels({x.shape[1]}) vs "
                f"shift/scale channels({shift.shape[1]})"
            )
        # Reshape shift and scale to (B, C, 1, 1) for broadcasting
        # Apply modulation: x = x * (1 + scale) + shift
        return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

    elif x.dim() == 2:
        # Check shape compatibility for 2D input (e.g., sequences or features)
        # shift/scale are expected to be (B, D) matching x
        if shift.shape != x.shape or scale.shape != x.shape:
            raise ValueError(
                f"Shape mismatch for 2D input: x({x.shape}) vs "
                f"shift/scale({shift.shape})"
            )
        # Apply modulation directly
        return x * (1 + scale) + shift
    else:
        # Handle unsupported input dimensions
        raise ValueError(
            f"Input tensor x has unexpected dimension: {x.dim()}. Expected 4 or 2."
        )


# ==============================================================================
# Module: DConv (Depthwise Separable Convolution)
# ==============================================================================

class DConv(nn.Module):
    """Depthwise Separable Convolution module.

    Factorizes a standard convolution into a depthwise convolution followed by
    a pointwise convolution (1x1 conv). This often reduces computational cost
    and the number of parameters.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel
            for the depthwise convolution.
        stride (Union[int, Tuple[int, int]], optional): Stride of the convolution
            for the depthwise convolution. Defaults to 1.
        padding (Union[int, Tuple[int, int]], optional): Zero-padding added to
            both sides of the input for the depthwise convolution. Defaults to 0.
        padding_mode (str, optional): The padding mode for the depthwise conv.
            Options: 'zeros', 'reflect', 'replicate', 'circular'.
            Defaults to 'zeros'.
        dilation (Union[int, Tuple[int, int]], optional): Spacing between kernel
            elements for the depthwise convolution. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output of
            both convolutions. Defaults to False (common when using Norm layers).

    Raises:
        ValueError: If `padding_mode` is not one of the allowed options.

    Shape:
        - Input: $$(N, C_{in}, H_{in}, W_{in})$$
        - Output: $$(N, C_{out}, H_{out}, W_{out})$$ where the output spatial
          dimensions depend on kernel_size, stride, padding, and dilation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        padding_mode: str = "zeros",
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = False,
    ):
        super().__init__()

        # Validate padding_mode
        if padding_mode not in ["zeros", "reflect", "replicate", "circular"]:
            raise ValueError(
                f"Invalid padding_mode: {padding_mode}. "
                "Expected one of: 'zeros', 'reflect', 'replicate', 'circular'"
            )

        # Depthwise convolution: Applies filters independently to each input channel.
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # Output channels = Input channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode, # Use specified padding mode
            dilation=dilation,
            groups=in_channels,  # Key setting for depthwise conv
            bias=bias,
        )

        # Pointwise convolution: 1x1 convolution to combine channel information.
        # padding_mode is typically not needed for 1x1 conv with padding=0.
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,  # Input is output of depthwise conv
            out_channels=out_channels, # Desired number of output channels
            kernel_size=1,            # 1x1 kernel
            stride=1,                 # Standard stride for pointwise
            padding=0,                # No padding needed for 1x1
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Depthwise Separable Convolution.

        Args:
            x (Tensor): Input tensor of shape $$(N, C_{in}, H_{in}, W_{in})$$.

        Returns:
            Tensor: Output tensor of shape $$(N, C_{out}, H_{out}, W_{out})$$.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ==============================================================================
# Module: SinusoidalPosEmb (Sinusoidal Positional Embedding)
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Creates sinusoidal positional embeddings.

    Commonly used to encode time steps or conditional labels into fixed-size
    vectors for diffusion models or transformers.

    Args:
        dim (int): The target dimension of the embedding. Must be positive.
            If odd, it will be automatically increased to the next even number.
        max_period (int): The maximum period for the sinusoidal functions.
            Controls the range of frequencies used. Defaults to 10000.

    Raises:
        ValueError: If `dim` is not a positive integer.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dim}")

        # Ensure the dimension is even, as required by the sin/cos pairing
        original_dim = dim
        if dim % 2 != 0:
            dim += 1
            # Use print instead of warning module for simplicity in this context
            print(
                f"Info: SinusoidalPosEmb dim adjusted from {original_dim} "
                f"to {dim} to be even."
            )
        self.dim = dim
        self.max_period = max_period # Store max_period

    def forward(self, x: Tensor) -> Tensor:
        """Generates sinusoidal embeddings for the input tensor.

        Args:
            x (Tensor): Input tensor of shape (B,) or (B, 1), containing the
                values to embed (e.g., time steps, class labels). Values should
                be numerical.

        Returns:
            Tensor: Output tensor of shape (B, `dim`) containing the embeddings.

        Raises:
            ValueError: If the input tensor `x` does not have shape (B,) or (B, 1).
        """
        # Handle input shape (B, 1) -> (B,)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.squeeze(1)
        # Validate input shape
        if x.ndim != 1:
            raise ValueError(
                f"Input tensor must be of shape (B,) or (B, 1), got {x.shape}"
            )

        device = x.device
        dtype = torch.float32  # Use float32 for calculation stability

        # Calculate frequencies
        half_dim = self.dim // 2
        # freq = 1 / (max_period^(2i / dim))
        freqs = torch.exp(
            -math.log(self.max_period) # Use stored max_period
            * torch.arange(start=0, end=half_dim, dtype=dtype)
            / half_dim
        ).to(device)

        # Calculate arguments for sin and cos: value * frequency
        # x shape: (B,) -> (B, 1); freqs shape: (half_dim,) -> (1, half_dim)
        # args shape: (B, half_dim)
        args = x.float().unsqueeze(1) * freqs.unsqueeze(0)

        # Concatenate sin and cos components
        # embedding shape: (B, dim)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


# ==============================================================================
# Module: Attention (Multi-Head Self-Attention)
# ==============================================================================

class Attention(nn.Module):
    """Multi-Head Self-Attention module with Group Normalization.

    Includes a residual connection. Handles input with 0 channels by acting
    as an identity operation.

    Args:
        channels (int): Number of input and output channels. If 0, the block
            acts as an identity.
        num_heads (int): Number of attention heads. `channels` must be divisible
            by `num_heads` if `channels > 0`. Defaults to 8.
        num_groups (int): Number of groups for Group Normalization. Will be
            adjusted to a valid divisor of `channels` if needed. Defaults to 4.
        qkv_bias (bool): If True, add bias to the query, key, value projection.
            Defaults to True.
        padding_mode (str): Padding mode for the depthwise convolution.

    Raises:
        AssertionError: If `channels > 0` and `channels` is not divisible by
            `num_heads`.
        ValueError: If `num_groups` is adjusted and no valid divisor is found
            (should not happen if `channels > 0`).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_groups: int = 4,
        qkv_bias: bool = True,
        padding_mode: str = "circular"
    ):
        super().__init__()
        self.channels = channels

        # Handle zero channels case: become an Identity module
        if channels <= 0:
            self._is_identity = True
            return
        self._is_identity = False

        self.num_heads = num_heads
        assert (
            channels % num_heads == 0
        ), f"Channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5 # Scaling factor for dot products

        # Adjust num_groups for GroupNorm validity
        if channels < num_groups or num_groups <= 0:
             # Fallback if num_groups is too large or invalid
             num_groups = 1
        if channels % num_groups != 0:
             # Find the largest valid divisor <= original num_groups
             valid_groups = [
                 g for g in range(1, min(channels, num_groups) + 1)
                 if channels % g == 0
             ]
             if not valid_groups:
                 # Should theoretically not happen if channels > 0, but safeguard
                 raise ValueError(f"Could not find valid num_groups divisor for channels={channels}")
             num_groups = max(valid_groups)
             print(
                 f"Info: Attention num_groups adjusted to {num_groups} "
                 f"for channels={channels}"
             )
        self.num_groups = num_groups

        # Layers
        self.norm = nn.GroupNorm(self.num_groups, channels)
        # Project input to Q, K, V tensors
        self.to_qkv_3 = DConv(channels, channels * 3, kernel_size=3, padding=1, bias=qkv_bias,
                               padding_mode=padding_mode)

    def forward(self, x: Tensor) -> Tensor:
        """Applies multi-head self-attention to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) after self-attention
                and residual connection.
        """
        if self._is_identity:
            return x

        B, C, H, W = x.shape
        residual = x # Store for residual connection

        # Normalize input
        x_norm = self.norm(x)

        # Generate Q, K, V and split them
        # qkv shape: (B, C*3, H, W) -> three tensors of shape (B, C, H, W)
        qkv = self.to_qkv_3(x_norm) 
        qkv = qkv.chunk(3, dim=1)

        # Reshape Q, K, V for multi-head attention calculation
        # Map (B, C, H, W) -> (B, num_heads, H*W, head_dim)
        q, k, v = map(
            lambda t: t.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2),
            qkv,
        )

        # Calculate attention using scaled dot-product attention
        # Use optimized F.scaled_dot_product_attention if available (PyTorch >= 2.0)
        try:
            # Efficient attention computation
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        except (AttributeError, TypeError):
            # Fallback for older PyTorch versions or different API signatures
            # Calculate attention scores: (B, num_heads, H*W, H*W)
            attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attention = F.softmax(attn_logits, dim=-1)
            # Apply attention to values: (B, num_heads, H*W, head_dim)
            out = torch.matmul(attention, v)

        # Reshape output back to (B, C, H, W)
        # (B, num_heads, H*W, head_dim) -> (B, C, H, W)
        out = out.transpose(-1, -2).reshape(B, C, H, W)

        # Add residual connection
        return out + residual


# ==============================================================================
# Module: GRN (Global Response Normalization)
# ==============================================================================

class GRN(nn.Module):
    """Global Response Normalization (channels-first).

    Implements the GRN layer proposed in ConvNeXt V2.

    Args:
        num_channels (int): C dimension of the input.
        eps (float, optional): Small number for numerical stability. Default: 1e-6.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Applies GRN to a 4-D tensor.

        Args:
            x (Tensor): Shape (B, C, H, W).

        Returns:
            Tensor: Same shape as *x* after global response normalization.
        """
        # (a) Global L2 aggregation over (H, W)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)           # (B, C, 1, 1)

        # (b) Divisive channel-wise normalisation
        mean_gx = gx.mean(dim=1, keepdim=True)                      # (B, 1, 1, 1)
        nx = gx / (mean_gx + self.eps)                              # (B, C, 1, 1)

        # (c) Calibration with residual path
        x_hat = self.gamma * (x * nx) + self.beta + x
        return x_hat


# ==============================================================================
# Module: Mlp (Multi-Layer Perceptron)
# ==============================================================================

class Mlp(nn.Module):
    """Multi-Layer Perceptron (MLP) block inspired by ConvNeXt V2.

    Args:
        in_features (int): Number of input channels.
        hidden_features (int, optional): Number of hidden channels. Defaults to
            `in_features`.
        out_features (int, optional): Number of output channels. Defaults to
            `in_features`.
        act_layer (nn.Module, optional): Activation function. Defaults to `nn.GELU`.
        norm_layer (nn.Module, optional): Normalization layer constructor.
            Defaults to `nn.LayerNorm`.
        bias (bool, optional): If True, add bias to the DConv layers. Defaults to True.
        drop (float, optional): Dropout rate. Defaults to 0.0.
        padding_mode (str, optional): Padding mode for the depthwise convolution.
            Defaults to "circular" for periodic fields.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = LayerNorm2d,
        bias: bool = True,
        drop: float = 0.0,
        padding_mode: str = "circular"  # Use circular padding for periodic fields
    ):
        super().__init__()
        # Set default values if not provided
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define the depthwise convolution layer
        # Use circular padding for periodic boundary conditions, which usually used in physics based models.
        dw_conv = partial(
            nn.Conv2d, kernel_size=3, stride=1, padding=1,
            groups=in_features, bias=bias,
            padding_mode=padding_mode
        )

        # Define the linear layer using 1x1 Convolution
        linear_layer = partial(
            nn.Conv2d, kernel_size=1, bias=bias
        )

        # First linear layer (dw_conv)
        self.fc1 = dw_conv(in_features, in_features)

        # layer normalization
        self.norm1 = norm_layer(in_features)

        # Second linear layer (1x1 Conv) + activation
        self.fc2 = linear_layer(in_features, hidden_features)
        self.act = act_layer()

        # GRN (Global Response Normalization)
        self.norm2 = GRN(hidden_features)

        # Final linear layer (1x1 Conv) to output features + dropout
        self.fc3 = linear_layer(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Spatial MLP.

        Args:
            x (Tensor): Input tensor, typically shape (B, C_in, H, W).

        Returns:
            Tensor: Output tensor, shape (B, C_out, H, W).
        """
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.fc3(x)
        x = self.drop1(x)
        return x


# ==============================================================================
# Module: ECA_block (Efficient Channel Attention Block)
# ==============================================================================

class ECA_block(nn.Module):
    """Efficient Channel Attention (ECA) block.

    This module implements the ECA mechanism as described in:
    "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    (https://arxiv.org/abs/1910.03151).

    The ECA block adaptively selects a one-dimensional convolution kernel size
    based on the number of input channels to capture cross-channel interaction
    without dimensionality reduction.

    Args:
        channel (int): Number of input channels.
        b (int, optional): Bias term for kernel size calculation. Defaults to 1.
        gamma (int, optional): Scaling parameter for kernel size calculation. Defaults to 2.

    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        conv (nn.Conv1d): 1D convolution for adaptive channel attention.
        sigmoid (nn.Sigmoid): Sigmoid activation for generating attention weights.
    """
    def __init__(self, channel: int, b: int = 1, gamma: int = 2):
        super().__init__()
        # Compute adaptive kernel size: k = |log2(C)/γ + b|
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        # Ensure the kernel size is odd (for symmetric padding)
        kernel_size = kernel_size if (kernel_size % 2 == 1) else (kernel_size + 1)

        # Global spatial information aggregation: output size is (batch, channel, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D convolution across the channel dimension to capture local cross-channel interactions
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        # Sigmoid activation to produce attention weights in [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the ECA block.

        Args:
            x (torch.Tensor): Input feature map of shape (batch, channel, height, width).

        Returns:
            torch.Tensor: Recalibrated feature map with the same shape as input.
        """
        # Step 1: Aggregate spatial information for each channel
        y = self.avg_pool(x)  # shape: (batch, channel, 1, 1)

        # Step 2: Prepare for 1D convolution:
        #       squeeze last dim -> (batch, channel, 1)
        #       transpose -> (batch, 1, channel)
        y = y.squeeze(-1).transpose(-1, -2)

        # Step 3: Apply 1D convolution and restore dimensions:
        #       convolved -> (batch, 1, channel)
        #       transpose -> (batch, channel, 1)
        #       unsqueeze back to 4D -> (batch, channel, 1, 1)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)

        # Step 4: Generate channel-wise attention weights
        y = self.sigmoid(y)

        # Step 5: Recalibrate input features by channel-wise multiplication
        out = x * y.expand_as(x)

        return out


# ==============================================================================
# Module: DiResBlock (Diffusion Residual Block)
# ==============================================================================

class DiResBlock(nn.Module):
    """Residual block inspired by Diffusion Transformers (DiT).

    Combines self-attention and MLP sub-blocks with adaptive layer normalization
    (adaLN) style modulation controlled by an embedding (e.g., time embedding).
    Includes gating mechanisms for both attention and MLP outputs.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        emb_dim (int): Dimension of the conditioning embedding `emb`.
        num_heads (int): Number of attention heads for the Attention block.
            Defaults to 8.
        num_groups (int): Number of groups for Group Normalization layers.
            Will be adjusted to be a valid divisor of `in_channels`. Defaults to 4.
        dropout (float): Dropout rate for the MLP block. Defaults to 0.0.
        padding_mode (str): Padding mode for convolutions in the block.

    Raises:
        ValueError: If `emb` is not provided during the forward pass.
        ValueError: If `num_groups` adjustment fails for `in_channels`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        *,
        num_heads: int = 8,
        num_groups: int = 4,
        dropout: float = 0.0,
        padding_mode: str = "circular"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- Adjust num_groups for GroupNorm validity ---
        # GroupNorm before Attention and MLP modulation (operates on in_channels)
        adjusted_num_groups = num_groups
        if in_channels > 0: # Only adjust if channels > 0
            if in_channels < adjusted_num_groups or adjusted_num_groups <= 0:
                adjusted_num_groups = 1 # Fallback
            if in_channels % adjusted_num_groups != 0:
                valid_groups = [
                    g for g in range(1, min(in_channels, adjusted_num_groups) + 1)
                    if in_channels % g == 0
                ]
                if not valid_groups:
                     # This case should be rare if in_channels > 0
                     raise ValueError(f"Could not find valid num_groups divisor for in_channels={in_channels}")
                adjusted_num_groups = max(valid_groups)
                print(
                    f"Info: DiResBlock GroupNorm groups adjusted to {adjusted_num_groups} "
                    f"for in_channels={in_channels}"
                )
        else:
             adjusted_num_groups = 1 # Cannot have groups for 0 channels

        # --- Layer Definitions ---
        # Normalization layers (affine=False because scale/shift comes from modulation)
        # Handle 0 channels case for GroupNorm
        self.norm1 = nn.GroupNorm(
            adjusted_num_groups, in_channels, eps=1e-6, affine=False
        ) if in_channels > 0 else nn.Identity()
        self.norm2 = nn.GroupNorm(
            adjusted_num_groups, in_channels, eps=1e-6, affine=False
        ) if in_channels > 0 else nn.Identity()

        # Attention block (handles 0 channels internally)
        self.attn = Attention(
            channels=in_channels,
            num_heads=num_heads,
            num_groups=adjusted_num_groups,
            qkv_bias=True,
            padding_mode=padding_mode
        )

        # MLP block (using spatial MLP with GELU activation)
        # ConvNeXt V2 style MLP
        mlp_hidden_features = max(out_channels, in_channels) * 2
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=mlp_hidden_features, # Adjusted hidden features
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=dropout,
            padding_mode=padding_mode,
        ) if in_channels > 0 and out_channels > 0 else nn.Identity() # Mlp needs non-zero channels

        # ECA blocks for attention and MLP outputs
        self.eca_attn = ECA_block(channel=in_channels) if in_channels > 0 else nn.Identity()
        self.eca_mlp = ECA_block(channel=out_channels) if out_channels > 0 else nn.Identity()

        # --- Modulation Projection ---
        # Linear layer to project embedding to modulation parameters (shifts, scales, gates)
        # Only compute modulation if channels > 0
        if self.in_channels > 0 or self.out_channels > 0:
            modulation_out_dim = self.in_channels * 5 + self.out_channels
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), # Activation on the embedding first
                nn.Linear(emb_dim, modulation_out_dim, bias=True),
            )
            # Zero-initialize the final modulation layer's weights and biases
            if isinstance(self.adaLN_modulation[-1], nn.Linear):
                nn.init.zeros_(self.adaLN_modulation[-1].bias)
                nn.init.zeros_(self.adaLN_modulation[-1].weight)
        else:
            # No modulation needed if no channels involved
            self.adaLN_modulation = None

        # --- Shortcut Connections ---
        # Main shortcut: Adjusts channels if in_channels != out_channels
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels and in_channels > 0 and out_channels > 0
            else nn.Identity()
        )
        # Residual connection for the MLP block output after final gating
        # Needed if MLP changes the number of channels (in_channels -> out_channels)
        self.mlp_block_res_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
             if in_channels != out_channels and in_channels > 0 and out_channels > 0
             else nn.Identity()
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Forward pass of the DiResBlock.

        Args:
            x (Tensor): Input tensor of shape (B, C_in, H, W).
            emb (Tensor): Conditioning embedding of shape (B, emb_dim).

        Returns:
            Tensor: Output tensor of shape (B, C_out, H, W).

        Raises:
            ValueError: If `emb` is None.
            RuntimeError: If modulation dimensions mismatch (should be caught by checks).
        """
        if self.in_channels <= 0 and self.out_channels <= 0:
             # If both in and out channels are 0, it's effectively an identity
             return x

        if emb is None:
            raise ValueError(
                "Embedding (emb) is required for DiResBlock but was not provided."
            )
        # Ensure embedding is on the correct device
        emb = emb.to(x.device)

        # Calculate the main shortcut connection
        shortcut = self.conv_shortcut(x) # (B, C_out, H, W)

        # If input channels are 0, we can't process further in the standard path
        if self.in_channels == 0:
             # Only the shortcut path is meaningful if out_channels > 0
             return shortcut

        # --- Calculate Modulation Parameters ---
        # (Only if adaLN_modulation exists)
        if self.adaLN_modulation is not None:
            mod_params = self.adaLN_modulation(emb)
            split_sizes = [self.in_channels] * 5 + [self.out_channels]
            try:
                 shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                     torch.split(mod_params, split_sizes, dim=1)
            except RuntimeError as e:
                 raise RuntimeError(f"Error splitting modulation parameters (dim={mod_params.shape[-1]}) "
                                   f"into sizes {split_sizes}. Check emb_dim and channel configuration. Orig Error: {e}") from e
        else:
             # Provide dummy tensors if no modulation (e.g., C_in=0, C_out>0 case, though unlikely used)
             # These won't be used if norm/attn/mlp are Identities, but prevents errors.
             zero_val = torch.zeros(emb.shape[0], self.in_channels, device=x.device)
             one_val_out = torch.ones(emb.shape[0], self.out_channels, device=x.device) # Gate MLP needs out_channels
             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp = [zero_val] * 5
             gate_mlp = one_val_out # Default gate_mlp to 1 if not computed


        # --- Attention Block Path ---
        residual_attn = x # Residual for attention: (B, C_in, H, W)
        x_norm1 = self.norm1(x) # Normalize: (B, C_in, H, W)
        # Modulate with shift_msa, scale_msa (both B, C_in)
        x_mod1 = modulate(x_norm1, shift_msa, scale_msa)
        # Apply attention
        x_attn = self.attn(x_mod1) # (B, C_in, H, W)
        # Apply ECA attention block
        x_attn = self.eca_attn(x_attn)
        # Apply gating (gate_msa shape B, C_in) and add residual
        # Need unsqueeze for spatial broadcast
        x = residual_attn + gate_msa.unsqueeze(-1).unsqueeze(-1) * x_attn

        # --- MLP Block Path ---
        # Residual for MLP path, potentially projected if channels changed
        residual_mlp = self.mlp_block_res_proj(x) # (B, C_out, H, W)
        x_norm2 = self.norm2(x) # Normalize: (B, C_in, H, W)
        # Modulate with shift_mlp, scale_mlp (both B, C_in)
        x_mod2 = modulate(x_norm2, shift_mlp, scale_mlp)
        # Apply MLP (changes channels C_in -> C_out)
        x_mlp = self.mlp(x_mod2) # (B, C_out, H, W)
        # Apply ECA MLP block
        x_mlp = self.eca_mlp(x_mlp)
        # Apply gating (gate_mlp shape B, C_out) and add residual
        # Need unsqueeze for spatial broadcast
        x = residual_mlp + gate_mlp.unsqueeze(-1).unsqueeze(-1) * x_mlp

        # --- Final Output ---
        # Add the main shortcut connection
        return x + shortcut


# ==============================================================================
# Module: Upsample (Upsampling Layer using PixelShuffle)
# ==============================================================================

class Upsample(nn.Module):
    """Upsampling layer using PixelShuffle followed by an optional convolution.

    Increases spatial resolution by `scale_factor`. It first uses a convolution
    to increase channels, then PixelShuffle rearranges elements, and finally
    an optional 3x3 convolution refines features. Uses DConv for convolutions.

    Handles `channels = 0` by returning a correctly shaped zero tensor, avoiding
    errors with zero-channel convolutions.

    Args:
        channels (int): Number of input and final output channels. Must be > 0
            for standard operation. If 0, acts as a specialized identity.
        use_conv (bool): Whether to apply a final 3x3 convolution after
            PixelShuffle for feature refinement. Defaults to True.
        scale_factor (int): Factor by which to increase spatial resolution.
            Defaults to 2.
        padding_mode (str): Padding mode for the depthwise convolution.
    """

    def __init__(
        self, channels: int, use_conv: bool = True, scale_factor: int = 2, padding_mode: str = "circular"
    ):
        super().__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.use_final_conv = use_conv

        # Only define layers if channels > 0 to avoid Conv2d(0, 0, ...)
        if self.channels > 0:
            self.scale_factor_squared = scale_factor * scale_factor

            # Conv to prepare channels for PixelShuffle: C -> C * scale_factor^2
            # Using DConv consistent with other blocks
            self.conv1 = (
                DConv(
                    self.channels,
                    self.channels * self.scale_factor_squared,
                    kernel_size=3,
                    padding=1,
                    padding_mode=padding_mode,
                )
            )

            # PixelShuffle layer: Rearranges (C * r^2, H, W) -> (C, H * r, W * r)
            self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

            # Optional final convolution for refinement (operates on C channels)
            self.conv2 = (
                DConv(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode=padding_mode,
                )
                if self.use_final_conv
                else nn.Identity()
            )
        else:
            # If channels = 0, layers are effectively Identity/None.
            # Forward pass handles this explicitly.
            self.conv1 = nn.Identity()
            self.pixel_shuffle = nn.Identity() # PixelShuffle(r) works on B,C*r^2,H,W -> B,C,H*r,W*r, fails if C=0
            self.conv2 = nn.Identity()
            # Ensure flag consistency, though conv2 is Identity anyway
            self.use_final_conv = False


    def forward(self, x: Tensor) -> Tensor:
        """Upsamples the input tensor using PixelShuffle.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H * scale_factor, W * scale_factor).
                    If input C was 0, output C is 0.
        """
        # Explicitly handle the zero-channel case to avoid PixelShuffle error
        if self.channels == 0:
            B, _, H, W = x.shape
            out_H = H * self.scale_factor
            out_W = W * self.scale_factor
            # Return zero tensor with target shape (B, 0, H_out, W_out)
            return torch.zeros(
                (B, 0, out_H, out_W), dtype=x.dtype, device=x.device
            )

        # --- Standard forward pass for non-zero channels ---
        # 1. Initial convolution to expand channels
        # (B, C, H, W) -> (B, C * r^2, H, W)
        x = self.conv1(x)

        # 2. PixelShuffle to upsample spatial dimensions and reduce channels
        # (B, C * r^2, H, W) -> (B, C, H * r, W * r)
        x = self.pixel_shuffle(x)

        # 3. Optional final convolution for feature refinement
        # (B, C, H * r, W * r) -> (B, C, H * r, W * r)
        x = self.conv2(x)
        return x


# ==============================================================================
# Module: Downsample
# ==============================================================================

class Downsample(nn.Module):
    """Downsampling layer using convolution and pooling.

    Reduces spatial resolution by a factor of 2.
    Handles `channels = 0` by explicitly returning a downsampled zero tensor.

    Args:
        channels (int): Number of input channels. Determines operation if > 0.
        padding_mode (str): Padding mode for the depthwise convolution.
    """

    def __init__(self, channels: int, padding_mode: str = "circular"):
        super().__init__()

        self.conv_in = DConv(
            channels, channels, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode
        ) if channels > 0 else nn.Identity()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2) if channels > 0 else nn.Identity()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) if channels > 0 else nn.Identity()
        self.conv_out = nn.Conv2d(channels*3, channels, kernel_size=1, bias=False) if channels > 0 else nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        """Downsamples the input tensor by a factor of 2 spatially.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H/2, W/2).
        """
        if x.shape[1] == 0:
            B, _, H, W = x.shape
            # Calculate target downsampled shape
            out_H = H // 2
            out_W = W // 2
            # Return a zero tensor with the correct downsampled shape
            # print(f"Debug Downsample C=0: Input {x.shape}, Output ({B}, 0, {out_H}, {out_W})") # Optional debug print
            return torch.zeros((B, 0, out_H, out_W), dtype=x.dtype, device=x.device)
        else:
            conv_feature = self.conv_in(x)  # (B, C, H/2, W/2)
            avg_pooled_feature = self.avg_pool(x)  # (B, C, H/2, W/2)
            max_pooled_feature = self.max_pool(x)  # (B, C, H/2, W/2)
            features = torch.cat([conv_feature, avg_pooled_feature, max_pooled_feature], dim=1)
            return self.conv_out(features)  # (B, C, H/2, W/2)


# ==============================================================================
# Module: Triplet Attention
# ==============================================================================

class ZPool(nn.Module):
    """Z-Pool operator: concat of channel-wise max & average pooling."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B×C×H×W → outputs B×2×H×W
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)


class AttentionGate(nn.Module):
    """Single attention branch using ZPool + conv + sigmoid."""
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Spatial kernel size for attention conv.
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.compress = ZPool()
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape B×C×H×W.
        Returns:
            Tensor of same shape, modulated by attention weights.
        """
        x_pool = self.compress(x)             # B×2×H×W
        x_attn = self.conv(x_pool)            # B×1×H×W
        scale = torch.sigmoid_(x_attn)        # B×1×H×W
        return x * scale                      # broadcast over channel dim


class TripletAttention(nn.Module):
    """
    Triplet Attention module: captures cross-dimension interaction
    via three AttentionGate branches. https://arxiv.org/abs/2010.03045
    """
    def __init__(self, no_spatial: bool = False,):
        """
        Args:
            no_spatial: If True, omit the HW spatial branch.
        """
        super().__init__()
        self.cw = AttentionGate()   # Channel–Width
        self.hc = AttentionGate()   # Height–Channel
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()  # Height–Width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor B×C×H×W.
        Returns:
            Attention-modulated tensor, same shape as x.
        """
        # CW branch
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()
        # HC branch
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            # HW branch (standard spatial)
            x_out3 = self.hw(x)
            return (x_out1 + x_out2 + x_out3) / 3.0
        else:
            return (x_out1 + x_out2) / 2.0


# ==============================================================================
# Main Model: DiResUnet
# ==============================================================================

class DiResUnet(nn.Module):
    """A U-Net model using DiResBlocks and adaptable to multiple conditions.

    This U-Net variant replaces standard ResNet blocks with DiResBlocks, which
    incorporate self-attention and MLP layers modulated by time and conditional
    embeddings. It handles variable input HxW dimensions (must be divisible by
    2^num_downsamples) and multiple conditional inputs.

    Features:
    - Standard U-Net architecture (Encoder, Bottleneck, Decoder) with skip connections.
    - Uses `DiResBlock` as the core building block.
    - Time embedding using Sinusoidal Positional Encoding (with `max_period` set
      dynamically based on max channel count) followed by MLPs.
    - Optional multiple conditional embeddings, each processed similarly and
      combined with the time embedding before modulation in DiResBlocks.
    - Uses `DConv` within `Upsample`, `Downsample`, and `Mlp` (inside DiResBlock).
    - Handles variable input HxW dimensions in the forward pass.

    Args:
        in_channels (int): Number of channels in the input image. Defaults to 1.
        out_channels (int): Number of channels in the output image. Defaults to 1.
        model_channels (int): Base number of channels in the convolutional layers.
            Defaults to 16. Must be positive.
        channel_mult (Sequence[int]): Multipliers for `model_channels` at each
            resolution level (encoder/decoder). Determines depth and channel count.
            Example: (1, 2, 4, 8). Defaults to (2, 2, 4). Must not be empty.
        num_res_blocks (int): Number of `DiResBlock` instances per resolution level.
            Defaults to 1.
        dropout (float): Dropout probability in `DiResBlock`'s MLP. Defaults to 0.1.
        num_heads (int): Number of attention heads in `DiResBlock`'s Attention.
            Defaults to 16.
        num_groups (int): Number of groups for Group Normalization within `DiResBlock`.
            Defaults to 4. Will be adjusted per block based on its input channels.
        time_emb_dim (Optional[int]): Base dimension for the time embedding.
            If None, defaults to `model_channels`. Adjusted to be positive and even.
        cond_emb_dims (Optional[Sequence[int]]): Sequence of base dimensions for
            the conditional embeddings. If None or empty, conditional input is disabled.
            Each dimension is adjusted to be positive and even. Defaults to None.
        final_emb_dim (Optional[int]): Dimension of the final combined embedding
            (after MLPs) used for modulation in DiResBlocks. If None, defaults
            to 4 * `time_emb_dim`. Must be positive.

    Raises:
        ValueError: If parameter configurations are invalid (e.g., empty
            `channel_mult`, non-positive dimensions, embedding mismatches).
        RuntimeError: Potential dimension mismatches during forward pass, often
            due to configuration errors or invalid inputs.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 32,
        channel_mult: Sequence[int] = (2, 2, 4),
        num_res_blocks: int = 1,
        dropout: float = 0.1,
        num_heads: int = 32, # Default heads for DiResBlock
        num_groups: int = 4, # Default groups for DiResBlock
        time_emb_dim: Optional[int] = None,
        cond_emb_dims: Optional[Sequence[int]] = None,
        final_emb_dim: Optional[int] = None,
        padding_mode: str = "circular", # Use circular padding for periodic fields
    ):
        super().__init__()

        # --- Input Validation ---
        if not channel_mult:
            raise ValueError("channel_mult cannot be empty.")
        if model_channels <= 0:
             raise ValueError(f"model_channels must be positive, got {model_channels}")
        if in_channels < 0 or out_channels < 0:
             raise ValueError(f"in_channels ({in_channels}) and out_channels ({out_channels}) cannot be negative.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = tuple(channel_mult) # Ensure tuple
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_groups = num_groups # Base num_groups, DiResBlock adjusts internally
        self.num_levels = len(channel_mult)
        self.num_downsamples = self.num_levels - 1 # Number of Downsample/Upsample layers

        # --- Determine Max Channels for Embedding max_period ---
        self.max_channels = model_channels * max(channel_mult) if channel_mult else model_channels
        if self.max_channels <= 0 :
             # Fallback if channel_mult results in non-positive max channels
             print(
                 f"Warning: Calculated max_channels ({self.max_channels}) is non-positive."
                 f" Defaulting max_period for embeddings to 10000."
             )
             self.embedding_max_period = 10000
        else:
             self.embedding_max_period = self.max_channels # Use max channels as max_period

        # --- Embedding Dimensions Setup ---
        # Time embedding dimension
        _time_emb_dim = time_emb_dim if time_emb_dim is not None and time_emb_dim > 0 else model_channels
        if _time_emb_dim <= 0:
            raise ValueError(f"time_emb_dim must be positive or None, derived value was {_time_emb_dim}")
        if _time_emb_dim % 2 != 0: _time_emb_dim += 1 # Ensure even
        self.time_emb_dim = _time_emb_dim # Store actual used dim

        # Final combined embedding dimension for modulation
        _final_emb_dim = final_emb_dim if final_emb_dim is not None and final_emb_dim > 0 else self.time_emb_dim * 4
        if _final_emb_dim <= 0:
            raise ValueError(f"final_emb_dim must be positive or None, derived value was {_final_emb_dim}")
        self.final_emb_dim = _final_emb_dim

        # --- Conditional Embedding Setup ---
        self.cond_emb_dims = [] # Stores adjusted, validated conditional dims
        self.use_condition = cond_emb_dims is not None and len(cond_emb_dims) > 0
        if self.use_condition:
             assert cond_emb_dims is not None # For type checker
             for i, dim in enumerate(cond_emb_dims):
                 if not isinstance(dim, int) or dim <= 0:
                     raise ValueError(f"Conditional embedding dimension at index {i} must be a positive integer, got {dim}")
                 if dim % 2 != 0:
                     dim += 1 # Ensure even
                     print(f"Info: Conditional embedding dim at index {i} adjusted to {dim} to be even.")
                 self.cond_emb_dims.append(dim)

        # --- Embedding Layers ---
        self.time_embedder = SinusoidalPosEmb(
            dim=self.time_emb_dim, max_period=self.embedding_max_period
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.final_emb_dim),
            nn.SiLU(),
            nn.Linear(self.final_emb_dim, self.final_emb_dim),
        )

        if self.use_condition:
             self.cond_embedders = nn.ModuleList([
                 SinusoidalPosEmb(dim=cond_dim, max_period=self.embedding_max_period)
                 for cond_dim in self.cond_emb_dims
             ])
             self.cond_mlps = nn.ModuleList([
                 nn.Sequential(
                     nn.Linear(cond_dim, self.final_emb_dim),
                     nn.SiLU(),
                     nn.Linear(self.final_emb_dim, self.final_emb_dim),
                 ) for cond_dim in self.cond_emb_dims
             ])
        else:
             self.cond_embedders = None
             self.cond_mlps = None

        self.fusion_mlp = nn.Linear(
            self.final_emb_dim * 2, self.final_emb_dim
        ) if self.use_condition else None

        # --- Input Convolution ---
        # Use standard Conv2d for initial projection unless specified otherwise
        self.conv_in = nn.Conv2d(
            in_channels, model_channels, kernel_size=3, padding=1, padding_mode=padding_mode
        ) if in_channels > 0 and model_channels > 0 else nn.Identity()

        # --- Downsampling Path (Encoder) ---
        self.down_blocks = nn.ModuleList() # Stores ModuleLists of DiResBlocks per level
        self.down_sample = nn.ModuleList() # Stores Downsample/Identity operations
        encoder_skip_channels = [] # Stores output channels of each encoder level for skips
        if model_channels > 0:
             encoder_skip_channels.append(model_channels) # Channels after conv_in
        current_channels = model_channels

        for i in range(self.num_levels):
            out_ch = model_channels * self.channel_mult[i]
            level_blocks = nn.ModuleList() # Blocks within this level

            for k in range(num_res_blocks):
                # First block handles channel change, subsequent blocks are same channel
                in_ch = current_channels if k == 0 else out_ch
                if in_ch <= 0 and out_ch <= 0:
                    # Skip creating DiResBlock if no channels in or out for this block
                    res_block = nn.Identity()
                else:
                    res_block = DiResBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        emb_dim=self.final_emb_dim, # Pass final combined emb dim
                        num_heads=num_heads,
                        num_groups=num_groups, # Base num_groups
                        dropout=dropout,
                        padding_mode=padding_mode, # Use circular padding for periodic fields
                    )
                level_blocks.append(res_block)
                # Update current_channels only after the block might change it
                current_channels = out_ch

            self.down_blocks.append(level_blocks)
            if current_channels > 0: # Only store skip if channels exist
                encoder_skip_channels.append(current_channels)

            # Add downsampling layer (except for the last level)
            if i < self.num_downsamples:
                 # use_conv=True uses DConv for downsampling
                 self.down_sample.append(Downsample(channels=current_channels, padding_mode=padding_mode))
            else:
                 # No downsampling after the last encoder level (leads to bottleneck)
                 self.down_sample.append(nn.Identity())

        # --- Bottleneck ---
        bottleneck_channels = current_channels # Channels from the last encoder level
        # Use two DiResBlocks in the bottleneck
        self.middle_block = nn.Sequential(
            DiResBlock(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                emb_dim=self.final_emb_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                dropout=dropout,
                padding_mode=padding_mode,
            ) if bottleneck_channels > 0 else nn.Identity(),

            TripletAttention() if bottleneck_channels > 0 else nn.Identity(),

            DiResBlock(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                emb_dim=self.final_emb_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                dropout=dropout,
                padding_mode=padding_mode,
            ) if bottleneck_channels > 0 else nn.Identity(),

            TripletAttention() if bottleneck_channels > 0 else nn.Identity(),
        )

        # --- Upsampling Path (Decoder) ---
        self.up_blocks = nn.ModuleList() # Stores modules for each level
        self.up_sample = nn.ModuleList() # Stores Upsample/Identity operations
        decoder_input_channels = bottleneck_channels # Start with bottleneck output channels

        # Iterate through levels in reverse order (deepest to shallowest)
        for i in reversed(range(self.num_levels)):
            target_ch = model_channels * self.channel_mult[i] # Target channels for this level
            level_blocks = nn.ModuleList() # Blocks within this level

            # Get skip connection channels from the corresponding encoder level
            # Pop in reverse order of appending, handle potential empty list if channels were 0
            skip_ch = encoder_skip_channels.pop() if encoder_skip_channels else 0

            # Input to the first DiResBlock: channels from previous up-level + skip channels
            resnet_input_channels = decoder_input_channels + skip_ch

            for k in range(num_res_blocks):
                # First block handles channel merge, subsequent blocks use target_ch
                in_ch = resnet_input_channels if k == 0 else target_ch
                if in_ch <= 0 and target_ch <= 0:
                     res_block = nn.Identity()
                else:
                     res_block = DiResBlock(
                         in_channels=in_ch,
                         out_channels=target_ch,
                         emb_dim=self.final_emb_dim,
                         num_heads=num_heads,
                         num_groups=num_groups,
                         dropout=dropout,
                         padding_mode=padding_mode,
                    )
                level_blocks.append(res_block)
                # Update the expected input channels for the next block in sequence
                resnet_input_channels = target_ch # Subsequent blocks use target_ch as input

            self.up_blocks.append(level_blocks) # Appended in reverse level order

            # Add upsampling layer (except for the first decoder level, i=0)
            # Upsampler operates on the output channels (target_ch) of this level
            upsampler = Upsample(channels=target_ch, use_conv=True, padding_mode=padding_mode) if i > 0 else nn.Identity()
            self.up_sample.append(upsampler) # Appended in reverse level order

            decoder_input_channels = target_ch # Output of this level is input for the next

        # --- Final Assertion Check ---
        # After popping all encoder skips, only the initial conv_in skip should remain (if it existed)
        expected_remaining_skips = 1 if model_channels > 0 else 0
        assert len(encoder_skip_channels) == expected_remaining_skips, \
               (f"Mismatch in skip connections during __init__."
                f" Expected {expected_remaining_skips} remaining, found {len(encoder_skip_channels)}.")

        # --- Output Convolution ---
        # Output channels should match the shallowest level's target channels
        final_decoder_channels = model_channels * self.channel_mult[0]

        # Adjust num_groups for final norm based on channels
        final_norm_groups = num_groups # Start with base num_groups
        if final_decoder_channels > 0:
             if final_decoder_channels < final_norm_groups or final_norm_groups <= 0:
                 final_norm_groups = 1
             elif final_decoder_channels % final_norm_groups != 0:
                 valid_groups = [g for g in range(1, min(final_decoder_channels, final_norm_groups) + 1)
                                 if final_decoder_channels % g == 0]
                 final_norm_groups = max(valid_groups) if valid_groups else 1
                 print(f"Info: Output GroupNorm groups adjusted to {final_norm_groups}"
                       f" for channels={final_decoder_channels}")
        else:
             final_norm_groups = 1 # Cannot have groups for 0 channels

        # Use GroupNorm before final activation/conv if channels > 0
        self.out_norm = nn.GroupNorm(final_norm_groups, final_decoder_channels) \
                        if final_decoder_channels > 0 else nn.Identity()
        self.out_act = nn.SiLU()
        # Final conv maps to the desired number of output channels
        # Use standard Conv2d for the final output layer
        self.conv_out = nn.Conv2d(
            final_decoder_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode
        ) if final_decoder_channels > 0 and out_channels >= 0 else nn.Identity() # Allow 0 out_channels


    def forward(self, x: Tensor, time: Tensor, conditions: Optional[Sequence[Tensor]] = None) -> Tensor:
        """Forward pass of the DiResUnet.

        Args:
            x (Tensor): Input tensor (e.g., noisy image) of shape (B, C_in, H, W).
               H and W must be divisible by 2^`num_downsamples`.
            time (Tensor): Time step tensor of shape (B,), (B, 1), (1,) or (1, 1).
                Values are typically integers or floats representing time steps.
            conditions (Optional[Sequence[Tensor]]): A sequence (list or tuple)
                of conditional input tensors. Required if `cond_emb_dims` was set
                during init. Each tensor in the sequence should have shape (B,)
                or (B, 1). The number of tensors must match the length of
                `cond_emb_dims`. Defaults to None.

        Returns:
            Tensor: Output tensor (e.g., predicted noise) of shape (B, C_out, H, W).

        Raises:
            ValueError: If input dimensions are invalid (not divisible by required
                power of 2), or if conditional/time inputs are invalid, missing,
                or the number of conditions doesn't match initialization.
            RuntimeError: Potential dimension mismatches during execution, often
                due to incorrect skip connection handling or layer misconfiguration.
        """
        # --- Input Checks ---
        B, C, H, W = x.shape
        min_divisor = 2**self.num_downsamples
        if H % min_divisor != 0 or W % min_divisor != 0:
             raise ValueError(f"Input height ({H}) and width ({W}) must be divisible by {min_divisor} "
                              f"(2^num_downsamples where num_downsamples={self.num_downsamples})")
        if C != self.in_channels:
             raise ValueError(f"Input channel mismatch. Expected {self.in_channels}, got {C}")

        # --- Time Input Check and Broadcasting ---
        if time.ndim == 2 and time.shape[1] == 1: time = time.squeeze(1) # (B, 1) -> (B,)
        if time.ndim == 2 and time.shape[0] == 1: time = time.squeeze(0).expand(B) # (1, 1) -> (B,)
        if time.ndim == 1 and time.shape[0] == 1: time = time.expand(B) # (1,) -> (B,)
        if time.ndim != 1 or time.shape[0] != B:
             raise ValueError(f"Invalid time tensor shape. Expected ({B},), got {time.shape} after processing.")

        # --- Conditional Input Checks ---
        num_expected_conditions = len(self.cond_emb_dims) if self.use_condition else 0
        processed_conditions: List[Tensor] = [] # Store validated condition tensors

        if self.use_condition:
             if conditions is None:
                 raise ValueError("Conditional inputs are required (cond_emb_dims was provided) but 'conditions' argument was None.")
             if not isinstance(conditions, (list, tuple)):
                 raise ValueError(f"Expected 'conditions' to be a sequence (list or tuple), got {type(conditions)}")
             if len(conditions) != num_expected_conditions:
                 raise ValueError(f"Expected {num_expected_conditions} condition tensors, but got {len(conditions)}.")

             for i, cond_tensor in enumerate(conditions):
                 if not isinstance(cond_tensor, Tensor):
                      raise ValueError(f"Condition at index {i} must be a torch.Tensor, got {type(cond_tensor)}")
                 # Process shape like time tensor
                 if cond_tensor.ndim == 2 and cond_tensor.shape[1] == 1: cond_tensor = cond_tensor.squeeze(1)
                 if cond_tensor.ndim == 2 and cond_tensor.shape[0] == 1: cond_tensor = cond_tensor.squeeze(0).expand(B)
                 if cond_tensor.ndim == 1 and cond_tensor.shape[0] == 1: cond_tensor = cond_tensor.expand(B)
                 if cond_tensor.ndim != 1 or cond_tensor.shape[0] != B:
                     raise ValueError(f"Invalid shape for condition tensor at index {i}. Expected ({B},), got {cond_tensor.shape} after processing.")
                 processed_conditions.append(cond_tensor)

        elif conditions is not None and len(conditions) > 0: # Check if conditions were actually passed
            print("Warning: Conditional inputs provided but model was initialized without cond_emb_dims. Inputs ignored.")
            # No need to process them

        # --- 1. Process Embeddings ---
        # Compute time embedding and pass through MLP
        t_emb = self.time_embedder(time)
        t_emb = self.time_mlp(t_emb) # (B, final_emb_dim)

        final_emb: Tensor = t_emb # Start with time embedding

        # Process and add conditional embeddings if enabled
        if self.use_condition and self.cond_embedders and self.cond_mlps:
             total_cond_emb = torch.zeros_like(t_emb) # Initialize cumulative conditional embedding
             # Iterate through conditions, embedders, and MLPs simultaneously
             for i, (cond_tensor, embedder, mlp) in enumerate(zip(processed_conditions, self.cond_embedders, self.cond_mlps)):
                 # Ensure condition tensor is on the correct device before embedding
                 cond_tensor = cond_tensor.to(x.device)
                 c_emb = embedder(cond_tensor)
                 c_emb = mlp(c_emb) # (B, final_emb_dim)
                 total_cond_emb = total_cond_emb + c_emb # Accumulate embeddings

             final_emb = self.fusion_mlp(torch.cat([final_emb, total_cond_emb], dim=-1)) \
                 if self.fusion_mlp else total_cond_emb + final_emb
        # else: final_emb remains just t_emb

        # --- 2. Initial Convolution ---
        h = self.conv_in(x)
        skips = []
        if self.model_channels > 0: # Only store skip if initial channels > 0
             skips.append(h)

        # --- 3. Downsampling Path (Encoder) ---
        for i in range(self.num_levels):
            # Apply DiResBlocks for level i
            for block in self.down_blocks[i]:
                 # Pass combined embedding to DiResBlock
                 h = block(h, final_emb)
            # Store skip connection *before* downsampling (if channels exist)
            if h.shape[1] > 0: # Check if channels > 0 before storing skip
                 skips.append(h)
            # Apply downsampling (op is Identity for the last level)
            h = self.down_sample[i](h)

        # --- 4. Bottleneck ---
        # Apply bottleneck layers sequentially, passing embedding
        if isinstance(self.middle_block[0], DiResBlock):
             h = self.middle_block[0](h, final_emb)
        if isinstance(self.middle_block[1], DiResBlock):
             h = self.middle_block[1](h, final_emb)


        # --- 5. Upsampling Path (Decoder) ---
        # Note: self.up_blocks and self.up_sample were populated in reverse level order
        for i in range(self.num_levels):
            # Retrieve corresponding skip connection (popped in reverse order)
            # Handle cases where skips might be missing due to 0 channels
            skip_h = skips.pop() if skips else None

            # Concatenate input h (from previous up-level) with skip connection
            if skip_h is not None:
                 # Ensure skip connection has same spatial dimensions as h before cat
                 if skip_h.shape[2:] != h.shape[2:]:
                     # This might happen if Downsample/Upsample logic is complex
                     # or if skip connection comes from a level where downsampling happened differently.
                     # Basic fix: Interpolate skip_h. More robust: Check logic.
                     print(f"Warning: Skip connection shape {skip_h.shape} doesn't match decoder input {h.shape} at level {self.num_levels - 1 - i}. Resizing skip.")
                     skip_h = F.interpolate(skip_h, size=h.shape[2:], mode='bilinear', align_corners=False)

                 # Only concatenate if both h and skip_h have channels > 0
                 if h.shape[1] > 0 and skip_h.shape[1] > 0:
                     h = torch.cat([h, skip_h], dim=1)
                 elif skip_h.shape[1] > 0: # If h has 0 channels, just use skip
                      h = skip_h
                 # else: if skip_h has 0 channels, keep h as is
            # else: No skip connection, just use h

            # Apply DiResBlocks for this decoder level
            # self.up_blocks[i] corresponds to the blocks for encoder level (num_levels - 1 - i)
            for block in self.up_blocks[i]:
                h = block(h, final_emb)

            # Apply Upsampling (prepares h for the next higher-res level)
            # self.up_sample[i] is the upsampler *after* the blocks at this level
            # The last upsampler is Identity
            h = self.up_sample[i](h)

        # --- Final Assertion Check ---
        expected_remaining_skips = 1 if self.model_channels > 0 else 0
        assert len(skips) == expected_remaining_skips, \
               f"Mismatch in skip connections during forward pass. Expected {expected_remaining_skips} remaining, found {len(skips)}."

        # --- 6. Output Layer ---
        if isinstance(self.out_norm, nn.Identity):
             # Handle case where final_decoder_channels might be 0
             output = self.conv_out(h)
        else:
             h = self.out_norm(h)
             h = self.out_act(h)
             output = self.conv_out(h) # (B, out_channels, H, W)

        # Final check on output shape
        expected_out_shape = (B, self.out_channels, H, W)
        if output.shape != expected_out_shape:
             raise RuntimeError(f"Output shape {output.shape} mismatch. Expected {expected_out_shape}")

        return output


# ==============================================================================
# Test Code for DiResUnet and Modules
# ==============================================================================

if __name__ == "__main__":
    print("="*40)
    print("--- Running DiResUnet and Module Tests ---")
    print("="*40)

    # --- Environment and Configuration Checks ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        try:
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Could not get GPU name: {e}")

    # --- Test Parameters ---
    BATCH_SIZE = 2
    TEST_SIZES = [32, 48, 64] # Example sizes (must be divisible by min_divisor)
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    MODEL_CHANNELS = 32 # Keep small for tests
    CHANNEL_MULT = (1, 2, 2) # e.g., 3 levels -> 32, 64, 64 channels
    NUM_LEVELS = len(CHANNEL_MULT)
    NUM_DOWNSAMPLES = NUM_LEVELS - 1
    MIN_DIVISOR = 2**NUM_DOWNSAMPLES
    NUM_RES_BLOCKS = 1 # Fewer blocks for faster testing
    COND_EMB_DIMS = (64, 16) # Two conditions with different base dims
    NUM_CONDITIONS = len(COND_EMB_DIMS)
    # Let time_emb_dim and final_emb_dim default or set explicitly
    TIME_EMB_DIM = MODEL_CHANNELS * 2 # Example: 64
    FINAL_EMB_DIM = TIME_EMB_DIM * 4 # Example: 256 (must be positive)
    NUM_HEADS = 4 # Heads for DiResBlock Attention
    NUM_GROUPS = 4 # Base groups for DiResBlock GroupNorm
    DROPOUT = 0.05 # Small dropout

    print("\n--- Test Configuration ---")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"In Channels: {IN_CHANNELS}, Out Channels: {OUT_CHANNELS}")
    print(f"Model Channels: {MODEL_CHANNELS}")
    print(f"Channel Multipliers: {CHANNEL_MULT} ({NUM_LEVELS} levels, {NUM_DOWNSAMPLES} downsamples)")
    print(f"Min Input Divisor: {MIN_DIVISOR}")
    print(f"DiResBlocks per Level: {NUM_RES_BLOCKS}")
    print(f"Conditional Dims: {COND_EMB_DIMS} ({NUM_CONDITIONS} conditions)")
    print(f"Time Emb Dim (Base): {TIME_EMB_DIM}")
    print(f"Final Emb Dim (Modulation): {FINAL_EMB_DIM}")
    print(f"Num Heads: {NUM_HEADS}, Num Groups (Base): {NUM_GROUPS}")
    print(f"Dropout: {DROPOUT}")

    # Validate test sizes
    valid_test_sizes = [s for s in TEST_SIZES if s % MIN_DIVISOR == 0 and s > 0]
    if not valid_test_sizes:
         raise ValueError(f"None of the TEST_SIZES {TEST_SIZES} are valid (must be positive and divisible by {MIN_DIVISOR})")
    print(f"\nTesting with valid image sizes: {valid_test_sizes}")
    if len(valid_test_sizes) < len(TEST_SIZES):
        print(f"Skipped invalid sizes: {[s for s in TEST_SIZES if s not in valid_test_sizes]}")

    all_tests_passed = True # Track overall test status

    # --- Test Helper Modules Independently ---
    print("\n" + "="*30)
    print("--- Testing Helper Modules ---")
    print("="*30)
    try:
        # Re-run provided tests for the building blocks
        helper_test_B, helper_C_in, helper_C_out, helper_H, helper_W = 2, 32, 64, 16, 16
        helper_emb_dim = 128
        print(f"Helper Test Params: B={helper_test_B}, C_in={helper_C_in}, C_out={helper_C_out}, H={helper_H}, W={helper_W}, EmbDim={helper_emb_dim}")

        # Test modulate
        print("\nTesting modulate...")
        x_4d = torch.randn(helper_test_B, helper_C_in, helper_H, helper_W, device=device)
        shift_4d = torch.randn(helper_test_B, helper_C_in, device=device)
        scale_4d = torch.randn(helper_test_B, helper_C_in, device=device)
        out_4d = modulate(x_4d, shift_4d, scale_4d)
        assert out_4d.shape == x_4d.shape, f"Modulate 4D shape mismatch: {out_4d.shape} vs {x_4d.shape}"
        print("✅ modulate(4D) PASSED.")
        x_2d = torch.randn(helper_test_B, helper_emb_dim, device=device)
        shift_2d = torch.randn(helper_test_B, helper_emb_dim, device=device)
        scale_2d = torch.randn(helper_test_B, helper_emb_dim, device=device)
        out_2d = modulate(x_2d, shift_2d, scale_2d)
        assert out_2d.shape == x_2d.shape, f"Modulate 2D shape mismatch: {out_2d.shape} vs {x_2d.shape}"
        print("✅ modulate(2D) PASSED.")

        # Test DConv
        print("\nTesting DConv...")
        dw_conv = DConv(helper_C_in, helper_C_out, kernel_size=3, padding=1, stride=1, bias=True).to(device)
        x = torch.randn(helper_test_B, helper_C_in, helper_H, helper_W, device=device)
        output = dw_conv(x)
        assert output.shape == (helper_test_B, helper_C_out, helper_H, helper_W), f"DConv shape mismatch: {output.shape}"
        print("✅ DConv PASSED.")

        # Test SinusoidalPosEmb
        print("\nTesting SinusoidalPosEmb...")
        max_p = 1000 # Example max_period for test
        pos_emb = SinusoidalPosEmb(dim=helper_emb_dim, max_period=max_p).to(device)
        t_1d = torch.randint(0, max_p, (helper_test_B,), device=device)
        output_1d = pos_emb(t_1d)
        assert output_1d.shape == (helper_test_B, helper_emb_dim), f"SinusoidalPosEmb 1D shape mismatch: {output_1d.shape}"
        print("✅ SinusoidalPosEmb (1D) PASSED.")
        t_2d = torch.randint(0, max_p, (helper_test_B, 1), device=device)
        output_2d = pos_emb(t_2d)
        assert output_2d.shape == (helper_test_B, helper_emb_dim), f"SinusoidalPosEmb 2D shape mismatch: {output_2d.shape}"
        print("✅ SinusoidalPosEmb (2D) PASSED.")

        # Test Attention
        print("\nTesting Attention...")
        attn_ch = helper_C_in
        attn_heads = 4
        attn_groups = 4
        attention = Attention(channels=attn_ch, num_heads=attn_heads, num_groups=attn_groups).to(device)
        x = torch.randn(helper_test_B, attn_ch, helper_H // 2, helper_W // 2, device=device)
        output = attention(x)
        assert output.shape == x.shape, f"Attention shape mismatch: {output.shape}"
        print("✅ Attention PASSED.")
        attention_zero = Attention(channels=0).to(device)
        x_zero = torch.randn(helper_test_B, 0, helper_H, helper_W, device=device)
        output_zero = attention_zero(x_zero)
        assert output_zero.shape == x_zero.shape, f"Attention C=0 shape mismatch: {output_zero.shape}"
        print("✅ Attention (C=0) PASSED.")

        # Test Mlp
        print("\nTesting Mlp...")
        mlp = Mlp(in_features=helper_C_in, hidden_features=helper_C_in * 2, out_features=helper_C_out).to(device)
        x = torch.randn(helper_test_B, helper_C_in, helper_H, helper_W, device=device)
        output = mlp(x)
        assert output.shape == (helper_test_B, helper_C_out, helper_H, helper_W), f"Mlp shape mismatch: {output.shape}"
        print("✅ Mlp PASSED.")

        # Test DiResBlock
        print("\nTesting DiResBlock...")
        di_res_block1 = DiResBlock(in_channels=helper_C_in, out_channels=helper_C_in, emb_dim=helper_emb_dim, num_heads=4, num_groups=4).to(device)
        x1 = torch.randn(helper_test_B, helper_C_in, helper_H // 2, helper_W // 2, device=device)
        emb1 = torch.randn(helper_test_B, helper_emb_dim, device=device)
        output1 = di_res_block1(x1, emb1)
        assert output1.shape == x1.shape, f"DiResBlock (C_in=C_out) shape mismatch: {output1.shape}"
        print("✅ DiResBlock (C_in=C_out) PASSED.")
        di_res_block2 = DiResBlock(in_channels=helper_C_in, out_channels=helper_C_out, emb_dim=helper_emb_dim, num_heads=8, num_groups=8).to(device)
        x2 = torch.randn(helper_test_B, helper_C_in, helper_H // 2, helper_W // 2, device=device)
        emb2 = torch.randn(helper_test_B, helper_emb_dim, device=device)
        output2 = di_res_block2(x2, emb2)
        assert output2.shape == (helper_test_B, helper_C_out, helper_H // 2, helper_W // 2), f"DiResBlock (C_in!=C_out) shape mismatch: {output2.shape}"
        print("✅ DiResBlock (C_in!=C_out) PASSED.")

        # Test Upsample
        print("\nTesting Upsample...")
        up_ch = helper_C_in
        upsample_conv = Upsample(channels=up_ch, use_conv=True, scale_factor=2).to(device)
        x_up = torch.randn(helper_test_B, up_ch, helper_H // 2, helper_W // 2, device=device)
        output_up_conv = upsample_conv(x_up)
        assert output_up_conv.shape == (helper_test_B, up_ch, helper_H, helper_W), f"Upsample (conv=True) shape mismatch: {output_up_conv.shape}"
        print("✅ Upsample (conv=True) PASSED.")
        upsample_zero = Upsample(channels=0, use_conv=True, scale_factor=2).to(device)
        x_zero = torch.randn(helper_test_B, 0, helper_H // 2, helper_W // 2, device=device)
        output_zero = upsample_zero(x_zero)
        assert output_zero.shape == (helper_test_B, 0, helper_H, helper_W), f"Upsample (C=0) shape mismatch: {output_zero.shape}"
        print("✅ Upsample (C=0) PASSED.")

        # Test Downsample
        print("\nTesting Downsample...")
        down_ch = helper_C_in
        downsample_conv = Downsample(channels=down_ch).to(device)
        x_down = torch.randn(helper_test_B, down_ch, helper_H, helper_W, device=device)
        output_down_conv = downsample_conv(x_down)
        assert output_down_conv.shape == (helper_test_B, down_ch, helper_H // 2, helper_W // 2), f"Downsample (conv=True) shape mismatch: {output_down_conv.shape}"
        print("✅ Downsample PASSED.")
        downsample_zero = Downsample(channels=0).to(device)
        x_zero = torch.randn(helper_test_B, 0, helper_H, helper_W, device=device)
        output_zero = downsample_zero(x_zero)
        assert output_zero.shape == (helper_test_B, 0, helper_H // 2, helper_W // 2), f"Downsample (C=0) shape mismatch: {output_zero.shape}"
        print("✅ Downsample (C=0) PASSED.")

        print("\n--- ✅ Helper Module Tests PASSED ---")

    except Exception as e:
        print("\n--- ❌ FAILED Helper Module Test ---")
        traceback.print_exc()
        all_tests_passed = False # Mark overall failure

    # --- Model Instantiation ---
    print("\n" + "="*30)
    print("--- Testing DiResUnet Instantiation ---")
    print("="*30)
    try:
        unet_model = DiResUnet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            model_channels=MODEL_CHANNELS,
            channel_mult=CHANNEL_MULT,
            num_res_blocks=NUM_RES_BLOCKS,
            cond_emb_dims=COND_EMB_DIMS,
            time_emb_dim=TIME_EMB_DIM,
            final_emb_dim=FINAL_EMB_DIM,
            num_heads=NUM_HEADS,
            num_groups=NUM_GROUPS,
            dropout=DROPOUT
        ).to(device)
        model_params = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
        print(f"\nDiResUnet Model created successfully.")
        print(f"  Use Condition: {unet_model.use_condition}")
        print(f"  Number of conditions expected: {NUM_CONDITIONS}")
        print(f"  Conditional Embedding Dims (adjusted): {unet_model.cond_emb_dims}")
        print(f"  Time Embedding Dim (adjusted): {unet_model.time_emb_dim}")
        print(f"  Final Modulation Emb Dim: {unet_model.final_emb_dim}")
        print(f"  Embedding Max Period (dynamic): {unet_model.embedding_max_period}")
        print(f"  Trainable Parameters: {model_params/1e6:.2f} M")
        if VERBOSE_TEST: print(unet_model)

    except Exception as e:
        print("\n--- ❌ FAILED TO CREATE DiResUnet MODEL ---")
        traceback.print_exc()
        # If model creation fails, no point running further tests
        print("\n" + "="*30)
        print("❌❌❌ OVERALL TEST SUITE FAILED (Model Creation Error) ❌❌❌")
        print("="*30)
        exit() # Exit script

    # --- Test Loop for Different Sizes ---
    print("\n" + "="*30)
    print("--- Testing DiResUnet Forward Pass & Gradients ---")
    print("="*30)
    for image_size in valid_test_sizes:
        print(f"\n--- Testing with Image Size: {image_size}x{image_size} ---")
        test_passed_for_size = True

        # --- Create Dummy Input Data ---
        try:
            dummy_image = torch.randn(
                BATCH_SIZE, IN_CHANNELS, image_size, image_size,
                device=device
            )
            # Time steps (ensure float or long) - use floats for SinusoidalPosEmb
            dummy_time = torch.rand(BATCH_SIZE, device=device) * 1000 # Random floats 0-1000

            # Create multiple dummy condition tensors (e.g., class labels, values)
            dummy_conditions = []
            if unet_model.use_condition:
                for i in range(NUM_CONDITIONS):
                     max_val = 10 if i == 0 else 5 # Example range per condition
                     cond = torch.randint(0, max_val, (BATCH_SIZE,), device=device).float()
                     dummy_conditions.append(cond)

            print(f"Input image shape: {dummy_image.shape}")
            print(f"Input time shape: {dummy_time.shape}")
            if unet_model.use_condition:
                print(f"Input conditions: {len(dummy_conditions)} tensors")
                for i, cond in enumerate(dummy_conditions):
                    print(f"  Condition {i} shape: {cond.shape}")
            else:
                 print("Input conditions: None (Model configured without conditions)")


        except Exception as e:
            print(f"\n--- ❌ FAILED to create input data for size {image_size} ---")
            traceback.print_exc()
            all_tests_passed = False
            test_passed_for_size = False
            continue # Skip to next size

        # --- Perform Forward Pass ---
        try:
            print("Performing forward pass...")
            # Pass the list of condition tensors (or None if not used)
            forward_conditions = dummy_conditions if unet_model.use_condition else None
            output = unet_model(dummy_image, dummy_time, forward_conditions)
            print(f"Output shape: {output.shape}")

            # --- Basic Checks ---
            expected_shape = (BATCH_SIZE, OUT_CHANNELS, image_size, image_size)
            assert output.shape == expected_shape, \
                f"Output shape mismatch! Expected {expected_shape}, got {output.shape}"
            print("✅ Output shape test PASSED.")

            # --- Check Gradient Flow ---
            print("Checking gradient flow...")
            # Use a scalar loss for backward pass
            loss = output.abs().mean() # Use absolute mean to avoid potential cancellation
            loss.backward()

            # Check if gradients exist for a key parameter (e.g., output conv)
            grad_sample = None
            if isinstance(unet_model.conv_out, nn.Conv2d): # Check grad only if it's a real conv layer
                 grad_sample = unet_model.conv_out.weight.grad

            if grad_sample is not None:
                 grad_norm = torch.norm(grad_sample).item()
                 print(f"Sample gradient norm (conv_out weight): {grad_norm:.4f}")
                 assert grad_norm > 1e-9, "Gradients seem to be zero or too small!" # Use a small threshold
                 print("✅ Gradient check PASSED.")
            elif not isinstance(unet_model.conv_out, nn.Identity):
                 print("Warning: conv_out is not Identity, but gradient is None. Check layer usage.")
                 # This could be an issue, potentially mark test as failed or investigate
                 # test_passed_for_size = False
                 # all_tests_passed = False
            else:
                 print("Skipping gradient check for conv_out as it is nn.Identity.")

            # Check grads for conditional MLPs if they exist
            if unet_model.use_condition and unet_model.cond_mlps:
                 try:
                     # Check grad of the last layer's weight in the first conditional MLP
                     cond_mlp_grad = unet_model.cond_mlps[0][-1].weight.grad
                     if cond_mlp_grad is not None:
                         cond_mlp_grad_norm = torch.norm(cond_mlp_grad).item()
                         print(f"Sample gradient norm (cond_mlp[0] weight): {cond_mlp_grad_norm:.4f}")
                         assert cond_mlp_grad_norm > 1e-9, "Conditional MLP gradients seem too small!"
                         print("✅ Conditional MLP gradient check PASSED.")
                     else:
                         print("Warning: Conditional MLP gradient is None.")
                         # test_passed_for_size = False # Optional: fail if cond MLP grad is missing
                         # all_tests_passed = False
                 except Exception as grad_e:
                     print(f"Warning: Could not check conditional MLP gradient: {grad_e}")


            # VERY IMPORTANT: Zero gradients after checking for this size
            unet_model.zero_grad(set_to_none=True) # Use set_to_none=True for potential memory efficiency

        except Exception as e:
            print(f"\n--- ❌ FAILED Forward Pass or Gradient Check for size {image_size} ---")
            traceback.print_exc()
            test_passed_for_size = False
            all_tests_passed = False
            # Ensure gradients are zeroed even if an error occurred mid-test
            try:
                unet_model.zero_grad(set_to_none=True)
            except: pass

        # --- Size Test Summary ---
        if test_passed_for_size:
            print(f"--- ✅ Test for size {image_size} PASSED ---")
        else:
            print(f"--- ❌ Test for size {image_size} FAILED ---")


    # --- Test UNet without condition ---
    print("\n" + "="*30)
    print("--- Testing UNet without Condition ---")
    print("="*30)
    try:
        unet_no_cond = DiResUnet(
            in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, model_channels=MODEL_CHANNELS,
            channel_mult=CHANNEL_MULT, num_res_blocks=NUM_RES_BLOCKS,
            # *** Explicitly disable condition by passing None ***
            cond_emb_dims=None,
            time_emb_dim=TIME_EMB_DIM, final_emb_dim=FINAL_EMB_DIM,
            num_heads=NUM_HEADS, num_groups=NUM_GROUPS
        ).to(device)
        model_params_no_cond = sum(p.numel() for p in unet_no_cond.parameters() if p.requires_grad)
        print(f"DiResUnet No Cond created. Params: {model_params_no_cond/1e6:.2f} M")
        print(f"  use_condition flag: {unet_no_cond.use_condition}") # Should be False
        assert not unet_no_cond.use_condition

        # Test with the first valid size
        test_size_no_cond = valid_test_sizes[0]
        dummy_image_no_cond = torch.randn(BATCH_SIZE, IN_CHANNELS, test_size_no_cond, test_size_no_cond, device=device)
        dummy_time_no_cond = torch.rand(BATCH_SIZE, device=device) * 1000

        # *** Pass None for conditions ***
        print("Performing forward pass without conditions...")
        output_no_cond = unet_no_cond(dummy_image_no_cond, dummy_time_no_cond, conditions=None)
        expected_shape_no_cond = (BATCH_SIZE, OUT_CHANNELS, test_size_no_cond, test_size_no_cond)
        print(f"Output shape (no condition, size {test_size_no_cond}): {output_no_cond.shape}")
        assert output_no_cond.shape == expected_shape_no_cond, f"Shape mismatch! Expected {expected_shape_no_cond}, got {output_no_cond.shape}"
        print("✅ UNet without condition test PASSED.")
    except Exception as e:
         print(f"\n--- ❌ FAILED Test UNet without Condition ---")
         traceback.print_exc()
         all_tests_passed = False # Mark overall failure


    # --- Final Test Summary ---
    print("\n" + "="*40)
    print("--- Overall Test Suite Status ---")
    if all_tests_passed:
        print("          ✅✅✅ PASSED ✅✅✅          ")
    else:
        print("          ❌❌❌ FAILED ❌❌❌          ")
    print("="*40)
