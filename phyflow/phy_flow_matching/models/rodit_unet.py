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


class SpatialMQA(nn.Module):
    """
    A plug-and-play Spatial Multi-Query Attention module.

    This module treats each spatial location (H*W) of the input feature map
    as a query and performs attention with a single Key/Value pair derived
    from an external condition vector. This is a direct application of the
    Multi-Query Attention (MQA) concept for spatial feature modulation.
    """
    def __init__(self,
                 channels: int,
                 emb_dim: int,
                 head_dim: int = None,
                 output_type: str = 'mask'):
        """
        Initializes the SpatialMQA module.

        Args:
            channels (int): Number of channels (C) of the input feature map.
            emb_dim (int): Dimension of the external condition vector.
            head_dim (int, optional): The dimension for Q, K, V projections.
                                      Defaults to `channels` if None.
            output_type (str, optional): The type of output to generate.
                                         Must be one of ['mask', 'feature'].
                                         Defaults to 'mask'.
        """
        super().__init__()

        if output_type not in ['mask', 'feature']:
            raise ValueError("output_type must be either 'mask' or 'feature'")

        self.output_type = output_type
        self.head_dim = head_dim if head_dim is not None else channels

        # Projection layers
        self.to_q = nn.Linear(channels, self.head_dim, bias=False)
        self.to_kv = nn.Linear(emb_dim, self.head_dim * 2, bias=False)

        if self.output_type == 'mask':
            # Layer to project attended features to a single score for the mask
            self.to_score = nn.Linear(self.head_dim, 1)
        else: # 'feature' mode
            # Layer to project attended features back to the original channel dimension
            self.to_out = nn.Linear(self.head_dim, channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SpatialMQA module.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).
            condition (torch.Tensor): Condition vector of shape (B, emb_dim).

        Returns:
            torch.Tensor: The output tensor.
                          If output_type is 'mask', shape is (B, 1, H, W).
                          If output_type is 'feature', shape is (B, C, H, W).
        """
        B, C, H, W = x.shape

        # 1. Prepare Q, K, V
        # Flatten x to a sequence of H*W spatial tokens (queries)
        q_input = x.view(B, C, H * W).permute(0, 2, 1) # (B, H*W, C)
        q = self.to_q(q_input) # (B, H*W, head_dim)

        # Treat condition as a single K/V pair
        condition_unsqueezed = condition.unsqueeze(1) # (B, 1, emb_dim)
        k, v = self.to_kv(condition_unsqueezed).chunk(2, dim=-1) # k, v are (B, 1, head_dim)

        # 2. Perform scaled dot-product attention
        # PyTorch automatically handles the MQA optimization here due to the shapes
        # q: (B, H*W, head_dim), k: (B, 1, head_dim), v: (B, 1, head_dim)
        attn_output = F.scaled_dot_product_attention(q, k, v, enable_gqa=True) # (B, H*W, head_dim)

        # 3. Generate final output based on the specified type
        if self.output_type == 'mask':
            scores = self.to_score(attn_output) # (B, H*W, 1)
            mask = scores.view(B, H, W, 1).permute(0, 3, 1, 2) # (B, 1, H, W)
            return self.sigmoid(mask)
        else: # 'feature' mode
            features = self.to_out(attn_output) # (B, H*W, channels)
            features = features.permute(0, 2, 1).view(B, C, H, W) # (B, C, H, W)
            return features


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


class ConditionalMSAWithRoPE(nn.Module):
    """A Multi-Head Self-Attention block for fusing conditional embeddings
    with RoPE.

    This module takes a sequence of embeddings, treats them as a sequence, and
    applies multi-head self-attention with Rotary Positional Embeddings (RoPE).
    This allows the embeddings to interact and create a context-aware fusion
    across different representation subspaces.

    Attributes:
        dim (int): The feature dimension of the embeddings.
        num_heads (int): The number of attention heads.
        seq_len (int): The length of the input sequence (number of conditions).
        head_dim (int): The dimension of each attention head.
        input_projection (nn.Sequential): Pre-processing MLP for the input.
        norm (nn.RMSNorm): Pre-attention layer normalization.
        qkv_projection (nn.Linear): The layer to project the input sequence
            to Q, K, V.
        rope (RotaryEmbedding1D): The 1D rotary embedding module, applied per
            head.
        head_combine_proj (nn.Linear): A linear layer to combine the outputs
            from different heads.
        seq_combine_proj (nn.Linear): A linear layer to combine the sequence
            length into a single output.
    """

    def __init__(self, dim: int, num_heads: int, seq_len: int, qkv_bias: bool = False):
        """Initializes the ConditionalMSAWithRoPE module.

        Args:
            dim (int): The feature dimension of the input and output. Must be
                divisible by num_heads.
            num_heads (int): The number of attention heads.
            seq_len (int): The length of the input sequence.
            qkv_bias (bool): If True, adds a learnable bias to the QKV
                projection.

        Raises:
            ValueError: If `dim` is not divisible by `num_heads`.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"Dimension ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = dim // num_heads

        # The input projection remains the same
        self.input_projection = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm = nn.RMSNorm(dim, eps=1e-6)

        # QKV projection layer is also the same
        self.qkv_projection = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # RoPE is now applied on the head dimension
        self.rope = RotaryEmbedding1D(dim=self.head_dim)

        # An output projection layer is added to combine heads
        self.head_combine_proj = nn.Linear(dim, dim)

        # A projection to combine the sequence length into a single output
        self.seq_combine_proj = nn.Linear(seq_len, 1, bias=False)

    def forward(self, embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the multi-head attention block.

        Args:
            embeddings (Sequence[torch.Tensor]): A sequence of embedding
                tensors. Each tensor should have a shape of (Batch, Dim).

        Returns:
            torch.Tensor: An aggregated embedding tensor of shape (Batch, Dim).
        """
        # Stack embeddings to form a sequence: (B, S, D)
        # S is the number of conditions.
        x = torch.stack(embeddings, dim=1)
        B, S, D = x.shape
        residual = x

        # 1. Pre-processing
        x = self.input_projection(x)
        x = self.norm(x)

        # 2. Project to Q, K, V
        # (B, S, D) -> (B, S, 3 * D)
        qkv = self.qkv_projection(x)

        # 3. Reshape for Multi-Head Attention
        # (B, S, 3 * D) -> (B, S, 3, H, D_h) -> (3, B, H, S, D_h)
        # where H is num_heads, and D_h is head_dim
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape of each: (B, H, S, D_h)

        # 4. Apply RoPE to Query and Key
        # To apply RoPE, we temporarily merge the Batch and Head dimensions
        q_orig_shape = q.shape
        k_orig_shape = k.shape
        q_reshaped = q.reshape(B * self.num_heads, S, self.head_dim)
        k_reshaped = k.reshape(B * self.num_heads, S, self.head_dim)

        q_rot, k_rot = self.rope(q_reshaped, k_reshaped)

        # Reshape back to the original multi-head format
        q = q_rot.view(q_orig_shape)
        k = k_rot.view(k_orig_shape)

        # 5. Compute scaled dot-product attention
        # F.scaled_dot_product_attention handles the multi-head format directly
        # Input shapes: (B, H, S, D_h). Output shape: (B, H, S, D_h)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # 6. Combine heads
        # (B, H, S, D_h) -> (B, S, H, D_h) -> (B, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)

        # 7. Final projection
        # (B, S, D) -> (B, S, D)
        out = self.head_combine_proj(attn_output)

        # 8. Add residual
        processed_sequence = (residual + out).permute(0, 2, 1)

        # 9. Combine sequence length into a single output
        processed_sequence = self.seq_combine_proj(processed_sequence)

        return processed_sequence.squeeze(-1)


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

        self.norm1 = nn.GroupNorm(num_groups, channels, affine=False, eps=1e-6)
        self.attn = MSAWithRoPE(
            channels=self.channels,
            num_heads=num_heads,
            padding_mode=padding_mode,
            alpha=alpha,
        )
        self.eca = ECA(channels)
        self.norm2 = nn.GroupNorm(num_groups, channels, affine=False, eps=1e-6)
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

        self.norm1 = nn.GroupNorm(num_groups, channels, affine=False, eps=1e-6)
        self.dw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            groups=channels  # Depthwise convolution
        )
        self.eca = ECA(channels)
        self.norm2 = nn.GroupNorm(num_groups, channels, affine=False, eps=1e-6)
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
# ConditionalDownsampling & ConditionalUpsampling Modules
# ==============================================================================

class ConditionalDownsample(nn.Module):
    """Downsampling layer conditioned by SpatialMQA.

    This layer downsamples the input by combining features from a strided
    convolution, average pooling, and max pooling. It then uses a SpatialMQA
    module to gate the concatenated features based on an external condition
    before a final projection.

    Attributes:
        out_channels (int): Number of output channels.
        is_identity (bool): If True, the module acts as a pass-through.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        padding_mode: str = "circular"
    ):
        """Initializes the ConditionalDownsample module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_dim (int): Dimension of the external condition vector for MQA.
            padding_mode (str): Padding mode for the strided convolution.
        """
        super().__init__()
        self.out_channels = out_channels
        self.is_identity = in_channels == 0 or out_channels == 0
        if self.is_identity:
            return

        self.conv_in = nn.Conv2d(
            in_channels, in_channels, 3,
            stride=2, padding=1, padding_mode=padding_mode, groups=in_channels
        )
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.max_pool = nn.MaxPool2d(2, stride=2)

        # MQA module operates on the concatenated features
        mqa_in_channels = in_channels * 3
        self.spatial_mqa = SpatialMQA(
            channels=mqa_in_channels,
            emb_dim=emb_dim,
            output_type='mask'
        )

        self.conv_out = nn.Conv2d(mqa_in_channels, out_channels, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Forward pass for the ConditionalDownsample module.

        Args:
            x (Tensor): Input feature map of shape (B, C_in, H, W).
            condition (Tensor): The external condition vector of shape (B, emb_dim).

        Returns:
            Tensor: The downsampled and conditioned feature map of shape
                    (B, C_out, H/2, W/2).
        """
        if self.is_identity:
            B, _, H, W = x.shape
            new_H, new_W = H // 2, W // 2
            return torch.zeros((B, 0, new_H, new_W), dtype=x.dtype, device=x.device)

        features = torch.cat([
            self.conv_in(x), self.avg_pool(x), self.max_pool(x)
        ], dim=1)

        spatial_mask = self.spatial_mqa(features, condition)
        gated_features = features * spatial_mask

        return self.conv_out(gated_features)


class ConditionalUpsample(nn.Module):
    """Upsampling layer using PixelShuffle, conditioned by SpatialMQA.

    This layer first increases channel count, then uses PixelShuffle to
    increase spatial resolution. It applies a conditional spatial gate via
    SpatialMQA after the upsampling operation and before a final
    refinement convolution.

    Attributes:
        out_channels (int): Number of output channels.
        scale_factor (int): The factor for spatial upsampling.
        is_identity (bool): If True, the module acts as a pass-through.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        scale_factor: int = 2,
        padding_mode: str = "circular"
    ):
        """Initializes the ConditionalUpsample module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_dim (int): Dimension of the external condition vector for MQA.
            scale_factor (int): The factor for spatial upsampling.
            padding_mode (str): Padding mode for convolutions.
        """
        super().__init__()
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.is_identity = in_channels == 0 or out_channels == 0
        if self.is_identity:
            return

        self.conv1 = nn.Conv2d(
            in_channels, out_channels * (scale_factor ** 2), 3,
            padding=1, padding_mode=padding_mode, groups=in_channels
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        # MQA module operates after PixelShuffle, on the target channel count
        self.spatial_mqa = SpatialMQA(
            channels=out_channels,
            emb_dim=emb_dim,
            output_type='mask'
        )

        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Forward pass for the ConditionalUpsample module.

        Args:
            x (Tensor): Input feature map of shape (B, C_in, H, W).
            condition (Tensor): The external condition vector of shape (B, emb_dim).

        Returns:
            Tensor: The upsampled and conditioned feature map of shape
                    (B, C_out, H*S, W*S), where S is the scale_factor.
        """
        if self.is_identity:
            B, _, H, W = x.shape
            new_H, new_W = H * self.scale_factor, W * self.scale_factor
            return torch.zeros((B, 0, new_H, new_W), dtype=x.dtype, device=x.device)

        x = self.conv1(x)
        x = self.pixel_shuffle(x)

        spatial_mask = self.spatial_mqa(x, condition)
        gated_x = x * spatial_mask

        return self.conv2(gated_x)


# ==============================================================================
# Main Model: RoDitUnet
# ==============================================================================
class RoDitUnet(nn.Module):
    """Diffusion Transformer like U-Net with Rotary Embedding.

    This model architecture follows the standard U-Net pattern with an encoder,
    a bottleneck, and a decoder with skip connections. To significantly improve
    computational speed and efficiency, this implementation deliberately adopts an
    "extract-then-expand" design choice. The main blocks at each level process
    features *before* the downsampling layers expand the channel depth for the
    next level. This approach markedly reduces the parameter count and computational
    load, making the model inherently lighter and faster.

    It uses RoDitBlock as the main building block and supports time and conditional
    embeddings. The `start_attn_level` parameter complements the efficiency-focused
    design by allowing the use of ResnetBlocks for shallower layers and switching to
    RoDitBlocks (with self-attention) at deeper levels, further balancing performance
    and computational cost.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        model_channels: Base number of channels for the first level of the U-Net.
        downsample_out_ch_mult (Sequence[int]): A sequence of multipliers that defines the
            output channels for each downsampling layer, which in turn sets the input
            channels for the *next* level. For example, with `model_channels=32` and
            `downsample_out_ch_mult=(1, 2, 4)`, the process is as follows:
            - **Level 0 Blocks** operate on `32` channels.
            - **Level 0 Downsampler** outputs `32 * 1 = 32` channels.
            - **Level 1 Blocks** operate on `32` channels.
            - **Level 1 Downsampler** outputs `32 * 2 = 64` channels.
            - **The final element (4)** is for the bottleneck's channel multiplier.
            Note: The blocks at a given level `i` do *not* operate on `model_channels * downsample_out_ch_mult[i]`.
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
        model_channels: int = 32,
        downsample_out_ch_mult: Sequence[int] = (2, 2, 4,),
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
            downsample_out_ch_mult (Sequence[int]): A sequence of multipliers that defines the
                output channels for each downsampling layer, which in turn sets the input
                channels for the *next* level. For example, with `model_channels=32` and
                `downsample_out_ch_mult=(1, 2, 4)`, the process is as follows:
                - **Level 0 Blocks** operate on `32` channels.
                - **Level 0 Downsampler** outputs `32 * 1 = 32` channels.
                - **Level 1 Blocks** operate on `32` channels.
                - **Level 1 Downsampler** outputs `32 * 2 = 64` channels.
                - **The final element (4)** is for the bottleneck's channel multiplier.
                Note: The blocks at a given level `i` do *not* operate on `model_channels * downsample_out_ch_mult[i]`.
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
        if len(downsample_out_ch_mult) < 2:
            raise ValueError(
                "downsample_out_ch_mult must have at least two elements: "
                "one for the first encoder/decoder level and one for the bottleneck."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.downsample_out_ch_mult = downsample_out_ch_mult
        self.num_levels = len(downsample_out_ch_mult) - 1
        self.encoder_channel_mults = downsample_out_ch_mult[:-1]
        self.bottleneck_channel_mult = downsample_out_ch_mult[-1]
        if start_attn_level < 0:
            raise ValueError(f"start_attn_level must be positive, got {start_attn_level}.")
        self.start_attn_level = start_attn_level
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.padding_mode = padding_mode
        self.alpha = alpha

        # --- Embedding Setup ---
        emb_dim = emb_dim or model_channels
        if emb_dim % 2 != 0:
            emb_dim += 1
        self.emb_dim = emb_dim

        self.time_embedder = SinusoidalPosEmb(self.emb_dim)

        self.num_conditions = num_conditions
        if self.num_conditions > 0:
            self.cond_embedders = nn.ModuleList([SinusoidalPosEmb(self.emb_dim) for _ in range(num_conditions)])
            self.embedding_attention = ConditionalMSAWithRoPE(
                dim=self.emb_dim,
                num_heads=num_heads,
                seq_len=num_conditions+1
            )

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

        # Pre-build a clear channel schedule
        # e.g., model_channels=64, downsample_out_ch_mult=(1, 2, 4) yields ch_schedule=[64, 64, 128, 256]
        ch_schedule = [model_channels] + [model_channels * m for m in self.encoder_channel_mults]

        # Iterate over each encoder level
        for i in range(self.num_levels):
            # Channels for the blocks at the CURRENT level, which is the output from the PREVIOUS level's downsampler.
            current_level_channels = ch_schedule[i]

            # Channels for the NEXT level, which will be the output of THIS level's downsampler.
            next_level_channels = ch_schedule[i + 1]

            # Determine whether to use ResnetBlock or RoDitBlock
            if i < self.start_attn_level:
                Block = ResnetBlock
                block_args = resnet_block_args
            else:
                Block = RoDitBlock
                block_args = rodit_block_args

            # Build the blocks for the current level. They operate on `current_level_channels`.
            self.down_blocks.append(
                nn.ModuleList([Block(channels=current_level_channels, **block_args) for _ in range(num_blocks)])
            )

            # Store `current_level_channels` for the skip connection.
            skip_channels.append(current_level_channels)

            # Create a downsampler to transition from `current_level_channels` to `next_level_channels`.
            self.down_samplers.append(
                ConditionalDownsample(
                    in_channels=current_level_channels,
                    out_channels=next_level_channels,
                    emb_dim=self.emb_dim,
                    padding_mode=padding_mode,
                )
            )

        # Update channels entering the bottleneck
        bottleneck_input_ch = ch_schedule[-1]

        # === Bottleneck ===
        bottleneck_ch = model_channels * self.bottleneck_channel_mult
        self.to_bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_input_ch, bottleneck_ch, kernel_size=1),
            GRN(bottleneck_ch, eps=1e-6),
        )
        self.middle_blocks = nn.ModuleList([
            RoDitBlock(channels=bottleneck_ch, **rodit_block_args),
            RoDitBlock(channels=bottleneck_ch, **rodit_block_args)
        ])
        ch = bottleneck_ch

        # === Decoder ===
        self.up_blocks = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        self.up_proj_convs = nn.ModuleList()

        # Build decoder from the bottom up (deepest to shallowest)
        for i in reversed(range(self.num_levels)):
            target_ch = skip_channels.pop()

            # Create an upsampler to transition from the deeper layer's channel count (`ch`)
            # to the current level's channel count (`target_ch`).
            # The ch in the first iteration is the bottleneck channel count.
            self.up_samplers.append(
                ConditionalUpsample(
                    in_channels=ch,
                    out_channels=target_ch,
                    emb_dim=self.emb_dim,
                    scale_factor=2,
                    padding_mode=padding_mode
                )
            )

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
        self.out_norm = nn.GroupNorm(num_groups, ch, eps=1e-6) if ch > 0 and ch % num_groups == 0 else nn.Identity()
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
        min_size = 2 ** self.num_levels
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
            h = self.down_samplers[i](h, final_emb)

        # === Bottleneck ===
        h = self.to_bottleneck(h)
        for block in self.middle_blocks:
            h = block(h, final_emb)

        # === Decoder ===
        # The decoder modules were built from deep to shallow, so we iterate through them directly.
        for i in range(self.num_levels):
            # Retrieve the corresponding module from ModuleList (index i corresponds to the i-th decoder level, from deep to shallow)

            # Upsample and change channels of h from the deeper layer via Upsampler
            h = self.up_samplers[i](h, final_emb)

            # Extract the corresponding skip connection
            skip_h = skips.pop()  # pop() retrieves from the end, matching the deep-to-shallow order

            # Concatenate
            h = torch.cat([h, skip_h], dim=1)

            # Project to merge channels
            h = self.up_proj_convs[i](h)

            # Execute blocks at this level
            for block in self.up_blocks[i]:
                h = block(h, final_emb)

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

    def test_spatial_mqa(self):
        """Test SpatialMQA module for both 'mask' and 'feature' modes."""
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        condition = torch.randn(self.batch_size, self.emb_dim).to(self.device)

        mqa_mask = SpatialMQA(
            channels=self.channels,
            emb_dim=self.emb_dim,
            output_type='mask'
        ).to(self.device)
        mask_output = mqa_mask(x, condition)

        self.assertEqual(mask_output.shape, (self.batch_size, 1, self.height, self.width),
                         "Shape mismatch in 'mask' mode")

        self.assertTrue(torch.all(mask_output >= 0) and torch.all(mask_output <= 1),
                        "Mask output values should be in the range [0, 1]")

        mqa_feature = SpatialMQA(
            channels=self.channels,
            emb_dim=self.emb_dim,
            output_type='feature'
        ).to(self.device)
        feature_output = mqa_feature(x, condition)

        self.assertEqual(feature_output.shape, x.shape,
                         "Shape mismatch in 'feature' mode")

        self.assertFalse(torch.allclose(feature_output, x, atol=1e-6),
                         "Feature output should be different from the input")

        with self.assertRaises(ValueError, msg="Should raise ValueError for invalid output_type"):
            SpatialMQA(channels=self.channels, emb_dim=self.emb_dim, output_type='invalid_type')

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

    def test_conditional_msa_with_rope(self):
        """Test ConditionalMSAWithRoPE module."""
        num_heads = 4
        self.assertTrue(self.emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads for test.")
        msa_cond = ConditionalMSAWithRoPE(
            dim=self.emb_dim,
            num_heads=num_heads,
            seq_len=self.seq_len,
        ).to(self.device)
        embeddings = [
            torch.randn(self.batch_size, self.emb_dim).to(self.device)
            for _ in range(self.seq_len)
        ]
        output = msa_cond(embeddings)
        self.assertEqual(output.shape, (self.batch_size, self.emb_dim))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaises(ValueError):
            ConditionalMSAWithRoPE(dim=self.emb_dim, num_heads=7, seq_len=self.seq_len)
        wrong_embeddings = embeddings + [torch.randn(self.batch_size, self.emb_dim).to(self.device)]
        with self.assertRaises(RuntimeError):
            msa_cond(wrong_embeddings)

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

    def test_conditional_downsample(self):
        """Test ConditionalDownsample module."""
        down = ConditionalDownsample(
            in_channels=self.channels, out_channels=self.channels * 2, emb_dim=self.emb_dim
        ).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = down(x, torch.randn(self.batch_size, self.emb_dim).to(self.device))
        expected_shape = (self.batch_size, self.channels * 2, self.height // 2, self.width // 2)
        self.assertEqual(output.shape, expected_shape)

    def test_conditional_upsample(self):
        """Test ConditionalUpsample module."""
        scale_factor = 2
        up = ConditionalUpsample(
            in_channels=self.channels, out_channels=self.channels // 2, scale_factor=scale_factor, emb_dim=self.emb_dim
        ).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = up(x, torch.randn(self.batch_size, self.emb_dim).to(self.device))
        expected_shape = (self.batch_size, self.channels // 2, self.height * scale_factor, self.width * scale_factor)
        self.assertEqual(output.shape, expected_shape)


class TestRoDitUnet(unittest.TestCase):
    """
    Comprehensive tests for the full RoDitUnet model, updated for the new
    downsample_out_ch_mult behavior where the last element controls the bottleneck.
    """

    def setUp(self):
        """Set up common variables for the U-Net tests."""
        self.batch_size = 2
        self.in_channels = 3
        self.out_channels = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # To test 3 Encoder/Decoder levels, downsample_out_ch_mult must have 4 elements.
        # With 3 downsampling stages (2^3=8), height/width should be a multiple of 8.
        self.height = 32
        self.width = 32
        self.base_config = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "model_channels": 32,
            "downsample_out_ch_mult": (1, 2, 4, 8),  # Represents 3 encoder/decoder levels + 1 bottleneck level
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
        num_levels = len(self.base_config["downsample_out_ch_mult"]) - 1
        self.assertEqual(num_levels, 3, "Test setup should result in 3 encoder/decoder levels.")

        # Update test cases to match the 3-level structure (levels 0, 1, 2).
        test_cases = [
            {
                "desc": "All levels use attention",
                "start_attn_level": 0,
                "expected_block": RoDitBlock,  # Applies to all levels
            },
            {
                "desc": "Attention starts at level 1",
                "start_attn_level": 1,
                "expected_block": [ResnetBlock, RoDitBlock, RoDitBlock],  # For levels 0, 1, 2
            },
            {
                "desc": "Attention starts at the last level (level 2)",
                "start_attn_level": num_levels - 1,  # This is 2
                "expected_block": [ResnetBlock, ResnetBlock, RoDitBlock],  # For levels 0, 1, 2
            },
            {
                "desc": "Only bottleneck uses attention (all encoder/decoder levels are Resnet)",
                "start_attn_level": num_levels,  # This is 3
                "expected_block": ResnetBlock,  # Applies to all levels
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
                # up_blocks are ordered from deep to shallow (corresponding to levels 2, 1, 0)
                for i, level_blocks in enumerate(model.up_blocks):
                    # Convert the up_block index i (0, 1, 2) back to the level index (2, 1, 0)
                    level_idx = num_levels - 1 - i
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
