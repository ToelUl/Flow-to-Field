# -*- coding: utf-8 -*-
"""
Flexible Diffusion Transformer Model supporting variable input resolutions.

This module implements a Diffusion Transformer (DiT) model inspired by
Stable Diffusion 3 and FiT principles. It accepts a noisy latent image of
variable height and width, a timestep, and one or more scalar conditions
as input. It uses 2D Rotary Position Embedding (RoPE) instead of
absolute positional embeddings to handle variable input sizes and incorporates
Masked Multi-Head Self-Attention. Scalar conditions are embedded using
sinusoidal positional embeddings and combined with the timestep embedding
for modulation within the transformer blocks.
"""

import torch
import torch.nn as nn
import math
import unittest
from typing import Sequence, Union, Optional, Tuple

# ========= Helper Functions & Modules =========

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies adaptive instance normalization modulation.

    Args:
        x: Input tensor of shape (B, N, C) or (B, C). B is batch size, N is
           number of tokens (optional), C is channel dimension.
        shift: Shift tensor of shape (B, C).
        scale: Scale tensor of shape (B, C).

    Returns:
        Modulated tensor with the same shape as x.

    Raises:
        ValueError: If the input tensor `x` has an unexpected dimension or
                    if batch/channel dimensions mismatch between x and shift/scale.
    """
    # Unsqueeze shift and scale to match x's dimensions if x is (B, N, C)
    if x.dim() == 3:
        if shift.shape[0] != x.shape[0] or scale.shape[0] != x.shape[0]:
             raise ValueError(f"Batch size mismatch: x({x.shape[0]}) vs shift/scale({shift.shape[0]})")
        if shift.shape[1] != x.shape[2] or scale.shape[1] != x.shape[2]:
             raise ValueError(f"Channel dimension mismatch: x({x.shape[2]}) vs shift/scale({shift.shape[1]})")
        # Ensure shift/scale have correct device before unsqueeze
        shift = shift.to(x.device)
        scale = scale.to(x.device)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif x.dim() == 2:
        # This case might be less common now with tokens, but keep for robustness
        if shift.shape[0] != x.shape[0] or scale.shape[0] != x.shape[0]:
             raise ValueError(f"Batch size mismatch for 2D input: x({x.shape[0]}) vs shift/scale({shift.shape[0]})")
        if shift.shape[1] != x.shape[1] or scale.shape[1] != x.shape[1]:
             raise ValueError(f"Channel dimension mismatch for 2D input: x({x.shape[1]}) vs shift/scale({shift.shape[1]})")
        # Ensure shift/scale have correct device
        shift = shift.to(x.device)
        scale = scale.to(x.device)
        return x * (1 + scale) + shift
    else:
        raise ValueError(f"Input tensor x has unexpected dimension: {x.dim()}. Expected 2 or 3.")

# ========= Flexible Vision Transformer Components =========

class FlexiblePatchEmbed(nn.Module):
    """ Flexible 2D Image to Patch Embedding.

    Divides an image of variable H, W into patches and projects them linearly.
    """
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, norm_layer: Optional[nn.Module] = None, flatten: bool = True):
        """Initializes the flexible patch embedding layer.

        Args:
            patch_size: The size of each square patch. Defaults to 16.
            in_chans: Number of input channels. Defaults to 3.
            embed_dim: The dimension of the output token embeddings. Defaults to 768.
            norm_layer: Optional normalization layer to apply after projection. Defaults to None.
            flatten: Whether to flatten the spatial dimensions of the output tokens. Defaults to True.
        """
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten

        # Convolutional layer to extract patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Performs the forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            A tuple containing:
            - Embedded patches tensor of shape (B, N, E) if flatten is True,
              otherwise (B, E, H_patch, W_patch). N = H_patch * W_patch.
            - H_patch: Number of patches along the height dimension.
            - W_patch: Number of patches along the width dimension.

        Raises:
            ValueError: If input H or W are not divisible by patch_size or if input is not 4D.
        """
        if x.dim() != 4:
             raise ValueError(f"Input tensor x must be 4-dimensional (B, C, H, W), but got shape {x.shape}")
        B, C, H, W = x.shape
        pH, pW = self.patch_size
        if H % pH != 0 or W % pW != 0:
            raise ValueError(f"Input image dimensions ({H}x{W}) must be divisible by patch size ({pH}x{pW}).")

        H_patch = H // pH
        W_patch = W // pW

        x = self.proj(x) # Shape: (B, E, H_patch, W_patch)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # Shape: (B, N, E), where N = H_patch * W_patch
        x = self.norm(x)
        return x, H_patch, W_patch

class RotaryEmbedding(nn.Module):
    """Implements 2D Rotary Position Embedding (RoPE).

    Applies rotary embeddings to input tensors (typically Query and Key in Attention)
    based on their 2D spatial positions.

    Args:
        dim: The feature dimension of the input tensor to be rotated. This is
             typically the dimension per attention head (head_dim).
        max_seq_len_h: Maximum expected height (in patches) for pre-computation.
                       Defaults to 256.
        max_seq_len_w: Maximum expected width (in patches) for pre-computation.
                       Defaults to 256.
        base: The base value for frequency calculation. Defaults to 10000.
        device: Device for precomputed embeddings. Defaults to None (uses input device).
    """
    def __init__(
        self,
        dim: int,
        max_seq_len_h: int = 256,
        max_seq_len_w: int = 256,
        base: int = 10000,
        device: Optional[torch.device] = None, # Changed default to None
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len_h = max_seq_len_h
        self.max_seq_len_w = max_seq_len_w
        self.base = base
        # Store device, default to None if not provided
        self._device = device # Use internal name to avoid conflict

        # dim must be divisible by 4 for 2D RoPE (D/4 per coordinate)
        if dim % 4 != 0:
             raise ValueError(f"RoPE dimension ({dim}) must be divisible by 4 for 2D split.")
        self.half_dim = dim // 2
        self.quarter_dim = dim // 4

        # Determine the device to use for tensor creation
        tensor_device = self._device if self._device is not None else torch.device("cpu")

        inv_freq = 1.0 / (self.base ** (torch.arange(
            0, self.quarter_dim,
            dtype=torch.float32,
            device=tensor_device # Use determined device
            ) / self.quarter_dim))
        # Register inv_freq as buffer
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Compute embeddings
        emb_cos_h, emb_sin_h = self._compute_embeddings(inv_freq, max_seq_len_h, tensor_device)
        emb_cos_w, emb_sin_w = self._compute_embeddings(inv_freq, max_seq_len_w, tensor_device)

        # *** Register buffers directly without prior self assignment ***
        self.register_buffer("emb_cos_h", emb_cos_h, persistent=False)
        self.register_buffer("emb_sin_h", emb_sin_h, persistent=False)
        self.register_buffer("emb_cos_w", emb_cos_w, persistent=False)
        self.register_buffer("emb_sin_w", emb_sin_w, persistent=False)


    def _compute_embeddings(self, inv_freq: torch.Tensor, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes sin/cos embeddings for a given sequence length."""
        # `inv_freq` should already be on the correct device
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq) # Shape: (seq_len, dim/4)
        cos_emb = torch.cos(freqs) # Shape: (seq_len, dim/4)
        sin_emb = torch.sin(freqs) # Shape: (seq_len, dim/4)
        return cos_emb, sin_emb

    def _get_pos_embeddings(self, H_patch: int, W_patch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves precomputed cos/sin embeddings for given grid dimensions."""
        if H_patch > self.max_seq_len_h or W_patch > self.max_seq_len_w:
            raise ValueError(f"Input grid size ({H_patch}x{W_patch}) exceeds RoPE max lengths ({self.max_seq_len_h}x{self.max_seq_len_w}).")

        # Retrieve precomputed embeddings (buffers should handle device)
        emb_cos_h = self.emb_cos_h.to(device) # Ensure device match anyway
        emb_sin_h = self.emb_sin_h.to(device)
        emb_cos_w = self.emb_cos_w.to(device)
        emb_sin_w = self.emb_sin_w.to(device)

        # Get embeddings for the required H and W dimensions
        cos_h = emb_cos_h[:H_patch] # Shape: (H_patch, dim/4)
        sin_h = emb_sin_h[:H_patch] # Shape: (H_patch, dim/4)
        cos_w = emb_cos_w[:W_patch] # Shape: (W_patch, dim/4)
        sin_w = emb_sin_w[:W_patch] # Shape: (W_patch, dim/4)

        # Create grid coordinates (indices)
        grid_h, grid_w = torch.meshgrid(
            torch.arange(H_patch, device=device),
            torch.arange(W_patch, device=device),
            indexing='ij' # Important: ij indexing for H, W
        ) # Shapes: (H_patch, W_patch)

        # Gather embeddings based on grid coordinates and flatten
        # Shapes: (N, dim/4), where N = H_patch * W_patch
        cos_h_seq = cos_h[grid_h.flatten()].view(-1, self.quarter_dim)
        sin_h_seq = sin_h[grid_h.flatten()].view(-1, self.quarter_dim)
        cos_w_seq = cos_w[grid_w.flatten()].view(-1, self.quarter_dim)
        sin_w_seq = sin_w[grid_w.flatten()].view(-1, self.quarter_dim)

        # Reshape for broadcasting: (1, N, dim/4)
        return (
            cos_h_seq.unsqueeze(0),
            sin_h_seq.unsqueeze(0),
            cos_w_seq.unsqueeze(0),
            sin_w_seq.unsqueeze(0)
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Helper function to rotate half the dimensions."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
         """Applies RoPE rotation to a tensor x using precomputed cos/sin values.

         Args:
             x: Input tensor, shape (..., N, D_half). D_half = RoPE dim / 2.
             cos: Cosine embeddings, shape (1, N, D_quarter). D_quarter = RoPE dim / 4.
             sin: Sine embeddings, shape (1, N, D_quarter).

         Returns:
             Rotated tensor, same shape as x.
         """
         if x.shape[-1] != cos.shape[-1] * 2:
              if x.shape[-1] % 2 == 0 and cos.shape[-1] == x.shape[-1] // 2:
                  pass
              else:
                  raise ValueError(f"Shape mismatch: Input last dim {x.shape[-1]} must be twice cos/sin last dim {cos.shape[-1]}")

         x1, x2 = x.chunk(2, dim=-1) # Each shape (..., N, D_quarter)
         rotated_x = torch.cat(
             [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
         )
         return rotated_x

    def forward(self, x: torch.Tensor, H_patch: int, W_patch: int) -> torch.Tensor:
        """Applies 2D RoPE to the input tensor.

        Args:
            x: Input tensor, typically Q or K, shape (B, N, E) or (B, H, N, E_head).
               If 4D, assumes H is num_heads. E or E_head must match self.dim.
            H_patch: Number of patches along the height dimension.
            W_patch: Number of patches along the width dimension.

        Returns:
            Tensor with rotary embeddings applied, same shape as input x.
        """
        shape = x.shape
        device = x.device # Get device from input tensor

        if x.dim() == 4: # (B, num_heads, N, head_dim)
            B, H_heads, N, E_head = shape
            x_rope = x.reshape(B * H_heads, N, E_head)
        elif x.dim() == 3: # (B, N, E)
            B, N, E = shape
            x_rope = x
            H_heads = 1
        else:
            raise ValueError(f"Input tensor x has unexpected dimension: {x.dim()}. Expected 3 or 4.")

        if x_rope.shape[-1] != self.dim:
             raise ValueError(f"Input tensor final dim ({x_rope.shape[-1]}) does not match RoPE dim ({self.dim})")

        cos_h, sin_h, cos_w, sin_w = self._get_pos_embeddings(H_patch, W_patch, device)

        x_rope_h, x_rope_w = x_rope.split(self.half_dim, dim=-1)

        x_rotated_h = self.apply_rotary_pos_emb(x_rope_h, cos_h, sin_h)
        x_rotated_w = self.apply_rotary_pos_emb(x_rope_w, cos_w, sin_w)

        x_out = torch.cat((x_rotated_h, x_rotated_w), dim=-1)

        if x.dim() == 4:
            x_out = x_out.view(B, H_heads, N, E_head)

        return x_out


# Need to import Mlp correctly if timm is used partially
try:
    from timm.models.vision_transformer import Mlp
except ImportError:
    print("Warning: timm library not found. Defining a basic Mlp.")
    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x


class FlexibleAttention(nn.Module):
    """Multi-Head Self-Attention with 2D RoPE and optional Masking.

    Args:
        dim: Total dimension of the input tokens.
        num_heads: Number of attention heads. Defaults to 8.
        qkv_bias: Whether to include bias in the QKV linear layers. Defaults to False.
        attn_drop: Dropout probability for attention weights. Defaults to 0.0.
        proj_drop: Dropout probability for the output projection. Defaults to 0.0.
        rope: An instance of the RotaryEmbedding module. Must be provided.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        if dim <= 0 or num_heads <= 0:
             raise ValueError("dim and num_heads must be positive.")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        if rope is None:
            raise ValueError("A RotaryEmbedding instance must be provided to FlexibleAttention.")
        self.head_dim = dim // num_heads
        if rope.dim != self.head_dim:
            raise ValueError(f"RoPE dimension ({rope.dim}) must match attention head dimension ({self.head_dim}).")

        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H_patch: int, W_patch: int, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for FlexibleAttention.

        Args:
            x: Input tensor of shape (B, N, C), where C is `dim`.
            H_patch: Number of patches along height.
            W_patch: Number of patches along width.
            attn_mask: Optional attention mask of shape (B, N, N) or (N, N).
                       Masked positions should be -inf or a large negative number,
                       unmasked positions should be 0.

        Returns:
            Output tensor of shape (B, N, C).
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor x must have 3 dimensions (B, N, C), got {x.dim()}")
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.rope(q, H_patch, W_patch)
        k = self.rope(k, H_patch, W_patch)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            else:
                 raise ValueError(f"attn_mask has unexpected dimensions: {attn_mask.dim()}. Expected 2 or 3.")
            attn_mask = attn_mask.to(attn.device)
            # Ensure mask broadcasts correctly: (B or 1, 1, N, N) + (B, H, N, N) -> (B, H, N, N)
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# ========= DiT Block (Modified for FlexibleAttention) =========

class DitBlock(nn.Module):
    """A Diffusion Transformer (DiT) block with adaptive layer norm modulation,
       using FlexibleAttention with 2D RoPE.

    Args:
        hidden_dim: Dimension of the hidden state.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio for MLP hidden dimension. Defaults to 4.0.
        rope: An instance of the RotaryEmbedding module. Must be provided.
    """
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or mlp_ratio <= 0:
            raise ValueError("hidden_dim, num_heads, and mlp_ratio must be positive.")
        if rope is None:
             raise ValueError("A RotaryEmbedding instance must be provided to DitBlock.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = FlexibleAttention(
            hidden_dim, num_heads=num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, rope=rope
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, H_patch: int, W_patch: int, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for the DiT block.

        Args:
            x: Input tensor (B, N, C).
            c: Conditioning tensor (B, C).
            H_patch: Number of patches along height for RoPE.
            W_patch: Number of patches along width for RoPE.
            attn_mask: Optional attention mask.

        Returns:
            Output tensor (B, N, C).
        """
        if x.dim() != 3 or x.shape[2] != self.hidden_dim:
             raise ValueError(f"Input x shape error: {x.shape}. Expected (B, N, {self.hidden_dim})")
        if c.dim() != 2 or c.shape[1] != self.hidden_dim:
             raise ValueError(f"Conditioning c shape error: {c.shape}. Expected (B, {self.hidden_dim})")
        if x.shape[0] != c.shape[0]:
             raise ValueError(f"Batch size mismatch x ({x.shape[0]}) vs c ({c.shape[0]})")

        c_mod = c.to(next(self.adaLN_modulation.parameters()).device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c_mod).chunk(6, dim=1)

        residual = x
        x_norm1 = self.norm1(x)
        x_mod1 = modulate(x_norm1, shift_msa, scale_msa)
        x_attn = self.attn(x_mod1, H_patch, W_patch, attn_mask=attn_mask)
        x = residual + gate_msa.unsqueeze(1) * x_attn

        residual = x
        x_norm2 = self.norm2(x)
        x_mod2 = modulate(x_norm2, shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_mod2)
        x = residual + gate_mlp.unsqueeze(1) * x_mlp

        return x

# ========= Final Layer (Unchanged) =========

class FinalLayer(nn.Module):
    """The final layer of the Flexible DiT model.

    Args:
        hidden_dim: Dimension of the input token features.
        patch_size: Size of the square image patches.
        out_channels: Number of output channels.
    """
    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        if hidden_dim <= 0 or patch_size <= 0 or out_channels <= 0:
            raise ValueError("hidden_dim, patch_size, and out_channels must be positive.")

        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass for the final layer.

        Args:
            x: Input tensor (B, N, C).
            c: Conditioning tensor (B, C).

        Returns:
            Output tensor representing flattened patches, (B, N, P*P*C_out).
        """
        if x.dim() != 3 or x.shape[2] != self.hidden_dim:
             raise ValueError(f"Input x shape error: {x.shape}. Expected (B, N, {self.hidden_dim})")
        if c.dim() != 2 or c.shape[1] != self.hidden_dim:
             raise ValueError(f"Conditioning c shape error: {c.shape}. Expected (B, {self.hidden_dim})")
        if x.shape[0] != c.shape[0]:
             raise ValueError(f"Batch size mismatch x ({x.shape[0]}) vs c ({c.shape[0]})")

        c_mod = c.to(next(self.adaLN_modulation.parameters()).device)
        shift, scale = self.adaLN_modulation(c_mod).chunk(2, dim=1)

        x_norm = self.norm_final(x)
        x_mod = modulate(x_norm, shift, scale)
        x = self.linear(x_mod)
        return x

# ========= Embedding Modules (TimestepEmbedder, ScalarConditionEmbedder) =========

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Args:
        hidden_dim: Output embedding dimension.
        frequency_embedding_size: Intermediate sinusoidal embedding dimension. Defaults to 256.
    """
    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        if hidden_dim <= 0 or frequency_embedding_size <= 0:
             raise ValueError("hidden_dim and frequency_embedding_size must be positive.")
        if frequency_embedding_size % 2 != 0:
             raise ValueError("frequency_embedding_size must be even.")

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.out_dim = hidden_dim

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep (or general scalar) embeddings."""
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, but got {dim}")
        assert t.ndim == 1, f"Input tensor t must be 1-dimensional, got {t.ndim}."

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embeds timesteps."""
        if t.ndim != 1:
             raise ValueError(f"Input t must be 1-dimensional (B,), got shape {t.shape}")
        t = t.to(next(self.mlp.parameters()).device)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ScalarConditionEmbedder(nn.Module):
    """Embeds scalar conditions into vector representations.

    Args:
        hidden_dim: Output embedding dimension.
        frequency_embedding_size: Intermediate sinusoidal embedding dimension. Defaults to 256.
    """
    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        if hidden_dim <= 0 or frequency_embedding_size <= 0:
             raise ValueError("hidden_dim and frequency_embedding_size must be positive.")
        if frequency_embedding_size % 2 != 0:
             raise ValueError("frequency_embedding_size must be even.")

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.out_dim = hidden_dim

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Embeds scalar conditions."""
        if c.ndim != 1:
             raise ValueError(f"Input c must be 1-dimensional (B,), got shape {c.shape}")
        c = c.to(next(self.mlp.parameters()).device)
        c_freq = TimestepEmbedder.timestep_embedding(c, self.frequency_embedding_size)
        c_emb = self.mlp(c_freq)
        return c_emb


# ========= Main Model: Flexible Diffusion Transformer =========
@torch.compile
class Flexible_Diffusion_Transformer(nn.Module):
    """
    A Flexible Diffusion Transformer (DiT) supporting variable input resolutions.

    Args:
        patch_size: Size of the patches.
        in_channels: Number of input channels.
        hidden_dim: Transformer hidden dimension.
        depth: Number of DitBlock layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio. Defaults to 4.0.
        learn_sigma: If True, predicts both noise and variance. Defaults to True.
        cond_freq_emb_size: Freq embedding size for conditions. Defaults to 256.
        time_freq_emb_size: Freq embedding size for timestep. Defaults to 256.
        rope_base: Base value for RoPE frequency calculation. Defaults to 10000.
        max_rope_res_h: Max height (patches) for RoPE. Defaults to 64.
        max_rope_res_w: Max width (patches) for RoPE. Defaults to 64.
    """
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        cond_freq_emb_size: int = 256,
        time_freq_emb_size: int = 256,
        rope_base: int = 10000,
        max_rope_res_h: int = 64,
        max_rope_res_w: int = 64,
    ):
        super().__init__()

        if patch_size <= 0 or in_channels <= 0 or hidden_dim <= 0 or depth <= 0 or num_heads <= 0:
            raise ValueError("Dimensions must be positive.")
        if hidden_dim % num_heads != 0:
             raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.x_embedder = FlexiblePatchEmbed(
            patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_dim
        )
        self.t_embedder = TimestepEmbedder(
            hidden_dim=hidden_dim, frequency_embedding_size=time_freq_emb_size
        )
        self.c_embedder = ScalarConditionEmbedder(
            hidden_dim=hidden_dim, frequency_embedding_size=cond_freq_emb_size
        )

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len_h=max_rope_res_h,
            max_seq_len_w=max_rope_res_w,
            base=rope_base,
            device=None # Device determined later
        )

        self.blocks = nn.ModuleList([
            DitBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, rope=self.rope) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_dim, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """Initializes model weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if hasattr(block.mlp, 'fc2') and isinstance(block.mlp.fc2, nn.Linear):
                nn.init.constant_(block.mlp.fc2.weight, 0)
                nn.init.constant_(block.mlp.fc2.bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x_tokens: torch.Tensor, H_patch: int, W_patch: int) -> torch.Tensor:
        """Converts patch tokens back into image format.

        Args:
            x_tokens: Tensor of shape (B, N, P*P*C_out).
            H_patch: Number of patches along height.
            W_patch: Number of patches along width.

        Returns:
            Image tensor of shape (B, C_out, H, W).
        """
        if x_tokens.dim() != 3:
             raise ValueError(f"Input x must be 3D (B, N, PPCOut), got {x_tokens.shape}")

        B, N, PPCOut = x_tokens.shape
        P = self.patch_size
        C_out = self.out_channels
        expected_N = H_patch * W_patch
        expected_PPCOut = P * P * C_out
        if N != expected_N:
            raise ValueError(f"Input N ({N}) mismatch H_patch*W_patch ({expected_N}).")
        if PPCOut != expected_PPCOut:
             raise ValueError(f"Input PPCOut ({PPCOut}) mismatch P*P*C_out ({expected_PPCOut}).")

        x = x_tokens.view(B, H_patch, W_patch, P, P, C_out)
        x = x.permute(0, 5, 1, 3, 2, 4) # B, C_out, H_patch, P, W_patch, P
        H, W = H_patch * P, W_patch * P
        image = x.reshape(B, C_out, H, W)
        return image

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Union[torch.Tensor, Sequence[torch.Tensor]],
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the Flexible DiT model.

        Args:
            x: Input latent tensor (B, C_in, H, W).
            t: Timestep tensor (B,).
            conditions: Single condition tensor (B,) or sequence of tensors.
            attn_mask: Optional attention mask (B, N, N) or (N, N).

        Returns:
            Output tensor (B, C_out, H, W).
        """
        model_device = next(self.parameters()).device
        x = x.to(model_device)
        t = t.to(model_device)
        batch_size = x.shape[0]

        # Input validation check
        if x.dim() != 4 or x.shape[1] != self.in_channels:
             raise ValueError(f"Input x shape error: {x.shape}. Expected (B, {self.in_channels}, H, W)")
        if t.dim() != 1 or t.shape[0] != batch_size:
             raise ValueError(f"Input t shape error: {t.shape}. Expected ({batch_size},)")

        x_tokens, H_patch, W_patch = self.x_embedder(x)
        t_emb = self.t_embedder(t)

        c_total_emb = torch.zeros_like(t_emb)
        if isinstance(conditions, torch.Tensor):
            if not (conditions.shape == (batch_size,)):
                raise ValueError(f"Single condition c shape error: {conditions.shape}. Expected ({batch_size},)")
            conditions = conditions.to(model_device)
            c_total_emb = self.c_embedder(conditions)
        elif isinstance(conditions, (list, tuple)):
            for i, c_tensor in enumerate(conditions):
                if not isinstance(c_tensor, torch.Tensor):
                     raise TypeError(f"Condition element {i} is not a Tensor: {type(c_tensor)}")
                if not (c_tensor.shape == (batch_size,)):
                    raise ValueError(f"Condition tensor {i} shape error: {c_tensor.shape}. Expected ({batch_size},)")
                c_tensor = c_tensor.to(model_device)
                c_emb = self.c_embedder(c_tensor)
                c_total_emb = c_total_emb + c_emb
        else:
            raise TypeError(f"Conditions must be Tensor or sequence, got {type(conditions)}")

        c_combined = t_emb + c_total_emb

        for block in self.blocks:
            x_tokens = block(x_tokens, c_combined, H_patch, W_patch, attn_mask=attn_mask)

        output_tokens = self.final_layer(x_tokens, c_combined)
        output_image = self.unpatchify(output_tokens, H_patch, W_patch)

        return output_image


# ========= Updated Test Suite =========

class TestFlexibleDiffusionTransformer(unittest.TestCase):
    """Unit test suite for the Flexible_Diffusion_Transformer and components."""

    def setUp(self):
        """Set up common parameters and models for testing."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n--- Setting up test on device: {self.device} ---")

        self.patch_size = 4
        self.in_channels = 4
        self.hidden_dim = 128
        self.depth = 2
        self.num_heads = 4
        self.head_dim = self.hidden_dim // self.num_heads
        self.batch_size = 2
        self.num_classes = 10
        self.num_styles = 5
        self.max_rope_res = 32

        self.res_list = [(32, 32), (24, 40), (48, 32)]

        self.dummy_data = {}
        for H, W in self.res_list:
            shape = (self.batch_size, self.in_channels, H, W)
            self.dummy_data[(H, W)] = {
                "x": torch.randn(shape, device=self.device),
                "t": torch.randint(0, 1000, (self.batch_size,), device=self.device),
                "c_single": torch.randint(0, self.num_classes, (self.batch_size,), device=self.device),
                "c_list": [
                    torch.randint(0, self.num_classes, (self.batch_size,), device=self.device),
                    torch.randint(0, self.num_styles, (self.batch_size,), device=self.device)
                ]
            }

        # Initialize RoPE for component tests here, ensuring correct device handling
        try:
             self.test_rope = RotaryEmbedding(
                 dim=self.head_dim,
                 max_seq_len_h=self.max_rope_res,
                 max_seq_len_w=self.max_rope_res,
                 device=self.device
             )
        except Exception as e:
             print(f"\nWARNING: Failed to initialize self.test_rope in setUp: {e}")
             self.test_rope = None


    def test_01_model_initialization(self):
        """Test if the main flexible model initializes correctly."""
        print("Test 01: Flexible Model Initialization")
        try:
            model = Flexible_Diffusion_Transformer(
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                hidden_dim=self.hidden_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                learn_sigma=True,
                max_rope_res_h=self.max_rope_res,
                max_rope_res_w=self.max_rope_res,
            ).to(self.device)
            self.assertIsNotNone(model, "Model should initialize.")
            self.assertEqual(next(model.parameters()).device.type, self.device.type, "Model parameters not on correct device.")
            self.assertEqual(model.rope.dim, self.head_dim, "Model RoPE dim mismatch.")
            print("PASS: Flexible Model initialized successfully.")
        except Exception as e:
            self.fail(f"Flexible Model initialization failed: {e}")

    def test_02_flexible_patch_embed(self):
        """Test FlexiblePatchEmbed with different resolutions."""
        print("Test 02: FlexiblePatchEmbed")
        embedder = FlexiblePatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_dim
        ).to(self.device)

        for (H, W), data in self.dummy_data.items():
            print(f"  Testing resolution: {H}x{W}")
            x = data["x"]
            tokens, H_patch, W_patch = embedder(x)
            expected_H_patch = H // self.patch_size
            expected_W_patch = W // self.patch_size
            expected_N = expected_H_patch * expected_W_patch
            expected_shape = (self.batch_size, expected_N, self.hidden_dim)
            self.assertEqual(H_patch, expected_H_patch, f"H_patch mismatch for {H}x{W}")
            self.assertEqual(W_patch, expected_W_patch, f"W_patch mismatch for {H}x{W}")
            self.assertEqual(tokens.shape, expected_shape, f"Token shape mismatch for {H}x{W}")
            print(f"  PASS: Resolution {H}x{W} -> Tokens {tokens.shape}, Grid {H_patch}x{W_patch}")
        print("PASS: FlexiblePatchEmbed works for multiple resolutions.")


    def test_03_rotary_embedding(self):
        """Test the RotaryEmbedding module."""
        print("Test 03: RotaryEmbedding")
        if self.test_rope is None: self.skipTest("Skipping RoPE test: setup failed.")

        H_patch, W_patch = 8, 10
        N = H_patch * W_patch
        dummy_q_4d = torch.randn(self.batch_size, self.num_heads, N, self.head_dim, device=self.device)

        try:
            q_rotated_4d = self.test_rope(dummy_q_4d, H_patch, W_patch)
            self.assertEqual(q_rotated_4d.shape, dummy_q_4d.shape, "RoPE 4D output shape mismatch.")
            orig_norm_4d = torch.linalg.norm(dummy_q_4d, dim=-1)
            rot_norm_4d = torch.linalg.norm(q_rotated_4d, dim=-1)
            self.assertTrue(torch.allclose(orig_norm_4d, rot_norm_4d, atol=1e-5), "RoPE significantly changed 4D vector norms.")
            print("  PASS: 4D input case.")

            with self.assertRaises(ValueError): self.test_rope(dummy_q_4d, self.max_rope_res + 1, W_patch)
            with self.assertRaises(ValueError): self.test_rope(dummy_q_4d, H_patch, self.max_rope_res + 1)
            print("  PASS: Max length check.")

            print("PASS: RotaryEmbedding basic checks passed.")
        except Exception as e: self.fail(f"RotaryEmbedding test failed: {e}")


    def test_04_flexible_attention(self):
        """Test the FlexibleAttention module with RoPE and masking."""
        print("Test 04: FlexibleAttention")
        if self.test_rope is None: self.skipTest("Skipping Attention test: RoPE setup failed.")

        H, W = 32, 24
        H_patch, W_patch = H // self.patch_size, W // self.patch_size
        N = H_patch * W_patch
        dummy_tokens = torch.randn(self.batch_size, N, self.hidden_dim, device=self.device)

        attention = FlexibleAttention(
            dim=self.hidden_dim, num_heads=self.num_heads, rope=self.test_rope
        ).to(self.device)
        attention.eval()

        try:
            print("  Testing without mask...")
            with torch.no_grad(): output_no_mask = attention(dummy_tokens, H_patch, W_patch)
            self.assertEqual(output_no_mask.shape, dummy_tokens.shape, "Attention shape mismatch (no mask).")
            print("  PASS: No mask case.")

            print("  Testing with mask...")
            mask = torch.zeros(N, N, device=self.device)
            mask[:, -1] = -torch.inf
            with torch.no_grad(): output_mask = attention(dummy_tokens, H_patch, W_patch, attn_mask=mask)
            self.assertEqual(output_mask.shape, dummy_tokens.shape, "Attention shape mismatch (with mask).")
            self.assertFalse(torch.allclose(output_no_mask, output_mask), "Mask did not affect output.")
            print("  PASS: Masking case.")

            batch_mask = torch.zeros(self.batch_size, N, N, device=self.device)
            batch_mask[0, :, -1] = -torch.inf
            with torch.no_grad(): output_batch_mask = attention(dummy_tokens, H_patch, W_patch, attn_mask=batch_mask)
            self.assertEqual(output_batch_mask.shape, dummy_tokens.shape, "Attention shape mismatch (batch mask).")
            print("  PASS: Batch masking case.")

            print("PASS: FlexibleAttention works with RoPE and masking.")
        except Exception as e: self.fail(f"FlexibleAttention test failed: {e}")

    def test_05_unpatchify_flexible(self):
        """Test the flexible unpatchify method."""
        print("Test 05: unpatchify method (flexible)")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        model = Flexible_Diffusion_Transformer(
            patch_size=self.patch_size, in_channels=self.in_channels,
            hidden_dim=self.hidden_dim, depth=self.depth, num_heads=self.num_heads,
            learn_sigma=True
        ).to(self.device)

        for (H, W), data in self.dummy_data.items():
             print(f"  Testing resolution: {H}x{W}")
             H_patch, W_patch = H // self.patch_size, W // self.patch_size
             N = H_patch * W_patch
             out_channels = self.in_channels * 2
             patch_dim = self.patch_size * self.patch_size * out_channels
             dummy_tokens = torch.randn(self.batch_size, N, patch_dim, device=self.device)
             try:
                 image = model.unpatchify(dummy_tokens, H_patch, W_patch)
                 expected_shape = (self.batch_size, out_channels, H, W)
                 self.assertEqual(image.shape, expected_shape, f"Unpatchify shape mismatch for {H}x{W}.")
                 print(f"  PASS: Resolution {H}x{W} -> Image {image.shape}")
             except Exception as e: self.fail(f"Unpatchify failed for {H}x{W}: {e}")
        print("PASS: unpatchify works for multiple resolutions.")


    def test_06_forward_pass_flexible_res_sigma_true(self):
        """Test forward pass with flexible resolutions (learn_sigma=True)."""
        print("Test 06: Forward Pass Shape (Flexible Res, learn_sigma=True)")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        model = Flexible_Diffusion_Transformer(
            patch_size=self.patch_size, in_channels=self.in_channels, hidden_dim=self.hidden_dim,
            depth=self.depth, num_heads=self.num_heads, learn_sigma=True,
            max_rope_res_h=self.max_rope_res, max_rope_res_w=self.max_rope_res,
        ).to(self.device)
        model.eval()

        for (H, W), data in self.dummy_data.items():
            print(f"  Testing resolution: {H}x{W} (single condition)")
            x, t, c = data["x"], data["t"], data["c_single"]
            try:
                with torch.no_grad(): output = model(x, t, c)
                expected_shape = (self.batch_size, self.in_channels * 2, H, W)
                self.assertEqual(output.shape, expected_shape, f"Output shape mismatch for {H}x{W}.")
                self.assertEqual(output.device.type, self.device.type, f"Device mismatch for {H}x{W}.")
                print(f"  PASS: Resolution {H}x{W} -> Output {output.shape}")
            except Exception as e: self.fail(f"Forward pass failed for {H}x{W} (sigma=True): {e}")
        print("PASS: Forward pass (sigma=True) works for multiple resolutions.")


    def test_07_forward_pass_flexible_res_sigma_false(self):
        """Test forward pass with flexible resolutions (learn_sigma=False)."""
        print("Test 07: Forward Pass Shape (Flexible Res, learn_sigma=False)")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        model = Flexible_Diffusion_Transformer(
            patch_size=self.patch_size, in_channels=self.in_channels, hidden_dim=self.hidden_dim,
            depth=self.depth, num_heads=self.num_heads, learn_sigma=False,
            max_rope_res_h=self.max_rope_res, max_rope_res_w=self.max_rope_res,
        ).to(self.device)
        model.eval()

        for (H, W), data in self.dummy_data.items():
            print(f"  Testing resolution: {H}x{W} (single condition)")
            x, t, c = data["x"], data["t"], data["c_single"]
            try:
                with torch.no_grad(): output = model(x, t, c)
                expected_shape = (self.batch_size, self.in_channels, H, W)
                self.assertEqual(output.shape, expected_shape, f"Output shape mismatch for {H}x{W}.")
                self.assertEqual(output.device.type, self.device.type, f"Device mismatch for {H}x{W}.")
                print(f"  PASS: Resolution {H}x{W} -> Output {output.shape}")
            except Exception as e: self.fail(f"Forward pass failed for {H}x{W} (sigma=False): {e}")
        print("PASS: Forward pass (sigma=False) works for multiple resolutions.")


    def test_08_forward_pass_flexible_res_multi_cond(self):
        """Test forward pass with flexible resolutions and multiple conditions."""
        print("Test 08: Forward Pass Shape (Flexible Res, Multiple Conditions)")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        model = Flexible_Diffusion_Transformer(
            patch_size=self.patch_size, in_channels=self.in_channels, hidden_dim=self.hidden_dim,
            depth=self.depth, num_heads=self.num_heads, learn_sigma=True,
            max_rope_res_h=self.max_rope_res, max_rope_res_w=self.max_rope_res,
        ).to(self.device)
        model.eval()

        for (H, W), data in self.dummy_data.items():
            print(f"  Testing resolution: {H}x{W} (multiple conditions)")
            x, t, c_list = data["x"], data["t"], data["c_list"]
            try:
                with torch.no_grad(): output = model(x, t, c_list)
                expected_shape = (self.batch_size, self.in_channels * 2, H, W)
                self.assertEqual(output.shape, expected_shape, f"Output shape mismatch for {H}x{W}.")
                print(f"  PASS: Resolution {H}x{W} -> Output {output.shape}")
            except Exception as e: self.fail(f"Forward pass failed for {H}x{W} (multi-cond): {e}")
        print("PASS: Forward pass (multi-condition) works for multiple resolutions.")


    def test_09_forward_pass_with_mask(self):
        """Test forward pass with an attention mask."""
        print("Test 09: Forward Pass with Attention Mask")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        model = Flexible_Diffusion_Transformer(
            patch_size=self.patch_size, in_channels=self.in_channels, hidden_dim=self.hidden_dim,
            depth=self.depth, num_heads=self.num_heads, learn_sigma=False,
            max_rope_res_h=self.max_rope_res, max_rope_res_w=self.max_rope_res,
        ).to(self.device)
        model.eval()

        H, W = 32, 32
        data = self.dummy_data[(H,W)]
        x, t, c = data["x"], data["t"], data["c_single"]
        H_patch, W_patch = H // self.patch_size, W // self.patch_size
        N = H_patch * W_patch

        # *** MODIFIED MASK: Mask attention FROM last token ***
        attn_mask = torch.zeros(N, N, device=self.device)
        attn_mask[-1, :] = -torch.inf # Mask attention FROM last token

        try:
            print("  Running forward pass without mask...")
            with torch.no_grad(): output_no_mask = model(x, t, c, attn_mask=None)
            print("  Running forward pass with mask...")
            with torch.no_grad(): output_mask = model(x, t, c, attn_mask=attn_mask)
            expected_shape = (self.batch_size, self.in_channels, H, W)
            self.assertEqual(output_no_mask.shape, expected_shape, "Shape mismatch (no mask).")
            self.assertEqual(output_mask.shape, expected_shape, "Shape mismatch (with mask).")
            # Check that the mask actually changed the output
            self.assertFalse(torch.allclose(output_no_mask, output_mask, atol=1e-6),
                             "Attention mask did not affect output.")
            print("PASS: Forward pass correctly handles attention mask.")
        except AssertionError as e:
             # Provide more info if assertion fails
             diff = torch.abs(output_no_mask - output_mask).max()
             print(f"    Outputs with/without mask are too close. Max difference: {diff.item()}")
             self.fail(f"Forward pass with mask failed: {e}")
        except Exception as e:
            self.fail(f"Forward pass with mask failed: {e}")

    def test_10_timestep_embedder(self):
        """Test the TimestepEmbedder component."""
        print("Test 10: TimestepEmbedder")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        freq_emb_size = 64
        embedder = TimestepEmbedder(hidden_dim=self.hidden_dim, frequency_embedding_size=freq_emb_size).to(self.device)
        t = list(self.dummy_data.values())[0]['t']
        output = embedder(t)
        expected_shape = (self.batch_size, self.hidden_dim)
        self.assertEqual(output.shape, expected_shape, f"TimestepEmbedder shape mismatch.")
        print(f"PASS: TimestepEmbedder output shape {output.shape} is correct.")

    def test_11_scalar_condition_embedder(self):
        """Test the ScalarConditionEmbedder component."""
        print("Test 11: ScalarConditionEmbedder")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        freq_emb_size = 64
        embedder = ScalarConditionEmbedder(hidden_dim=self.hidden_dim, frequency_embedding_size=freq_emb_size).to(self.device)
        c = list(self.dummy_data.values())[0]['c_single']
        output = embedder(c)
        expected_shape = (self.batch_size, self.hidden_dim)
        self.assertEqual(output.shape, expected_shape, f"ScalarConditionEmbedder shape mismatch.")
        print(f"PASS: ScalarConditionEmbedder output shape {output.shape} is correct.")

    def test_12_modulate_function(self):
        """Test the modulate helper function."""
        print("Test 12: modulate function")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        C, N = self.hidden_dim, 50
        x_3d = torch.randn(self.batch_size, N, C, device=self.device)
        x_2d = torch.randn(self.batch_size, C, device=self.device)
        shift = torch.randn(self.batch_size, C, device=self.device)
        scale = torch.randn(self.batch_size, C, device=self.device)

        out_3d = modulate(x_3d, shift, scale)
        self.assertEqual(out_3d.shape, x_3d.shape, "Modulate 3D shape mismatch.")
        out_identity = modulate(x_3d, torch.zeros_like(shift), torch.zeros_like(scale))
        self.assertTrue(torch.allclose(out_identity, x_3d, atol=1e-6), "Modulate identity failed.")

        out_2d = modulate(x_2d, shift, scale)
        self.assertEqual(out_2d.shape, x_2d.shape, "Modulate 2D shape mismatch.")

        print("PASS: modulate function works correctly.")


    def test_13_initialization_edge_cases(self):
        """Test initialization with invalid parameters."""
        print("Test 13: Initialization Edge Cases")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        with self.assertRaises(ValueError): Flexible_Diffusion_Transformer(hidden_dim=0)
        with self.assertRaises(ValueError): Flexible_Diffusion_Transformer(depth=0)
        with self.assertRaises(ValueError): Flexible_Diffusion_Transformer(hidden_dim=130, num_heads=4)
        with self.assertRaises(ValueError): RotaryEmbedding(dim=30) # RoPE dim not div by 4
        bad_rope = RotaryEmbedding(dim=self.head_dim + 4)
        with self.assertRaises(ValueError): FlexibleAttention(dim=self.hidden_dim, num_heads=self.num_heads, rope=bad_rope)
        print("PASS: Initialization correctly handles invalid parameters.")

    def test_14_forward_pass_input_errors(self):
        """Test forward pass with invalid input shapes/types."""
        print("Test 14: Forward Pass Input Errors")
        if self.test_rope is None: self.skipTest("Skipping test: RoPE setup failed.")
        model = Flexible_Diffusion_Transformer(
            patch_size=self.patch_size, in_channels=self.in_channels, hidden_dim=self.hidden_dim,
            depth=self.depth, num_heads=self.num_heads,
            max_rope_res_h=self.max_rope_res, max_rope_res_w=self.max_rope_res,
        ).to(self.device)
        model.eval()

        H, W = 32, 32
        data = self.dummy_data[(H,W)]
        x, t, c_single, c_list = data["x"], data["t"], data["c_single"], data["c_list"]
        B, C, _, _ = x.shape

        # Wrong x dims (should be 4D)
        # *** MODIFIED REGEX to match actual error message ***
        with self.assertRaisesRegex(ValueError, "Input x shape error"):
             bad_x_3d = x.view(B * C, H, W) # Create 3D tensor
             model(bad_x_3d, t, c_single)
        with self.assertRaisesRegex(ValueError, "Input x shape error"):
             model(x.unsqueeze(0), t, c_single) # 5D input

        # Wrong t dims (should be 1D)
        with self.assertRaisesRegex(ValueError, "Input t shape error"):
            model(x, t.unsqueeze(1), c_single) # 2D input

        # Wrong condition type
        with self.assertRaises(TypeError): model(x, t, "not_a_tensor")
        # Wrong condition shape (single)
        with self.assertRaisesRegex(ValueError, "Single condition c shape error"): model(x, t, c_single.unsqueeze(1))
        # Wrong condition shape (list element)
        with self.assertRaisesRegex(ValueError, "Condition tensor 1 shape error"): model(x, t, [c_list[0], c_list[1].unsqueeze(1)])

        # Indivisible H/W (caught by PatchEmbed)
        if self.patch_size > 1:
            bad_x_hw = torch.randn(B, C, H+1, W, device=self.device)
            with self.assertRaisesRegex(ValueError, "must be divisible by patch size"):
                 model(bad_x_hw, t, c_single)
        else: print("  Skipping non-divisible H/W test (patch_size=1)")

        print("PASS: Forward pass correctly handles invalid inputs.")


# ========= Main Execution Block for Testing =========

if __name__ == '__main__':
    print("="*70)
    print(" Starting Unit Tests for Flexible_Diffusion_Transformer ".center(70, "="))
    print("="*70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_names = [name for name in dir(TestFlexibleDiffusionTransformer) if name.startswith('test_')]
    test_names.sort()
    for name in test_names:
        suite.addTest(TestFlexibleDiffusionTransformer(name))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("="*70)
    print(" Unit Tests Completed ".center(70, "="))
    print("="*70)

    # --- Example Usage ---
    if result.wasSuccessful():
        print("\n--- Example Usage (Multiple Resolutions & Conditions) ---")
        try:
            patch_size_ex = 2
            in_channels_ex = 4
            hidden_dim_ex = 256
            depth_ex = 4
            num_heads_ex = 8
            learn_sigma_ex = False
            batch_size_ex = 1
            max_rope_res_ex = 64
            num_classes_ex = 10
            num_styles_ex = 5

            model_ex = Flexible_Diffusion_Transformer(
                patch_size=patch_size_ex, in_channels=in_channels_ex, hidden_dim=hidden_dim_ex,
                depth=depth_ex, num_heads=num_heads_ex, learn_sigma=learn_sigma_ex,
                max_rope_res_h=max_rope_res_ex, max_rope_res_w=max_rope_res_ex
            )
            device_ex = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_ex.to(device_ex)
            model_ex.eval()

            test_resolutions = [(32, 32), (24, 40), (48, 32)]

            for H_ex, W_ex in test_resolutions:
                 print(f"\n--- Testing H={H_ex}, W={W_ex} ---")
                 dummy_x_ex = torch.randn(batch_size_ex, in_channels_ex, H_ex, W_ex, device=device_ex)
                 dummy_t_ex = torch.randint(0, 1000, (batch_size_ex,), device=device_ex)
                 dummy_c1_ex = torch.randint(0, num_classes_ex, (batch_size_ex,), device=device_ex)
                 dummy_c2_ex = torch.randint(0, num_styles_ex, (batch_size_ex,), device=device_ex)
                 dummy_conditions_ex = [dummy_c1_ex, dummy_c2_ex]

                 print(f"Input x shape: {dummy_x_ex.shape}")
                 print(f"Input t shape: {dummy_t_ex.shape}")
                 print(f"Input conditions: list shapes {[c.shape for c in dummy_conditions_ex]}")

                 with torch.no_grad(): output_ex = model_ex(dummy_x_ex, dummy_t_ex, dummy_conditions_ex)

                 expected_out_channels_ex = in_channels_ex * 2 if learn_sigma_ex else in_channels_ex
                 expected_shape_ex = (batch_size_ex, expected_out_channels_ex, H_ex, W_ex)
                 print(f"Output shape: {output_ex.shape}")
                 print(f"Expected shape: {expected_shape_ex}")
                 assert output_ex.shape == expected_shape_ex, f"Example Usage FAILED for {H_ex}x{W_ex}!"
                 print(f"Example Usage PASSED for {H_ex}x{W_ex}.")

            num_params = sum(p.numel() for p in model_ex.parameters() if p.requires_grad)
            print(f"\nTotal Trainable Parameters: {num_params / 1e6:.2f} M")

        except Exception as e:
            print(f"\nExample Usage FAILED: An error occurred.")
            import traceback
            traceback.print_exc()