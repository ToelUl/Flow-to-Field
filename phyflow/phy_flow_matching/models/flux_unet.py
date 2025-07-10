import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence
from dataclasses import dataclass
import unittest

torch.set_float32_matmul_precision('high')

# ==============================================================================
# Section 1: Basic Helper Modules
# ==============================================================================

@dataclass
class ModulationOut:
    """A dataclass for storing modulation parameters.

    Attributes:
        shift: The shift tensor for adaptive layer normalization.
        scale: The scale tensor for adaptive layer normalization.
        gate: The gating tensor to scale the output of a block.
    """
    shift: Tensor
    scale: Tensor
    gate: Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Applies Ada-LayerNorm modulation to a 4D tensor.

    Args:
        x: The input tensor of shape (B, C, H, W).
        shift: The shift tensor of shape (B, C).
        scale: The scale tensor of shape (B, C).

    Returns:
        The modulated tensor.
    """
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half of the dimensions of a tensor.

    This is a helper function for implementing Rotary Positional Embeddings (RoPE).

    Args:
        x: The input tensor. The last dimension must be even.

    Returns:
        A tensor with the second half of its last dimension negated and
        swapped with the first half.
    """
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x_rotated_pairs = torch.cat([-x_reshaped[..., 1:], x_reshaped[..., :1]], dim=-1)
    return x_rotated_pairs.flatten(start_dim=-2)


class RMSNorm2d(nn.Module):
    """A 2D Root Mean Square Layer Normalization module.

    This normalization is applied across the channel dimension of a 4D tensor.

    Attributes:
        eps: A small value added to the denominator for numerical stability.
        elementwise_affine: If True, this module has learnable per-channel
            affine parameters.
        scale: The learnable per-channel scaling parameter.
    """
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """Initializes the RMSNorm2d module.

        Args:
            dim: The number of channels in the input tensor.
            eps: A value added to the denominator for numerical stability.
            elementwise_affine: A boolean value that when set to True, this
                module has learnable per-channel affine parameters.
        """
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.scale = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('scale', None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RMSNorm2d.

        Args:
            x: The input tensor of shape (B, C, H, W).

        Returns:
            The normalized tensor.
        """
        norm_factor = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        x_norm = x * norm_factor
        if self.elementwise_affine:
            scale_reshaped = self.scale.view(1, -1, 1, 1)
            return x_norm * scale_reshaped
        return x_norm


class QKNorm(nn.Module):
    """A module for normalizing Query and Key tensors in an attention mechanism.

    Attributes:
        query_norm: Layer normalization for the query tensor.
        key_norm: Layer normalization for the key tensor.
    """
    def __init__(self, dim: int):
        """Initializes the QKNorm module.

        Args:
            dim: The feature dimension of the Query and Key tensors.
        """
        super().__init__()
        self.query_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.key_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Applies layer normalization to Query and Key.

        Args:
            q: The Query tensor.
            k: The Key tensor.

        Returns:
            A tuple containing the normalized Query and Key tensors.
        """
        return self.query_norm(q), self.key_norm(k)


class Modulation(nn.Module):
    """Generates modulation parameters from a conditioning vector.

    Produces shift, scale, and gate parameters for Ada-LayerNorm. Can optionally
    produce two sets of parameters (e.g., for two resnet blocks).
    """
    def __init__(self, emb_dim: int, out_channels: int, is_double: bool = False):
        """Initializes the Modulation module.

        Args:
            emb_dim: The dimension of the input conditioning vector.
            out_channels: The number of output channels for the modulation params.
            is_double: If True, generates two sets of modulation parameters.
        """
        super().__init__()
        self.multiplier = 6 if is_double else 3
        self.lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, self.multiplier * out_channels, bias=True)
        )
        # Initialize weights and biases to zero for stability.
        nn.init.zeros_(self.lin[-1].weight)
        nn.init.zeros_(self.lin[-1].bias)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, Optional[ModulationOut]]:
        """Generates modulation parameters from the conditioning vector.

        Args:
            vec: The input conditioning vector of shape (B, emb_dim).

        Returns:
            A tuple containing:
                - mod1: The first set of modulation parameters.
                - mod2: The second set of parameters, or None if is_double=False.
        """
        params = self.lin(vec)
        chunks = params.chunk(self.multiplier, dim=-1)
        mod1 = ModulationOut(shift=chunks[0], scale=chunks[1], gate=chunks[2])
        mod2 = None
        if self.multiplier == 6:
            mod2 = ModulationOut(shift=chunks[3], scale=chunks[4], gate=chunks[5])
        return mod1, mod2


class Mlp(nn.Module):
    """A simple MLP implemented with 1x1 convolutions for spatial features.

    Used in non-attention blocks to process features spatially.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        """Initializes the Mlp module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of hidden channels. Defaults to 4*in_channels.
            act_layer: The activation function to use.
            drop: Dropout rate.
        """
        super().__init__()
        hidden_channels = hidden_channels or in_channels * 4
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = act_layer()
        self.pw_conv2 = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            The output tensor.
        """
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw_conv2(x)
        x = self.drop(x)
        return x

# ==============================================================================
# Section 2: Embedding Helpers
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Creates sinusoidal positional embeddings.

    Attributes:
        embedding_dim: The dimension of the embedding.
        max_period: The maximum period of the sinusoidal functions.
        factor: A scaling factor for the input before embedding.
    """
    def __init__(self, embedding_dim: int, max_period: int = 10000, factor: float = 1.0):
        """Initializes the SinusoidalPosEmb module.

        Args:
            embedding_dim: The desired dimension of the embedding.
            max_period: The maximum period for the sinusoidal functions.
            factor: A multiplicative factor applied to the input tensor.
        """
        super().__init__()
        # Ensure embedding_dim is even
        self.embedding_dim = embedding_dim if embedding_dim % 2 == 0 else embedding_dim + 1
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer('freqs', freqs)
        self.factor = factor

    def forward(self, x: Tensor) -> Tensor:
        """Generates sinusoidal embeddings for the input tensor.

        Args:
            x: A 1D tensor of values to be embedded.

        Returns:
            A 2D tensor of shape (len(x), embedding_dim) containing the embeddings.
        """
        args = x.float().unsqueeze(1) * self.freqs.unsqueeze(0) * self.factor
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MLPEmbedder(nn.Module):
    """An MLP for processing embedding vectors.

    Consists of a simple MLP with one hidden layer and SiLU activation.
    """
    def __init__(self, embedding_dim: int):
        """Initializes the MLPEmbedder module.

        Args:
            embedding_dim: The input and output dimension of the MLP.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Processes the embedding vector through the MLP.

        Args:
            x: The input embedding tensor.

        Returns:
            The processed tensor.
        """
        return self.mlp(x)


# ==============================================================================
# Section 3: Rotary Positional Encoding (RoPE) Modules
# ==============================================================================

class RoPE(nn.Module):
    """Standard 1D Rotary Positional Encoding for sequential data.

    Applies RoPE to the last dimension of the input tensor, which is assumed
    to be the feature dimension.
    """
    def __init__(self, dim: int, base: int = 10000):
        """Initializes the RoPE module.

        Args:
            dim: The feature dimension, which must be even.
            base: The base value for the frequency calculation.
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"
        freqs = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (base**freqs)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 1D RoPE to the input tensor.

        Args:
            x: The input tensor of shape (..., seq_len, dim).

        Returns:
            The tensor with RoPE applied.
        """
        seq_len = x.shape[-2]
        device = x.device
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return x * cos + rotate_half(x) * sin


class RoPE_Mixed(nn.Module):
    """2D Rotary Positional Encoding for spatial feature maps.

    Applies RoPE based on 2D grid coordinates (H, W).
    """
    def __init__(self, dim: int):
        """Initializes the RoPE_Mixed module.

        Args:
            dim: The feature dimension, which must be even.
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"
        self.freqs = nn.Parameter(torch.randn(dim // 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 2D RoPE to the input spatial tensor.

        Args:
            x: Input tensor of shape (..., H, W, D).

        Returns:
            The tensor with 2D RoPE applied.
        """
        H, W, D = x.shape[-3:]
        device = x.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        positions_2d = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1).float()

        theta_y = self.freqs[:, 0]
        theta_x = self.freqs[:, 1]
        angles = (torch.einsum("n,d->nd", positions_2d[:, 0], theta_y) +
                  torch.einsum("n,d->nd", positions_2d[:, 1], theta_x))

        cos_vals = angles.cos().repeat_interleave(2, dim=-1).reshape(H, W, D)
        sin_vals = angles.sin().repeat_interleave(2, dim=-1).reshape(H, W, D)

        while cos_vals.dim() < x.dim():
            cos_vals = cos_vals.unsqueeze(0)
            sin_vals = sin_vals.unsqueeze(0)

        return x * cos_vals + rotate_half(x) * sin_vals


# ==============================================================================
# Section 4: Core U-Net Building Blocks
# ==============================================================================

class FluxlikeResnetBlock(nn.Module):
    """A ResNet block inspired by the Flux architecture.

    This block can operate in two modes:
    1.  Attention mode: Fuses depthwise convolution, cross-attention, and an MLP
        into a single path for efficiency. It uses a single modulation step.
    2.  Non-attention mode: A standard ResNet block with two convolution-like
        operations (depthwise conv and MLP), each preceded by modulation.
    """
    def __init__(self, channels: int, emb_dim: int, num_heads: int,
                 use_attention: bool, dropout: float, padding_mode: str):
        """Initializes the FluxlikeResnetBlock.

        Args:
            channels: Number of input and output channels.
            emb_dim: Dimension of the conditioning embedding.
            num_heads: Number of attention heads (if use_attention is True).
            use_attention: Whether to use the attention mechanism.
            dropout: Dropout rate (only used in non-attention MLP).
            padding_mode: The padding mode for convolutions.
        """
        super().__init__()
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.channels = channels
        mlp_ratio = 4

        self.modulation = Modulation(emb_dim, channels, is_double=not use_attention)
        self.norm1 = RMSNorm2d(channels, elementwise_affine=False)

        if self.use_attention:
            self.head_dim = channels // num_heads
            assert self.head_dim * num_heads == channels, \
                "channels must be divisible by num_heads"
            mlp_hidden_channels = int(channels * mlp_ratio)

            # Fused convolution for QKV (spatial) and MLP input
            self.fused_conv1 = nn.Conv2d(
                channels, channels * 3 + mlp_hidden_channels, 3,
                padding=1, groups=channels, padding_mode=padding_mode
            )
            # Linear layer for QKV (conditional)
            self.to_qkv_cond = nn.Linear(emb_dim, channels * 3, bias=False)

            self.rope_spatial = RoPE_Mixed(self.head_dim)
            self.rope_conditional = RoPE(self.head_dim)
            self.qknorm = QKNorm(self.head_dim)
            self.mlp_act = nn.GELU(approximate="tanh")

            # Final projection layer
            self.fused_proj = nn.Conv2d(channels + mlp_hidden_channels, channels, 1)
        else:
            self.norm2 = RMSNorm2d(channels, elementwise_affine=False)
            self.op1 = nn.Conv2d(channels, channels, 3, padding=1,
                                 padding_mode=padding_mode, groups=channels)
            self.mlp = Mlp(channels,
                           hidden_channels=int(channels * mlp_ratio),
                           act_layer=lambda: nn.GELU(approximate="tanh"),
                           drop=dropout)

    def forward(self, x: Tensor, cond_seq: Tensor, final_emb: Tensor) -> Tensor:
        """Forward pass of the FluxlikeResnetBlock.

        Args:
            x: The input spatial tensor of shape (B, C, H, W).
            cond_seq: The sequence of conditioning embeddings (B, S, E).
            final_emb: The final aggregated conditioning embedding (B, E).

        Returns:
            The output tensor of the same shape as x.
        """
        if self.use_attention:
            B, C, H, W = x.shape
            S = cond_seq.shape[1]

            mod, _ = self.modulation(final_emb)
            x_mod = modulate(self.norm1(x), mod.shift, mod.scale)

            # 1. Fused Convolution and MLP path
            fused_output = self.fused_conv1(x_mod)
            qkv_s_raw, mlp_in = torch.split(fused_output, [C * 3, fused_output.shape[1] - C * 3], dim=1)
            mlp_activated = self.mlp_act(mlp_in)

            # 2. Prepare Spatial QKV
            q_s_raw, k_s_raw, v_s_raw = qkv_s_raw.chunk(3, dim=1)
            v_s = v_s_raw.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2)

            # 3. Prepare Conditional QKV
            qkv_c_raw = self.to_qkv_cond(cond_seq).reshape(
                B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q_c_raw, k_c_raw, v_c = qkv_c_raw[0], qkv_c_raw[1], qkv_c_raw[2]

            # 4. Apply RoPE
            q_s_for_rope = q_s_raw.view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
            k_s_for_rope = k_s_raw.view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
            q_s_rot = self.rope_spatial(q_s_for_rope).permute(0, 1, 4, 2, 3).reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
            k_s_rot = self.rope_spatial(k_s_for_rope).permute(0, 1, 4, 2, 3).reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
            q_c_rot = self.rope_conditional(q_c_raw.reshape(B * self.num_heads, S, self.head_dim)).view(B, self.num_heads, S, self.head_dim)
            k_c_rot = self.rope_conditional(k_c_raw.reshape(B * self.num_heads, S, self.head_dim)).view(B, self.num_heads, S, self.head_dim)

            # 5. Concatenate and Normalize
            q = torch.cat((q_s_rot, q_c_rot), dim=2)
            k = torch.cat((k_s_rot, k_c_rot), dim=2)
            v = torch.cat((v_s, v_c), dim=2)
            q, k = self.qknorm(q, k)

            # 6. Scaled Dot-Product Attention
            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_spatial_output = attn_output[:, :, :H*W, :].transpose(1, 2).reshape(B, H * W, C).permute(0, 2, 1).view(B, C, H, W)

            # 7. Final Projection and Gating
            combined = torch.cat([attn_spatial_output, mlp_activated], dim=1)
            projected_out = self.fused_proj(combined)
            return x + mod.gate.unsqueeze(-1).unsqueeze(-1) * projected_out
        else:
            mod1, mod2 = self.modulation(final_emb)

            # First block
            h_mod1 = modulate(self.norm1(x), mod1.shift, mod1.scale)
            h_op1 = self.op1(h_mod1)
            x_res = x + mod1.gate.unsqueeze(-1).unsqueeze(-1) * h_op1

            # Second block (MLP)
            h_mod2 = modulate(self.norm2(x_res), mod2.shift, mod2.scale)
            h_op2 = self.mlp(h_mod2)
            return x_res + mod2.gate.unsqueeze(-1).unsqueeze(-1) * h_op2


class Downsample(nn.Module):
    """Downsamples a feature map by a factor of 2.

    Uses a strided convolution to simultaneously downsample and change the
    number of channels.
    """
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str):
        """Initializes the Downsample module.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            padding_mode: The padding mode for the convolution.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=0)
        self.padding_mode = padding_mode

    def forward(self, x: Tensor, *args) -> Tensor:
        """Forward pass for downsampling.

        The `*args` are included to maintain a consistent signature with
        other blocks but are not used.

        Args:
            x: The input tensor of shape (B, C_in, H, W).

        Returns:
            The downsampled tensor of shape (B, C_out, H/2, W/2).
        """
        # Manual padding to replicate torch's 'same' padding for stride 2
        pad = (0, 1, 0, 1)
        pad_x = nn.functional.pad(x, pad, mode=self.padding_mode)
        return self.conv(pad_x)


class Upsample(nn.Module):
    """Upsamples a feature map by a factor of 2.

    Uses nearest-neighbor upsampling followed by a convolution to change
    the number of channels.
    """
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str):
        """Initializes the Upsample module.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            padding_mode: The padding mode for the convolution.
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x: Tensor, *args) -> Tensor:
        """Forward pass for upsampling.

        The `*args` are included to maintain a consistent signature with
        other blocks but are not used.

        Args:
            x: The input tensor of shape (B, C_in, H, W).

        Returns:
            The upsampled tensor of shape (B, C_out, H*2, W*2).
        """
        return self.conv(self.upsample(x))


# ==============================================================================
# Section 5: Main U-Net Model
# ==============================================================================
@torch.compile
class FluxUNet(nn.Module):
    """A U-Net model with a Flux-style architecture.

    This model uses FluxlikeResnetBlocks, RMSNorm, and RoPE for positional
    embeddings. It processes an input tensor conditioned on a time step and
    an optional sequence of other conditions.

    Attributes:
        emb_dim: The base dimension for embeddings.
        num_conditions: The number of expected conditional inputs.
        time_embedder: Module to create time embeddings.
        cond_embedders: A list of modules for other condition embeddings.
        conv_in: Initial convolution layer.
        down_blocks: A ModuleList for the encoder part of the U-Net.
        middle_blocks: A ModuleList for the bottleneck of the U-Net.
        up_blocks: A ModuleList for the decoder part of the U-Net.
        conv_out: Final convolution layer.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4),
        num_blocks: int = 2,
        num_heads: int = 8,
        start_attn_level: int = 1,
        num_conditions: int = 1,
        dropout: float = 0.0,
        padding_mode: str = "replicate",
    ):
        """Initializes the FluxUNet model.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels in the output tensor.
            model_channels: The base number of channels for the model.
            channel_mults: A sequence of channel multipliers for each U-Net level.
            num_blocks: The number of resnet blocks per U-Net level.
            num_heads: The number of attention heads.
            start_attn_level: The U-Net level at which to start using attention.
            num_conditions: The number of conditional inputs (excluding time).
            dropout: The dropout rate.
            padding_mode: The padding mode for all convolutions.
        """
        super().__init__()
        self.emb_dim = model_channels
        self.num_conditions = num_conditions

        # --- Embedding Layers ---
        self.time_embedder = nn.Sequential(SinusoidalPosEmb(self.emb_dim, factor=1000), MLPEmbedder(self.emb_dim))
        if num_conditions > 0:
            self.cond_embedders = nn.ModuleList([
                nn.Sequential(SinusoidalPosEmb(self.emb_dim), MLPEmbedder(self.emb_dim))
                for _ in range(num_conditions)
            ])

        # --- Network Architecture ---
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1, padding_mode=padding_mode)
        block_args = {"emb_dim": self.emb_dim, "num_heads": num_heads, "dropout": dropout, "padding_mode": padding_mode}
        ch_schedule = [model_channels] + [model_channels * m for m in channel_mults]

        # 1. Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(channel_mults)):
            level_modules = nn.ModuleList()
            ch_in, ch_out = ch_schedule[i], ch_schedule[i+1]
            for _ in range(num_blocks):
                level_modules.append(FluxlikeResnetBlock(channels=ch_in, use_attention=(i >= start_attn_level), **block_args))
            level_modules.append(Downsample(in_channels=ch_in, out_channels=ch_out, padding_mode=padding_mode))
            self.down_blocks.append(level_modules)

        # 2. Bottleneck
        bottleneck_ch = ch_schedule[-1]
        self.middle_blocks = nn.ModuleList([
            FluxlikeResnetBlock(channels=bottleneck_ch, use_attention=True, **block_args),
            FluxlikeResnetBlock(channels=bottleneck_ch, use_attention=True, **block_args),
        ])

        # 3. Decoder
        self.up_blocks = nn.ModuleList()
        ch_schedule_rev = list(reversed(ch_schedule))
        for i in range(len(channel_mults)):
            ch_from_below = ch_schedule_rev[i]
            ch_from_skip = ch_schedule_rev[i+1]
            ch_out_level = ch_schedule_rev[i+1]

            level_modules = nn.ModuleDict()
            level_modules['upsample'] = Upsample(in_channels=ch_from_below, out_channels=ch_out_level, padding_mode=padding_mode)
            level_modules['skip_proj'] = nn.Conv2d(ch_out_level + ch_from_skip, ch_out_level, 1)
            level_blocks = nn.ModuleList([
                FluxlikeResnetBlock(
                    channels=ch_out_level,
                    use_attention=(len(channel_mults) - 1 - i >= start_attn_level),
                    **block_args)
                for _ in range(num_blocks)
            ])
            level_modules['blocks'] = level_blocks
            self.up_blocks.append(level_modules)

        # 4. Final Output
        self.conv_out = nn.Sequential(
            RMSNorm2d(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1, padding_mode=padding_mode)
        )

    def forward(self, x: Tensor, time: Tensor, conditions: Optional[Sequence[Tensor]] = None) -> Tensor:
        """Defines the forward pass of the U-Net.

        Args:
            x: The input tensor of shape (B, C, H, W).
            time: A tensor of time steps, shape (B,).
            conditions: An optional sequence of conditional tensors, each of shape (B,).

        Returns:
            The output tensor of shape (B, C_out, H, W).
        """
        # 1. Prepare conditioning vectors
        t_emb = self.time_embedder(time.to(x.device))
        cond_seq_list, final_emb = [t_emb], t_emb.clone()
        if self.num_conditions > 0 and conditions is not None:
            for i, cond in enumerate(conditions):
                cond_emb = self.cond_embedders[i](cond.to(x.device))
                cond_seq_list.append(cond_emb)
                final_emb += cond_emb
        cond_seq = torch.stack(cond_seq_list, dim=1)

        # 2. U-Net Forward Pass
        h = self.conv_in(x)
        skips = [h]

        # === Encoder ===
        for level_modules in self.down_blocks:
            res_blocks, downsampler = level_modules[:-1], level_modules[-1]
            for block in res_blocks:
                h = block(h, cond_seq, final_emb)
            skips.append(h)
            h = downsampler(h, cond_seq, final_emb)

        # === Bottleneck ===
        for block in self.middle_blocks:
            h = block(h, cond_seq, final_emb)

        # === Decoder ===
        for level_modules in self.up_blocks:
            skip_h = skips.pop()
            h = level_modules['upsample'](h, cond_seq, final_emb)
            h = torch.cat([h, skip_h], dim=1)
            h = level_modules['skip_proj'](h)
            for block in level_modules['blocks']:
                h = block(h, cond_seq, final_emb)

        return self.conv_out(h)

# ==============================================================================
# Section 6: Testing Code
# ==============================================================================

class TestFluxUNetComponents(unittest.TestCase):
    """Test individual components of the FluxUNet architecture."""

    def setUp(self):
        """Set up common parameters and tensors for tests."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.channels = 16
        self.height = 32
        self.width = 32
        self.emb_dim = 64
        self.seq_len = 10
        self.head_dim = 8
        self.num_heads = self.channels // self.head_dim

        self.x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        self.vec = torch.randn(self.batch_size, self.emb_dim).to(self.device)
        self.cond_seq = torch.randn(self.batch_size, self.seq_len, self.emb_dim).to(self.device)

    def test_rotate_half(self):
        """Test the rotate_half helper for RoPE."""
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
        expected = torch.tensor([[-2., 1., -4., 3.], [-6., 5., -8., 7.]], dtype=torch.float32)
        output = rotate_half(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.allclose(output, expected))

    def test_modulate(self):
        """Test the Ada-LayerNorm modulation function."""
        shift = torch.randn(self.batch_size, self.channels).to(self.device)
        scale = torch.randn(self.batch_size, self.channels).to(self.device)
        output = modulate(self.x, shift, scale)
        self.assertEqual(output.shape, self.x.shape)

    def test_rmsnorm2d(self):
        """Test the RMSNorm2d layer."""
        norm = RMSNorm2d(self.channels).to(self.device)
        output = norm(self.x)
        self.assertEqual(output.shape, self.x.shape)

    def test_qknorm(self):
        """Test the QKNorm layer."""
        q = torch.randn(self.batch_size, self.seq_len, self.head_dim).to(self.device)
        k = torch.randn(self.batch_size, self.seq_len, self.head_dim).to(self.device)
        norm = QKNorm(self.head_dim).to(self.device)
        q_norm, k_norm = norm(q, k)
        self.assertEqual(q_norm.shape, q.shape)
        self.assertEqual(k_norm.shape, k.shape)

    def test_modulation_module(self):
        """Test the Modulation module for generating shift/scale/gate."""
        mod_single = Modulation(self.emb_dim, self.channels, is_double=False).to(self.device)
        mod1, mod2 = mod_single(self.vec)
        self.assertIsInstance(mod1, ModulationOut)
        self.assertIsNone(mod2)
        self.assertEqual(mod1.shift.shape, (self.batch_size, self.channels))

        mod_double = Modulation(self.emb_dim, self.channels, is_double=True).to(self.device)
        mod1_d, mod2_d = mod_double(self.vec)
        self.assertIsInstance(mod1_d, ModulationOut)
        self.assertIsInstance(mod2_d, ModulationOut)
        self.assertEqual(mod2_d.shift.shape, (self.batch_size, self.channels))

    def test_mlp(self):
        """Test the spatial MLP block."""
        mlp = Mlp(self.channels).to(self.device)
        output = mlp(self.x)
        self.assertEqual(output.shape, self.x.shape)

    def test_sinusoidal_pos_emb(self):
        """Test the SinusoidalPosEmb module."""
        emb = SinusoidalPosEmb(self.emb_dim).to(self.device)
        t = torch.randint(0, 100, (self.batch_size,)).to(self.device)
        output = emb(t)
        self.assertEqual(output.shape, (self.batch_size, self.emb_dim))

    def test_rope_and_rope_mixed(self):
        """Test both 1D and 2D RoPE modules."""
        rope1d = RoPE(dim=self.head_dim).to(self.device)
        seq = torch.randn(self.batch_size, self.seq_len, self.head_dim).to(self.device)
        output1d = rope1d(seq)
        self.assertEqual(output1d.shape, seq.shape)

        rope2d = RoPE_Mixed(dim=self.head_dim).to(self.device)
        spatial_seq = torch.randn(self.batch_size, self.height, self.width, self.head_dim).to(self.device)
        output2d = rope2d(spatial_seq)
        self.assertEqual(output2d.shape, spatial_seq.shape)

    def test_fluxlike_resnet_block(self):
        """Test the FluxlikeResnetBlock in both attention and non-attention modes."""
        block_args = {
            "channels": self.channels, "emb_dim": self.emb_dim, "num_heads": self.num_heads,
            "dropout": 0.1, "padding_mode": "replicate"
        }
        block_no_attn = FluxlikeResnetBlock(use_attention=False, **block_args).to(self.device)
        output_no_attn = block_no_attn(self.x, self.cond_seq, self.vec)
        self.assertEqual(output_no_attn.shape, self.x.shape)

        block_attn = FluxlikeResnetBlock(use_attention=True, **block_args).to(self.device)
        output_attn = block_attn(self.x, self.cond_seq, self.vec)
        self.assertEqual(output_attn.shape, self.x.shape)

    def test_downsample_upsample(self):
        """Test the Downsample and Upsample layers."""
        out_channels = self.channels * 2
        downsampler = Downsample(self.channels, out_channels, "replicate").to(self.device)
        h_down = downsampler(self.x)
        self.assertEqual(h_down.shape, (self.batch_size, out_channels, self.height // 2, self.width // 2))

        upsampler = Upsample(out_channels, self.channels, "replicate").to(self.device)
        h_up = upsampler(h_down)
        self.assertEqual(h_up.shape, self.x.shape)


class TestFluxUNetEndToEnd(unittest.TestCase):
    """Perform end-to-end tests on the complete FluxUNet model."""

    def setUp(self):
        """Set up the model and input data for end-to-end tests."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.image_size = 32
        self.in_channels = 4
        self.model_channels = 32
        self.num_conditions = 2

        self.model_config = {
            "in_channels": self.in_channels, "out_channels": self.in_channels,
            "model_channels": self.model_channels, "channel_mults": (1, 2),
            "num_blocks": 1, "num_heads": 4, "start_attn_level": 0,
            "num_conditions": self.num_conditions
        }
        self.x = torch.randn(self.batch_size, self.in_channels, self.image_size, self.image_size, device=self.device)
        self.time = torch.randint(0, 1000, (self.batch_size,), device=self.device)
        self.conditions = [
            torch.randint(0, 10, (self.batch_size,), device=self.device),
            torch.randn(self.batch_size, device=self.device)
        ]

    def test_forward_pass_shape(self):
        """Test the forward pass with conditions and validate output shape."""
        model = FluxUNet(**self.model_config).to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(self.x, self.time, self.conditions)
        self.assertEqual(output.shape, self.x.shape)

    def test_forward_pass_no_conditions(self):
        """Test forward pass when num_conditions is 0."""
        config = self.model_config.copy()
        config["num_conditions"] = 0
        model = FluxUNet(**config).to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(self.x, self.time, None)
        self.assertEqual(output.shape, self.x.shape)

    def test_torch_compile(self):
        """Test if the model can be compiled with torch.compile."""
        model = FluxUNet(**self.model_config).to(self.device)

        try:
            with torch.no_grad():
                output = model(self.x, self.time, self.conditions)
            self.assertEqual(output.shape, self.x.shape)
        except Exception as e:
            self.fail(f"torch.compile failed with an exception: {e}")


if __name__ == '__main__':
    print("🚀 Running Formal Unittest Suite for FluxUNet...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)