import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence, Tuple
from dataclasses import dataclass

torch.set_float32_matmul_precision('high')

# -----------------------------------------------------------------------------
# 核心理念：這是一個為擴散模型設計的U-Net，其架構經過重構，以融入
# Flux Transformer 的多項核心設計理念。它在保持卷積U-Net骨幹的同時，
# 引入了如統一條件向量、模塊化調製、雙流融合注意力、QK正規化以及
# 旋轉位置編碼(RoPE)等先進特性。
# -----------------------------------------------------------------------------


# ==============================================================================
# Section 1: 基礎輔助模塊 (Basic Helper Modules)
# ==============================================================================

@dataclass
class ModulationOut:
    """清晰地存儲調製參數"""
    shift: Tensor
    scale: Tensor
    gate: Tensor

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """應用 adaLN 調製"""
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE 的核心輔助函數，旋轉一半的維度"""
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x_rotated_pairs = torch.cat([-x_reshaped[..., 1:], x_reshaped[..., :1]], dim=-1)
    return x_rotated_pairs.flatten(start_dim=-2)

class RMSNorm2d(nn.Module):
    """
    均方根層正規化，修正後可正確處理 4D 卷積特徵圖 (B, C, H, W)。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # scale 的維度是通道數 C
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x 的 shape 是 (B, C, H, W)
        # 我們希望在 channel (C) 維度上進行正規化
        # keepdim=True 讓 rrms 的 shape 為 (B, 1, H, W) 以便廣播
        rrms = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)

        # self.scale 的 shape 是 (C,)
        # 為了與 (B, C, H, W) 的張量相乘，需要 reshape 成 (1, C, 1, 1)
        scale_reshaped = self.scale.view(1, -1, 1, 1)

        # (B,C,H,W) * (B,1,H,W) * (1,C,1,1) -> (B,C,H,W)
        return x * rrms * scale_reshaped

class QKNorm(nn.Module):
    """專門對 Query 和 Key 進行正規化的模塊"""
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim)
        self.key_norm = nn.RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q), self.key_norm(k)

class Modulation(nn.Module):
    """從統一的條件向量 vec 生成調製參數"""
    def __init__(self, emb_dim: int, out_channels: int, is_double: bool = False):
        super().__init__()
        self.multiplier = 6 if is_double else 3
        self.lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, self.multiplier * out_channels, bias=True)
        )
        nn.init.zeros_(self.lin[-1].weight)
        nn.init.zeros_(self.lin[-1].bias)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, Optional[ModulationOut]]:
        params = self.lin(vec)
        chunks = params.chunk(self.multiplier, dim=-1)
        mod1 = ModulationOut(shift=chunks[0], scale=chunks[1], gate=chunks[2])
        mod2 = ModulationOut(shift=chunks[3], scale=chunks[4], gate=chunks[5]) if self.multiplier == 6 else None
        return mod1, mod2

class Mlp(nn.Module):
    """空間特徵的 MLP (使用1x1卷積實現)"""
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None, act_layer: nn.Module = nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden_channels = hidden_channels or in_channels * 4
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = act_layer()
        self.pw_conv2 = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.pw_conv2(self.act(self.pw_conv1(x))))


# ==============================================================================
# Section 2: 嵌入輔助模塊 (Embedding Helpers)
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """正弦位置嵌入"""
    def __init__(self, embedding_dim: int, max_period: int = 10000, factor: float = 100.0):
        super().__init__()
        self.embedding_dim = embedding_dim if embedding_dim % 2 == 0 else embedding_dim + 1
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer('freqs', freqs)
        self.factor = factor

    def forward(self, x: Tensor) -> Tensor:
        args = x.float().unsqueeze(1) * self.freqs.unsqueeze(0) * self.factor
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class MLPEmbedder(nn.Module):
    """用於嵌入向量的MLP"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


# ==============================================================================
# Section 3: 旋轉位置編碼 (RoPE Modules)
# ==============================================================================

class RoPE(nn.Module):
    """標準一維 RoPE，用於條件序列"""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"
        freqs = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (base**freqs)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        device = x.device
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return x * cos + rotate_half(x) * sin

class RoPE_Mixed(nn.Module):
    """2D RoPE，用於空間特徵圖"""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"
        self.freqs = nn.Parameter(torch.randn(dim // 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W, D = x.shape[-3:]
        device = x.device
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        positions_2d = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)

        theta_y = self.freqs[:, 0]
        theta_x = self.freqs[:, 1]
        angles = torch.einsum("n,d->nd", positions_2d[:, 0], theta_y) + torch.einsum("n,d->nd", positions_2d[:, 1], theta_x)

        cos_vals = angles.cos().repeat_interleave(2, dim=-1).reshape(H, W, D)
        sin_vals = angles.sin().repeat_interleave(2, dim=-1).reshape(H, W, D)

        while cos_vals.dim() < x.dim():
            cos_vals = cos_vals.unsqueeze(0)
            sin_vals = sin_vals.unsqueeze(0)

        return x * cos_vals + rotate_half(x) * sin_vals


# ==============================================================================
# Section 4: 核心 U-Net 構建塊 (Core U-Net Blocks)
# ==============================================================================

class FusedAttentionBlock(nn.Module):
    """雙流融合注意力模塊 (修正版，支持不同維度)"""
    def __init__(self, channels: int, emb_dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        # 移除 assert channels == emb_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels 必須能被 num_heads 整除"

        self.to_qkv_spatial = nn.Conv2d(channels, channels * 3, 1, bias=qkv_bias)

        # --- 【核心修正】---
        # 讓條件流的 QKV 投射輸出維度與空間流對齊 (channels * 3)
        # 這樣它們才能被切分成相同 head_dim 的注意力頭
        self.to_qkv_cond = nn.Linear(emb_dim, channels * 3, bias=qkv_bias)

        # RoPE 和 QKNorm 的維度都是 head_dim，這部分不需要改變
        self.rope_spatial = RoPE_Mixed(self.head_dim)
        self.rope_conditional = RoPE(self.head_dim)
        self.qknorm = QKNorm(self.head_dim)

        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: Tensor, cond_seq: Tensor) -> Tensor:
        # forward 方法的內部邏輯完全不需要改變，因為維度已經在 __init__ 中對齊了
        B, C, H, W = x.shape
        S = cond_seq.shape[1]

        q_s_raw, k_s_raw, v_s_raw = self.to_qkv_spatial(x).chunk(3, dim=1)
        qkv_c_raw = self.to_qkv_cond(cond_seq).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_c_raw, k_c_raw, v_c = qkv_c_raw[0], qkv_c_raw[1], qkv_c_raw[2]

        q_s_for_rope = q_s_raw.view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
        k_s_for_rope = k_s_raw.view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
        q_s_rot = self.rope_spatial(q_s_for_rope).permute(0, 1, 4, 2, 3).reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)
        k_s_rot = self.rope_spatial(k_s_for_rope).permute(0, 1, 4, 2, 3).reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)

        q_c_rot = self.rope_conditional(q_c_raw.reshape(B * self.num_heads, S, self.head_dim)).view(B, self.num_heads, S, self.head_dim)
        k_c_rot = self.rope_conditional(k_c_raw.reshape(B * self.num_heads, S, self.head_dim)).view(B, self.num_heads, S, self.head_dim)

        q = torch.cat((q_s_rot, q_c_rot), dim=2)
        k = torch.cat((k_s_rot, k_c_rot), dim=2)
        v_s = v_s_raw.view(B, self.num_heads, self.head_dim, H*W).transpose(-1,-2)
        v = torch.cat((v_s, v_c), dim=2)

        q, k = self.qknorm(q, k)

        attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_spatial_output = attn_output[:, :, :H*W, :].transpose(1, 2).reshape(B, H * W, C)
        out = self.to_out(attn_spatial_output).permute(0, 2, 1).view(B, C, H, W)

        return out

class FluxlikeResnetBlock(nn.Module):
    """模仿 Flux/DiT 設計的 ResNet 塊 (最終版)"""
    def __init__(self, channels: int, emb_dim: int, num_heads: int, use_attention: bool, dropout: float, padding_mode: str):
        super().__init__()
        self.use_attention = use_attention
        self.modulation = Modulation(emb_dim, channels, is_double=True)
        self.norm1 = RMSNorm2d(channels)
        self.norm2 = RMSNorm2d(channels)

        if self.use_attention:
            self.op1 = FusedAttentionBlock(channels, emb_dim, num_heads)
        else:
            self.op1 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode=padding_mode, groups=channels)

        self.mlp = Mlp(channels, act_layer=lambda: nn.GELU(approximate="tanh"), drop=dropout)

    def forward(self, x: Tensor, cond_seq: Tensor, final_emb: Tensor) -> Tensor:
        mod1, mod2 = self.modulation(final_emb)
        h_mod = modulate(self.norm1(x), mod1.shift, mod1.scale)

        if self.use_attention:
            h_op1 = self.op1(h_mod, cond_seq)
        else:
            h_op1 = self.op1(h_mod)

        x = x + mod1.gate.unsqueeze(-1).unsqueeze(-1) * h_op1
        x = x + mod2.gate.unsqueeze(-1).unsqueeze(-1) * self.mlp(modulate(self.norm2(x), mod2.shift, mod2.scale))
        return x

class Downsample(nn.Module):
    """下採樣：同時縮小尺寸並改變通道數"""
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str):
        super().__init__()
        # 使用一個卷積層同時完成下採樣和通道變換
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=0)
        self.padding_mode = padding_mode

    def forward(self, x: Tensor, *args) -> Tensor: # 增加 *args 以接收多餘參數
        pad = (0, 1, 0, 1)
        pad_x = nn.functional.pad(x, pad, mode=self.padding_mode)
        return self.conv(pad_x)

class Upsample(nn.Module):
    """上採樣：同時放大尺寸並改變通道數"""
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str):
        super().__init__()
        # 先放大尺寸，再用卷積改變通道數
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x: Tensor, *args) -> Tensor: # 增加 *args
        return self.conv(self.upsample(x))


# ==============================================================================
# Section 5: 主模型 (Main U-Net Model)
# ==============================================================================
@torch.compile
class FluxUNet(nn.Module):
    """最終的 Flux-Style 卷積 U-Net (Decoder 和 Encoder 通道流修正版)"""
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
        super().__init__()
        self.emb_dim = model_channels
        self.num_conditions = num_conditions

        # --- 嵌入層 ---
        self.time_embedder = nn.Sequential(SinusoidalPosEmb(self.emb_dim, factor=1000), MLPEmbedder(self.emb_dim))
        if num_conditions > 0:
            self.cond_embedders = nn.ModuleList([
                nn.Sequential(SinusoidalPosEmb(self.emb_dim), MLPEmbedder(self.emb_dim))
                for _ in range(num_conditions)
            ])

        # --- 網絡結構 ---
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1, padding_mode=padding_mode)
        block_args = {"emb_dim": self.emb_dim, "num_heads": num_heads, "dropout": dropout, "padding_mode": padding_mode}

        # 1. Encoder
        self.down_blocks = nn.ModuleList()
        ch_schedule = [model_channels] + [model_channels * m for m in channel_mults]

        for i in range(len(channel_mults)):
            level_modules = nn.ModuleList()
            ch_in = ch_schedule[i]
            ch_out = ch_schedule[i+1]
            for _ in range(num_blocks):
                level_modules.append(FluxlikeResnetBlock(channels=ch_in, use_attention=(i >= start_attn_level), **block_args))

            # 使用修正後的 Downsample，正確地從 ch_in 轉換到 ch_out
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
            # Upsample 從 ch_from_below 轉換到 ch_out_level
            level_modules['upsample'] = Upsample(in_channels=ch_from_below, out_channels=ch_out_level, padding_mode=padding_mode)
            # skip_proj 合併後的通道數是 ch_out_level + ch_from_skip
            level_modules['skip_proj'] = nn.Conv2d(ch_out_level + ch_from_skip, ch_out_level, 1)

            level_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                level_blocks.append(FluxlikeResnetBlock(channels=ch_out_level, use_attention=(len(channel_mults) - 1 - i >= start_attn_level), **block_args))
            level_modules['blocks'] = level_blocks
            self.up_blocks.append(level_modules)

        # 4. Final Output
        self.conv_out = nn.Sequential(
            RMSNorm2d(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1, padding_mode=padding_mode)
        )

    def forward(self, x: Tensor, time: Tensor, conditions: Optional[Sequence[Tensor]] = None) -> Tensor:
        # 1. 準備條件向量 (這部分邏輯是正確的)
        t_emb = self.time_embedder(time.to(x.device))
        cond_seq_list, final_emb = [t_emb], t_emb.clone()
        if self.num_conditions > 0:
            for i, cond in enumerate(conditions):
                cond_emb = self.cond_embedders[i](cond.to(x.device))
                cond_seq_list.append(cond_emb)
                final_emb += cond_emb
        cond_seq = torch.stack(cond_seq_list, dim=1)

        # --- 2. U-Net 前向傳播 (修正部分) ---
        h = self.conv_in(x)
        skips = [h]

        # === Encoder ===
        # 遍歷每一層的模塊列表
        for level_modules in self.down_blocks:
            # level_modules 是一個 ModuleList，最後一個元素是 Downsample
            res_blocks = level_modules[:-1]
            downsampler = level_modules[-1]

            # 先讓 h 流過這一層所有的 ResBlock
            for block in res_blocks:
                h = block(h, cond_seq, final_emb)

            # 【核心修正】在下採樣之前，保存跳躍連接
            skips.append(h)

            # 最後執行下採樣
            h = downsampler(h, cond_seq, final_emb)

        # === Bottleneck ===
        for block in self.middle_blocks:
            h = block(h, cond_seq, final_emb)

        # === Decoder ===
        # 現在 Decoder 的邏輯是完全正確的了
        for level_modules in self.up_blocks:
            # 從 skips 列表的末尾取出對應層級的、尺寸正確的跳躍連接
            skip_h = skips.pop()

            # 上採樣來自下一層的 h
            h = level_modules['upsample'](h, cond_seq, final_emb)

            # 現在 h 和 skip_h 的空間尺寸應該完全匹配
            h = torch.cat([h, skip_h], dim=1)

            # 投影合併通道
            h = level_modules['skip_proj'](h)

            # 流過這一層的 ResBlock
            for block in level_modules['blocks']:
                h = block(h, cond_seq, final_emb)

        return self.conv_out(h)

# ==============================================================================
# Section 6: 測試代碼 (Testing Code)
# ==============================================================================

if __name__ == '__main__':
    # --- 1. 參數設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    image_size = 32  # 必須是 2^num_levels 的倍數
    in_channels = 3
    out_channels = 3
    model_channels = 64
    channel_mults = (1, 2, 2)  # 3 個 U-Net 層級
    num_blocks = 2
    num_heads = 4
    start_attn_level = 1 # 第 0 層 (最外層) 不用 attention，第 1, 2 層用
    num_conditions = 1 # 1 個額外條件

    # --- 2. 創建模型 ---
    print("🚀 正在創建 Flux-Style U-Net 模型...")
    model = FluxUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        channel_mults=channel_mults,
        num_blocks=num_blocks,
        num_heads=num_heads,
        start_attn_level=start_attn_level,
        num_conditions=num_conditions
    ).to(device)

    # 打印模型參數數量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型創建成功！")
    print(f"   - 總參數數量: {num_params / 1e6:.2f} M")
    print(f"   - 運行設備: {device.type.upper()}")

    # --- 3. 準備虛擬輸入數據 ---
    print("\n📦 正在準備虛擬輸入數據...")

    # 圖像輸入
    x = torch.randn(batch_size, in_channels, image_size, image_size, device=device)

    # 時間步輸入
    time = torch.randint(0, 1000, (batch_size,), device=device)

    # 額外條件輸入 (例如類別標籤、CFG強度等)
    # 這裡用一個隨機值模擬
    cond1 = torch.randint(0, 10, (batch_size,), device=device)
    conditions = [cond1]

    print("   - 圖像輸入 shape:    ", x.shape)
    print("   - 時間步輸入 shape:  ", time.shape)
    print("   - 條件輸入 shape:    ", [c.shape for c in conditions])

    # --- 4. 執行前向傳播測試 ---
    print("\n⚡️ 正在執行前向傳播測試...")
    try:
        with torch.no_grad(): # 測試時不需要計算梯度
            output = model(x, time, conditions)

        print("✅ 前向傳播成功！")
        print(f"   - 輸出 shape: {output.shape}")

        # 檢查輸出 shape 是否與輸入 shape 一致
        assert output.shape == x.shape, "輸出 shape 與輸入 shape 不匹配！"
        print("   - 輸出 shape 驗證成功！")

    except Exception as e:
        print(f"❌ 前向傳播失敗: {e}")
        import traceback
        traceback.print_exc()