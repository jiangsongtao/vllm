# SPDX-License-Identifier: Apache-2.0
# Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py.
# Below is the original copyright:
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
HuluMed Vision Encoder for vLLM - strictly aligned with transformers version.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


# ============================================================================
# Weight Initialization Functions (from transformers)
# ============================================================================

def _trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        logger.warning(
            "mean is more than 2 std from [a, b] in _trunc_normal_. "
            "The distribution of values may be incorrect."
        )

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    """Variance scaling initialization."""
    if mode not in ["fan_in", "fan_out", "fan_avg"]:
        raise ValueError(f"Invalid mode {mode}")

    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        _trunc_normal_(tensor, mean=0.0, std=math.sqrt(variance), a=-2.0, b=2.0)
    elif distribution == "normal":
        nn.init.normal_(tensor, mean=0.0, std=math.sqrt(variance))
    elif distribution == "uniform":
        nn.init.uniform_(tensor, a=-math.sqrt(3 * variance), b=math.sqrt(3 * variance))
    else:
        raise ValueError(f"Invalid distribution {distribution}")


def default_flax_embed_init(tensor: torch.Tensor) -> None:
    """Default Flax embedding initialization."""
    variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal")


def lecun_normal_(tensor: torch.Tensor) -> None:
    """LeCun normal initialization."""
    variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="truncated_normal")


# ============================================================================
# Rotary Position Embedding
# ============================================================================

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to vision features."""
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    """Rotary position embedding for vision."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


# ============================================================================
# Vision Embeddings
# ============================================================================

class HulumedVisionEmbeddings(nn.Module):
    """Vision embeddings using Conv2d patch embedding."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_patches, num_channels * patch_size * patch_size]
        Returns:
            embeddings: [num_patches, embed_dim]
        """
        # Reshape to [num_patches, num_channels, patch_size, patch_size]
        hidden_states = hidden_states.view(
            -1, self.config.num_channels, self.patch_size, self.patch_size
        )
        # Apply conv: [num_patches, embed_dim, 1, 1]
        patch_embeds = self.patch_embedding(hidden_states)
        # Flatten: [num_patches, embed_dim]
        embeddings = patch_embeds.view(-1, self.embed_dim)
        return embeddings


# ============================================================================
# Attention Modules
# ============================================================================

class VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} "
                f"and `num_heads`: {self.num_heads})."
            )
        
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [seq_len, embed_dim]
            cu_seqlens: cumulative sequence lengths
            rotary_pos_emb: rotary position embeddings
        """
        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_heads, self.head_dim)

        # Apply rotary position embedding
        query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # Build attention mask
        attention_mask = torch.zeros([1, q_len, q_len], device=query_states.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        query_states = query_states.transpose(0, 1)  # [num_heads, seq_len, head_dim]
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)

        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(0, 1)  # [seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(q_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output


class VisionFlashAttention2(VisionAttention):
    """Flash Attention 2 implementation."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        if flash_attn_varlen_func is None:
            raise ImportError(
                "Flash attention 2 is not installed. "
                "Install with: pip install flash-attn --no-build-isolation"
            )

        q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_heads, self.head_dim)
        
        query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
        ).reshape(q_len, -1)

        attn_output = self.out_proj(attn_output)
        return attn_output


class VisionSdpaAttention(VisionAttention):
    """SDPA attention implementation."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(seq_length, self.num_heads, self.head_dim)
        key_states = key_states.view(seq_length, self.num_heads, self.head_dim)
        value_states = value_states.view(seq_length, self.num_heads, self.head_dim)

        query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # Build attention mask
        attention_mask = torch.zeros([1, seq_length, seq_length], device=query_states.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        query_states = query_states.transpose(0, 1)  # [num_heads, seq_len, head_dim]
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        
        attn_output = F.scaled_attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )

        attn_output = attn_output.transpose(0, 1)  # [seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


VISION_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
}


# ============================================================================
# MLP Module
# ============================================================================

class HulumedVisionMLP(nn.Module):
    """MLP module for vision transformer."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# ============================================================================
# Encoder Layer
# ============================================================================

class HulumedVisionEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        
        # Determine attention implementation
        attn_implementation = getattr(config, '_attn_implementation', 'sdpa')
        if attn_implementation is None:
            attn_implementation = 'sdpa'
        
        if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
            logger.warning(
                "Flash attention 2 is not available, falling back to SDPA. "
                "Install with: pip install flash-attn --no-build-isolation"
            )
            attn_implementation = "sdpa"
        
        self.self_attn = VISION_ATTENTION_CLASSES[attn_implementation](config, quant_config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = HulumedVisionMLP(config, quant_config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Self attention with residual
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb
        )
        # MLP with residual
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


# ============================================================================
# Transformer Encoder
# ============================================================================

class HulumedVisionTransformerEncoder(nn.Module):
    """Vision transformer encoder."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.layers = nn.ModuleList([
            HulumedVisionEncoderLayer(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_sizes: torch.Tensor, merge_sizes: torch.Tensor) -> torch.Tensor:
        """
        Generate rotary position embeddings for vision patches.
        
        Args:
            grid_sizes: [num_images, 3] where each row is (t, h, w)
            merge_sizes: [num_images] merge size for each image
        
        Returns:
            rotary_pos_emb: [total_patches, head_dim]
        """
        pos_ids = []
        
        # Ensure proper tensor shapes
        if not isinstance(grid_sizes, torch.Tensor):
            grid_sizes = torch.tensor(grid_sizes, device=self.rotary_pos_emb.inv_freq.device)
        
        if grid_sizes.ndim == 1:
            grid_sizes = grid_sizes.unsqueeze(0)
        
        if merge_sizes is None:
            merge_sizes = torch.ones(grid_sizes.shape[0], dtype=torch.long, device=grid_sizes.device)
        elif not isinstance(merge_sizes, torch.Tensor):
            merge_sizes = torch.tensor(merge_sizes, device=grid_sizes.device)
        
        if merge_sizes.ndim == 0:
            merge_sizes = merge_sizes.unsqueeze(0)
        
        max_grid_size = 0
        
        # Generate position IDs for each image
        for i in range(grid_sizes.shape[0]):
            thw = grid_sizes[i]
            if thw.ndim > 1:
                thw = thw.squeeze()
            
            t, h, w = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
            
            max_grid_size = max(max_grid_size, h, w)
            
            merge_size_val = merge_sizes[i]
            if merge_size_val.ndim > 0:
                merge_size_val = merge_size_val.squeeze()
            merge_size = int(merge_size_val.item())
            
            # Generate height position IDs
            hpos_ids = torch.arange(h, device=grid_sizes.device).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // merge_size, merge_size, w // merge_size, merge_size
            ).permute(0, 2, 1, 3).flatten()
            
            # Generate width position IDs
            wpos_ids = torch.arange(w, device=grid_sizes.device).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // merge_size, merge_size, w // merge_size, merge_size
            ).permute(0, 2, 1, 3).flatten()
            
            # Stack and repeat for temporal dimension
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        
        pos_ids = torch.cat(pos_ids, dim=0)
        
        # Generate full rotary embeddings
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        
        return rotary_pos_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_sizes: torch.Tensor,
        merge_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [total_patches, hidden_size]
            grid_sizes: [num_images, 3]
            merge_sizes: [num_images]
        """
        # Generate rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_sizes, merge_sizes)
        
        # Calculate cumulative sequence lengths for attention masking
        cu_seqlens = torch.repeat_interleave(
            grid_sizes[:, 1] * grid_sizes[:, 2], grid_sizes[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Apply transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                )

        return hidden_states


# ============================================================================
# Main Vision Encoder
# ============================================================================

class HulumedVisionEncoder(nn.Module):
    """
    HuluMed Vision Encoder - fully aligned with transformers implementation.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.prefix = prefix
        
        embed_dim = config.hidden_size
        
        self.embeddings = HulumedVisionEmbeddings(config, quant_config)
        self.encoder = HulumedVisionTransformerEncoder(config, quant_config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @property
    def device(self):
        """Get device of the model."""
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_sizes: torch.Tensor,
        merge_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of vision encoder.
        
        Args:
            pixel_values: [total_patches, channels * patch_size * patch_size]
            grid_sizes: [num_images, 3] where each row is (t, h, w)
            merge_sizes: [num_images] spatial merge size for each image
        
        Returns:
            vision_features: [total_output_patches, hidden_size]
        """
        # 1. Embed patches
        hidden_states = self.embeddings(pixel_values)
        
        # 2. Ensure proper tensor shapes
        if not isinstance(grid_sizes, torch.Tensor):
            grid_sizes = torch.tensor(grid_sizes, device=hidden_states.device)
        if grid_sizes.ndim == 1:
            grid_sizes = grid_sizes.unsqueeze(0)
        
        if merge_sizes is None:
            merge_sizes = torch.ones(grid_sizes.shape[0], dtype=torch.long, device=hidden_states.device)
        elif not isinstance(merge_sizes, torch.Tensor):
            merge_sizes = torch.tensor(merge_sizes, device=hidden_states.device)
        if merge_sizes.ndim == 0:
            merge_sizes = merge_sizes.unsqueeze(0)
        
        # 3. Apply transformer encoder
        hidden_states = self.encoder(hidden_states, grid_sizes, merge_sizes)
        
        # 4. Apply post layer norm
        hidden_states = self.post_layernorm(hidden_states)
        
        # 5. Post-process: spatial pooling and interpolation
        # Split hidden states by image
        split_sizes = grid_sizes.prod(dim=1).tolist()
        hidden_states_chunks = hidden_states.split(split_sizes, dim=0)
        
        outputs = []
        for hs, grid_size, merge_size in zip(hidden_states_chunks, grid_sizes, merge_sizes):
            c = hs.shape[-1]
            
            # Extract t, h, w
            if isinstance(grid_size, torch.Tensor):
                t = int(grid_size[0].item())
                h = int(grid_size[1].item())
                w = int(grid_size[2].item())
            else:
                t, h, w = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
            
            # Extract merge_size value
            if isinstance(merge_size, torch.Tensor):
                ms = int(merge_size.item())
            else:
                ms = int(merge_size)
            
            # Reshape and pool spatial dimensions
            # [t*h*w, c] -> [t, h//ms, w//ms, ms, ms, c]
            hs = hs.view(t, h // ms, w // ms, ms, ms, c)
            # Permute to [t, h//ms, ms, w//ms, ms, c]
            hs = hs.permute(0, 1, 3, 2, 4, 5)
            # Reshape to [t, h, w, c]
            hs = hs.reshape(t, h, w, c)
            # Permute to [t, c, h, w] for interpolation
            hs = hs.permute(0, 3, 1, 2)
            
            # Bilinear interpolation to target size
            hs = F.interpolate(
                hs,
                size=(h // ms, w // ms),
                mode='bilinear',
                align_corners=False
            )
            
            # Permute back to [t, h//ms, w//ms, c] and flatten
            hs = hs.permute(0, 2, 3, 1).reshape(-1, c)
            
            outputs.append(hs)
        
        return torch.cat(outputs, dim=0)

    def _init_weights(self, module):
        """Initialize weights following transformers convention."""
        if isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, (VisionAttention, VisionFlashAttention2, VisionSdpaAttention)):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, HulumedVisionMLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

