"""
Multi-task model with SAM (Segment Anything Model) ViT-B backbone.
Replaces Swin Transformer with SAM's ImageEncoderViT while preserving
all downstream task heads: segmentation, classification, VQA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, Optional, Type
import math
import sys
import os

# Add SAM path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SAM.modeling.image_encoder import ImageEncoderViT, Block, Attention, PatchEmbed
from SAM.modeling.common import LayerNorm2d, MLPBlock
from Classification.models.class_head import classer
from vqa.VisualBertResMLPClassification import VisualBertResMLPClassification


# ============================================================
# Information entropy helper (for Entropy-Driven LoRA Tuning, ELT)
# ============================================================
def calculate_information_entropy(tensor):
    """Compute information entropy of a tensor treated as a probability distribution."""
    flat_tensor = tensor.flatten()
    probs = F.softmax(torch.abs(flat_tensor), dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    return entropy


# ============================================================
# LoRA Linear layer (optionally ELT-aware)
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, use_elt=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.max_rank = rank
        self.current_rank = rank
        self.alpha = alpha
        self.use_elt = use_elt

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scaling = self.alpha / self.rank

    def update_rank(self, new_rank):
        """Dynamically adjust the active LoRA rank (ELT)."""
        if new_rank > self.max_rank:
            new_rank = self.max_rank
        if new_rank <= 0:
            new_rank = 1
        self.current_rank = new_rank
        self.scaling = self.alpha / self.current_rank

    def analyze_rank_by_entropy(self, info_retention_ratio=0.9):
        """Return the smallest rank retaining `info_retention_ratio` of information entropy."""
        with torch.no_grad():
            total_entropy = (
                calculate_information_entropy(self.lora_A[:self.current_rank, :])
                + calculate_information_entropy(self.lora_B[:, :self.current_rank])
            )
            for test_rank in range(self.current_rank - 1, 0, -1):
                test_total_entropy = (
                    calculate_information_entropy(self.lora_A[:test_rank, :])
                    + calculate_information_entropy(self.lora_B[:, :test_rank])
                )
                if test_total_entropy / total_entropy < info_retention_ratio:
                    return test_rank + 1
            return 1

    def forward(self, x):
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_output = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A[:self.current_rank, :]),
            self.lora_B[:, :self.current_rank]
        )
        return lora_output * self.scaling + output


# ============================================================
# SAM Feature Extractor with LoRA
# ============================================================
class SAMFeatureExtractor(nn.Module):
    """
    Wraps SAM's ImageEncoderViT, loads pretrained weights,
    freezes original params, and applies LoRA to attention/MLP layers.
    Returns single-scale features at 1/16 resolution (14x14 for 224 input).
    """
    def __init__(self, sam_encoder, lora_rank=64, lora_alpha=128, use_elt=False):
        super(SAMFeatureExtractor, self).__init__()
        self.base_encoder = sam_encoder
        self.use_elt = use_elt

        # Freeze all original parameters
        for param in self.base_encoder.parameters():
            param.requires_grad = False

        # Apply LoRA to attention qkv, proj and MLP linear layers
        lora_layers = []
        self.lora_modules = []  # module references (for ELT rank adjustment)

        for block_idx, block in enumerate(self.base_encoder.blocks):
            # Attention qkv: Linear(dim, dim*3)
            if hasattr(block.attn, 'qkv'):
                in_features = block.attn.qkv.in_features
                out_features = block.attn.qkv.out_features
                original_weight = block.attn.qkv.weight.data.clone()
                original_bias = block.attn.qkv.bias.data.clone() if block.attn.qkv.bias is not None else None
                block.attn.qkv = LoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    use_elt=use_elt
                )
                with torch.no_grad():
                    block.attn.qkv.weight.copy_(original_weight)
                    if original_bias is not None:
                        block.attn.qkv.bias.copy_(original_bias)
                lora_layers.append(f"blocks[{block_idx}].attn.qkv")
                self.lora_modules.append(block.attn.qkv)

            # Attention proj: Linear(dim, dim)
            if hasattr(block.attn, 'proj'):
                in_features = block.attn.proj.in_features
                out_features = block.attn.proj.out_features
                original_weight = block.attn.proj.weight.data.clone()
                original_bias = block.attn.proj.bias.data.clone() if block.attn.proj.bias is not None else None
                block.attn.proj = LoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    use_elt=use_elt
                )
                with torch.no_grad():
                    block.attn.proj.weight.copy_(original_weight)
                    if original_bias is not None:
                        block.attn.proj.bias.copy_(original_bias)
                lora_layers.append(f"blocks[{block_idx}].attn.proj")
                self.lora_modules.append(block.attn.proj)

            # MLP lin1: Linear(dim, dim*mlp_ratio)
            if hasattr(block.mlp, 'lin1'):
                in_features = block.mlp.lin1.in_features
                out_features = block.mlp.lin1.out_features
                original_weight = block.mlp.lin1.weight.data.clone()
                original_bias = block.mlp.lin1.bias.data.clone() if block.mlp.lin1.bias is not None else None
                block.mlp.lin1 = LoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    use_elt=use_elt
                )
                with torch.no_grad():
                    block.mlp.lin1.weight.copy_(original_weight)
                    if original_bias is not None:
                        block.mlp.lin1.bias.copy_(original_bias)
                lora_layers.append(f"blocks[{block_idx}].mlp.lin1")
                self.lora_modules.append(block.mlp.lin1)

            # MLP lin2: Linear(dim*mlp_ratio, dim)
            if hasattr(block.mlp, 'lin2'):
                in_features = block.mlp.lin2.in_features
                out_features = block.mlp.lin2.out_features
                original_weight = block.mlp.lin2.weight.data.clone()
                original_bias = block.mlp.lin2.bias.data.clone() if block.mlp.lin2.bias is not None else None
                block.mlp.lin2 = LoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    use_elt=use_elt
                )
                with torch.no_grad():
                    block.mlp.lin2.weight.copy_(original_weight)
                    if original_bias is not None:
                        block.mlp.lin2.bias.copy_(original_bias)
                lora_layers.append(f"blocks[{block_idx}].mlp.lin2")
                self.lora_modules.append(block.mlp.lin2)

        print("Layers with LoRA applied:")
        for name in lora_layers:
            print(f"  - {name}")
        print(f"Total number of layers with LoRA: {len(lora_layers)}")

    def update_all_lora_ranks(self, new_rank):
        """Batch-update the active rank of every LoRA layer (ELT)."""
        for lora_layer in self.lora_modules:
            lora_layer.update_rank(new_rank)

    def get_optimal_rank_by_entropy(self, info_retention_ratio=0.9):
        """Compute the median entropy-based optimal rank across all LoRA layers."""
        rank_suggestions = [
            lora_layer.analyze_rank_by_entropy(info_retention_ratio)
            for lora_layer in self.lora_modules
        ]
        suggested_rank = int(torch.median(torch.tensor(rank_suggestions, dtype=torch.float)).item())
        return suggested_rank

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            features: [B, H', W', 768] where H'=W'=H/patch_size
            stage_outputs: list of intermediate features from [B, C, H', W']
        """
        x = self.base_encoder.patch_embed(x)
        if self.base_encoder.pos_embed is not None:
            x = x + self.base_encoder.pos_embed

        stage_outputs = []
        for i, blk in enumerate(self.base_encoder.blocks):
            x = blk(x)
            # Collect features from layers [2, 5, 8, 11] (global attention layers)
            if i in [2, 5, 8, 11]:
                stage_outputs.append(x.permute(0, 3, 1, 2))  # [B, C, H, W]

        # x: [B, H, W, 768] — all layers maintain same resolution
        return x, stage_outputs


# ============================================================
# SAM Decoder — progressive upsampling from 14x14 to 224x224
# ============================================================
class SAMDecoder(nn.Module):
    """
    Decoder designed for SAM's single-scale 14x14 features.
    Uses progressive upsampling: 14→28→56→112→224.
    Skip connections from intermediate ViT layers (all 14x14) are
    resized and concatenated at the corresponding decoder stage.
    """
    def __init__(self, num_classes=2):
        super(SAMDecoder, self).__init__()

        # Stage 1: 14x14 -> 28x28, 768 -> 384
        self.up1 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        # Stage 2: 28x28 -> 56x56, 384 -> 192
        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # Stage 3: 56x56 -> 112x112, 192 -> 96
        self.up3 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        # Stage 4: 112x112 -> 224x224, 96 -> 64
        self.up4 = nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, stage_outputs=None):
        """
        Args:
            x: [B, H, W, 768] — SAM features at 14x14
            stage_outputs: list of [B, 768, 14, 14] from intermediate layers (unused currently)
        Returns:
            [B, num_classes, 224, 224]
        """
        x = x.permute(0, 3, 1, 2)  # [B, 768, 14, 14]

        x = self.up1(x)             # [B, 384, 28, 28]
        x = self.conv1(x)           # [B, 384, 28, 28]

        x = self.up2(x)             # [B, 192, 56, 56]
        x = self.conv2(x)           # [B, 192, 56, 56]

        x = self.up3(x)             # [B, 96, 112, 112]
        x = self.conv3(x)           # [B, 96, 112, 112]

        x = self.up4(x)             # [B, 64, 224, 224]
        x = self.conv4(x)           # [B, 64, 224, 224]

        x = self.final_conv(x)      # [B, num_classes, 224, 224]
        return x


# ============================================================
# Dynamic CBAM Spatial Attention
# ============================================================
class CBAMSpatialAttention(nn.Module):
    """CBAM spatial attention with dynamic spatial dimensions."""
    def __init__(self, kernel_size=7):
        super(CBAMSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, num_heads, HW, HW] — attention probability map (4D)
        Returns:
            [B, num_heads, HW, HW] — refined attention probability map
        """
        B, num_heads, N, _ = x.shape
        H = W = int(math.sqrt(N))  # spatial dim of the feature map

        # ---- First spatial attention (over heads dimension) ----
        max_pool = torch.max(x, dim=1, keepdim=True)[0]   # [B, 1, N, N]
        avg_pool = torch.mean(x, dim=1, keepdim=True)      # [B, 1, N, N]
        pool_out = torch.cat([max_pool, avg_pool], dim=1)  # [B, 2, N, N]
        attention = self.conv(pool_out)                    # [B, 1, N, N]
        attention = self.sigmoid(attention)
        x = x * attention  # [B, num_heads, N, N]

        # ---- Second spatial attention (over spatial dimension) ----
        # Reshape to 5D: [B, num_heads, H, W, N]
        x = x.view(B, num_heads, H, W, N)

        # Pool over the last dimension (N = H*W)
        max_pool = torch.max(x, dim=-1, keepdim=True)[0]   # [B, num_heads, H, W, 1]
        avg_pool = torch.mean(x, dim=-1, keepdim=True)      # [B, num_heads, H, W, 1]
        pool_out = torch.cat([max_pool, avg_pool], dim=-1)  # [B, num_heads, H, W, 2]
        pool_out = pool_out.view(B * num_heads, 2, H, W)    # [B*num_heads, 2, H, W]

        attention = self.conv(pool_out)                     # [B*num_heads, 1, H, W]
        attention = self.sigmoid(attention)
        attention = attention.view(B, num_heads, H, W, 1)   # [B, num_heads, H, W, 1]

        x = x * attention                                    # [B, num_heads, H, W, N]
        x = x.view(B, num_heads, N, N)                      # [B, num_heads, N, N]

        return x


# ============================================================
# Cross-Attention with CBAM
# ============================================================
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, spatial_kernel_size=3, spatial_attention=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5
        self.use_spatial = spatial_attention
        self.cbam = CBAMSpatialAttention(kernel_size=spatial_kernel_size) if spatial_attention else None

    def forward(self, x):
        """
        Args:
            x: [B, H, W, C]
        Returns:
            [B, H, W, C]
        """
        B, H, W, C = x.shape
        N = H * W

        x_1 = x.view(B, N, C)

        q = self.q_linear(x_1)
        k = self.k_linear(x_1)
        v = self.v_linear(x_1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_probs = F.softmax(attn_scores, dim=-1)                      # [B, num_heads, N, N]

        # Apply CBAM spatial attention (HPHA second level)
        if self.use_spatial:
            attn_probs = self.cbam(attn_probs)                           # [B, num_heads, N, N]

        attn_output = torch.matmul(attn_probs, v)                        # [B, num_heads, N, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        output = self.out_linear(attn_output)                            # [B, N, C]
        output = output.view(B, H, W, C)                                 # [B, H, W, C]

        return output


# ============================================================
# Helper: Build SAM Image Encoder with pretrained weights
# ============================================================
def build_sam_image_encoder(
    img_size: int = 224,
    patch_size: int = 16,
    checkpoint_path: str = None,
    freeze: bool = True,
) -> ImageEncoderViT:
    """
    Build SAM ViT-B image encoder and optionally load pretrained weights.
    Handles interpolation of pos_embed and global rel_pos for different input sizes.

    Args:
        img_size: input image size
        patch_size: patch size for embedding
        checkpoint_path: path to SAM pretrained checkpoint
        freeze: whether to freeze backbone parameters
    Returns:
        ImageEncoderViT instance
    """
    encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11),
    )

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading SAM pretrained weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Extract only image_encoder weights
        img_encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('image_encoder.'):
                img_encoder_state[k[len('image_encoder.'):]] = v

        # Interpolate pos_embed if needed
        token_size = img_size // patch_size
        if 'pos_embed' in img_encoder_state:
            pos_embed = img_encoder_state['pos_embed']  # [1, orig_H, orig_W, 768]
            if pos_embed.shape[1] != token_size:
                print(f"  Interpolating pos_embed: {pos_embed.shape[1]}x{pos_embed.shape[2]} -> {token_size}x{token_size}")
                pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, C, H, W]
                pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
                pos_embed = pos_embed.permute(0, 2, 3, 1)  # [1, H, W, C]
                img_encoder_state['pos_embed'] = pos_embed

        # Interpolate rel_pos for global attention blocks (which see full feature map)
        global_attn_indexes = [2, 5, 8, 11]
        for k in list(img_encoder_state.keys()):
            if 'rel_pos' in k:
                # Extract block index from key like 'blocks.2.attn.rel_pos_h'
                parts = k.split('.')
                block_idx = int(parts[1])
                if block_idx in global_attn_indexes:
                    rel_pos_params = img_encoder_state[k]  # [2*token_size_orig-1, head_dim]
                    orig_rel_size = rel_pos_params.shape[0]
                    target_rel_size = token_size * 2 - 1
                    if orig_rel_size != target_rel_size:
                        print(f"  Interpolating {k}: rel_pos {orig_rel_size} -> {target_rel_size}")
                        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)  # [1, 1, orig, head_dim]
                        rel_pos_params = F.interpolate(
                            rel_pos_params, (target_rel_size, rel_pos_params.shape[-1]),
                            mode='bilinear', align_corners=False
                        )
                        img_encoder_state[k] = rel_pos_params[0, 0, ...]

        # Load state dict
        missing, unexpected = encoder.load_state_dict(img_encoder_state, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
            for mk in missing[:5]:
                print(f"    - {mk}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Pretrained weights loaded successfully.")
    else:
        print(f"No checkpoint found at {checkpoint_path}, using random initialization.")

    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder


# ============================================================
# Main Multi-Task Model
# ============================================================
class eye_multi_task(nn.Module):
    """
    Multi-task model with SAM ViT-B backbone.
    Supports three tasks: segmentation, classification, VQA.
    """
    def __init__(self, params=None) -> None:
        super().__init__()

        lora_rank = params.get("initial_lora_rank", 64) if params else 64
        image_size = params.get("image_size", 224) if params else 224
        use_elt = params.get("use_elt", False) if params else False
        attention = params.get("attention", "hpha") if params else "hpha"

        # Build SAM image encoder with pretrained weights
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "SAM", "sam_vit_b_01ec64.pth"
        )
        sam_encoder = build_sam_image_encoder(
            img_size=image_size,
            patch_size=16,
            checkpoint_path=checkpoint_path,
            freeze=True,
        )

        # Wrap in feature extractor with LoRA
        self.image_encoder = SAMFeatureExtractor(sam_encoder, lora_rank=lora_rank, use_elt=use_elt)

        # Classifier: 768-dim features → num_classes
        self.classer = classer(768, params["cls_num_class"] if params else 4)

        # VQA module
        self.vqa = VisualBertResMLPClassification(
            vocab_size=4105,
            layers=6,
            n_heads=8,
            num_class=params["vqa_class"] if params else 10,
            token_size=int(80 + (5 * 5))
        )

        # Segmentation decoder
        self.unet_decoder = SAMDecoder(num_classes=params["seg_num_class"] if params else 2)

        # Task embedding and projection
        self.num_tasks = 3
        self.task_embed = nn.Embedding(self.num_tasks, 768)
        self.task_project = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )

        # Cross-attention for feature enhancement (HPHA module).
        # attention: 'hpha' (self-attn + spatial), 'self' (self-attn only), 'none' (identity)
        if attention == "none":
            self.cross_attention = nn.Identity()
        else:
            self.cross_attention = CrossAttention(
                embed_dim=768, num_heads=12, spatial_attention=(attention != "self")
            )

    def update_lora_ranks(self, new_rank):
        """Update the active rank of all LoRA layers (ELT)."""
        self.image_encoder.update_all_lora_ranks(new_rank)

    def get_entropy_based_rank(self, info_retention_ratio=0.9):
        """Return the entropy-based suggested rank for all LoRA layers (ELT)."""
        return self.image_encoder.get_optimal_rank_by_entropy(info_retention_ratio)

    def get_current_lora_info(self):
        """Return the current LoRA configuration summary."""
        if len(self.image_encoder.lora_modules) > 0:
            sample_layer = self.image_encoder.lora_modules[0]
            return {
                'current_rank': sample_layer.current_rank,
                'max_rank': sample_layer.max_rank,
                'total_lora_layers': len(self.image_encoder.lora_modules),
                'scaling': sample_layer.scaling
            }
        return {}

    def forward(self, input, vqa_text_inputs=None, tasks=None):
        """
        Args:
            input: [B, 3, 224, 224] image tensor
            vqa_text_inputs: tokenized VQA questions
            tasks: 可选任务列表(如 ['seg','cls','vqa']); None 表示计算全部任务。
                   传入后跳过未启用任务的头, 避免单/双任务消融时未用头的激活被保留导致 OOM。
        Returns:
            seg_output: [B, num_seg_classes, 224, 224] 或 None
            cls_output: [B, num_cls_classes] 或 None
            vqa_output: [B, num_vqa_classes] 或 None
        """
        # SAM backbone: [B, 14, 14, 768]
        image_embeddings, stage_outputs = self.image_encoder(input)

        # Cross-attention enhancement: [B, 14, 14, 768]
        image_embeddings = self.cross_attention(image_embeddings)

        # Segmentation: [B, num_seg_classes, 224, 224]
        if tasks is None or 'seg' in tasks:
            seg_output = self.unet_decoder(image_embeddings, stage_outputs)
        else:
            seg_output = None

        # Classification: [B, num_cls_classes]
        if tasks is None or 'cls' in tasks:
            cls_output = self.classer(image_embeddings)
        else:
            cls_output = None

        # VQA: pool 14x14 → 7x7 to keep compatibility with VQA adapter
        # VQA adapter expects [B, 49, 768] (7x7 flattened)
        if vqa_text_inputs is not None and (tasks is None or 'vqa' in tasks):
            vqa_visual = F.adaptive_avg_pool2d(
                image_embeddings.permute(0, 3, 1, 2), (7, 7)
            ).permute(0, 2, 3, 1)  # [B_img, 7, 7, 768]
            # Expand visual features to match text batch size (multiple questions per image)
            text_bs = vqa_text_inputs['input_ids'].shape[0]
            if vqa_visual.shape[0] != text_bs:
                vqa_visual = vqa_visual.expand(text_bs, -1, -1, -1)  # [N_q, 7, 7, 768]
            vqa_output = self.vqa(vqa_text_inputs, vqa_visual)
        else:
            vqa_output = None

        return seg_output, cls_output, vqa_output
