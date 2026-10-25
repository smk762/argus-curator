"""MiVOLO v2 network definition, used only by ``argus-curator mivolo-convert``.

Adapted from MiVOLO (https://github.com/WildChlamydia/MiVOLO,
``mivolo/model/mivolo_model.py`` and ``cross_bottleneck_attn.py``),
Copyright 2023 Irina Tolstykh, Maxim Kuprashevich, licensed under the Apache
License 2.0, itself adapted from timm's VOLO. Changes: reduced to the
``mivolo_d1_384`` face+person variant that ``iitolstykh/mivolo_v2`` ships, and
``VOLO.__init__`` is called by keyword so the module works with current timm
(upstream's positional call only fits timm 0.8.x). The forward pass and the
state-dict layout are unchanged, so the published weights load strictly.

Importing this module needs torch + timm (``[mivolo-convert]``); nothing on the
inference path imports it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.layers import Mlp, make_divisible, trunc_normal_
from timm.layers.bottleneck_attn import PosEmbedRel
from timm.models.volo import VOLO

INPUT_SIZE = 384


class CrossBottleneckAttn(nn.Module):
    """Cross attention between the face and person token maps."""

    def __init__(self, dim, dim_out=None, feat_size=None, num_heads=1, qk_ratio=1.0, qkv_bias=False):
        super().__init__()
        assert feat_size is not None, "A concrete feature size matching expected input (H, W) is required"
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0

        self.num_heads = num_heads
        self.dim_head_qk = make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5

        self.qkv_f = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias)
        self.qkv_p = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias)
        self.pos_embed = PosEmbedRel(feat_size, dim_head=self.dim_head_qk, scale=self.scale)
        self.norm = nn.LayerNorm([self.dim_out_v * 2, *feat_size])
        self.mlp = Mlp(
            in_features=self.dim_out_v * 2,
            hidden_features=int(dim * 4),
            act_layer=nn.GELU,
            out_features=dim_out,
            drop=0,
            use_conv=True,
        )
        self.pool = nn.Identity()

    def get_qkv(self, x, qvk_conv):
        B, C, H, W = x.shape
        x = qvk_conv(x)
        q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)
        v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        return q, k, v

    def apply_attn(self, q, k, v, B, H, W):
        attn = (q @ k) * self.scale + self.pos_embed(q)
        attn = attn.softmax(dim=-1)
        return (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        dim = int(C / 2)
        q_f, k_f, v_f = self.get_qkv(x[:, :dim], self.qkv_f)
        q_p, k_p, v_p = self.get_qkv(x[:, dim:], self.qkv_p)
        out_f = self.apply_attn(q_f, k_p, v_p, B, H, W)  # person -> face
        out_p = self.apply_attn(q_p, k_f, v_f, B, H, W)  # face -> person
        x_pf = self.norm(torch.cat((out_f, out_p), dim=1))
        return self.pool(self.mlp(x_pf))


def _stem(in_chans: int, hidden_dim: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
    )


class PatchEmbed(nn.Module):
    """Separate face / person stems fused by cross attention (6-channel input)."""

    def __init__(self, img_size: int, stem_stride: int = 2, patch_size: int = 8, hidden_dim: int = 64, embed_dim=192):
        super().__init__()
        self.conv1 = _stem(3, hidden_dim, stem_stride)
        self.conv2 = _stem(3, hidden_dim, stem_stride)
        k = patch_size // stem_stride
        self.proj1 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=k, stride=k)
        self.proj2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=k, stride=k)
        feat = img_size // patch_size
        self.map = CrossBottleneckAttn(embed_dim, dim_out=embed_dim, num_heads=1, feat_size=(feat, feat))

    def forward(self, x):
        x1 = self.proj1(self.conv1(x[:, :3]))
        x2 = self.proj2(self.conv2(x[:, 3:]))
        return self.map(torch.cat([x1, x2], dim=1))


class MiVOLOModel(VOLO):
    """``mivolo_d1_384``: output is ``[gender logit x2, normalised age]``."""

    def __init__(self) -> None:
        super().__init__(
            layers=(4, 4, 8, 2),
            img_size=INPUT_SIZE,
            in_chans=6,
            num_classes=3,
            embed_dims=(192, 384, 384, 384),
            num_heads=(6, 12, 12, 12),
        )
        self.patch_embed = PatchEmbed(img_size=INPUT_SIZE, embed_dim=192)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        x = self.forward_tokens(x)
        if self.post_network is not None:
            x = self.forward_cls(x)
        return self.norm(x)

    def forward_head(self, x, pre_logits: bool = False):
        out = x[:, 0]
        if pre_logits:
            return out
        out = self.head(out)
        if self.aux_head is not None:
            out = out + 0.5 * self.aux_head(x[:, 1:]).max(1)[0]
        return out

    def forward(self, x):
        return self.forward_head(self.forward_features(x))
