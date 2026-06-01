"""PromptIR: Prompting for All-in-One Image Restoration (NeurIPS 2023).

Potlapalli et al., https://arxiv.org/abs/2306.13090
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Layer-norm helpers
# ---------------------------------------------------------------------------

def _to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def _to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type='WithBias'):
        super().__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


# ---------------------------------------------------------------------------
# Transformer building blocks (from Restormer)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Gated-DConv Feed-Forward Network (GDFN)."""

    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1,
            padding=1, groups=hidden * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA)."""

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1,
            padding=1, groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Downsampling / Upsampling
# ---------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    """n_feat -> n_feat*2 channels, spatial /2."""

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """n_feat -> n_feat//2 channels, spatial *2."""

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------
# Prompt generation block
# ---------------------------------------------------------------------------

class PromptGenBlock(nn.Module):
    """Learns a bank of prompt prototypes and weights them per input."""

    def __init__(self, prompt_dim, prompt_len=5, prompt_size=32, lin_dim=None):
        super().__init__()
        if lin_dim is None:
            lin_dim = prompt_dim
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1))                            # (B, C)
        weights = F.softmax(self.linear_layer(emb), dim=1)   # (B, prompt_len)
        weights = weights.view(b, -1, 1, 1, 1)               # (B, L, 1, 1, 1)
        prompt = (weights * self.prompt_param.expand(b, -1, -1, -1, -1)).sum(dim=1)
        prompt = F.interpolate(prompt, (h, w), mode='bilinear', align_corners=False)
        return self.conv3x3(prompt)                           # (B, prompt_dim, H, W)


# ---------------------------------------------------------------------------
# PromptIR main model
# ---------------------------------------------------------------------------

class PromptIR(nn.Module):
    """
    PromptIR for all-in-one image restoration.

    Default config matches the paper settings for two-degradation tasks.
    Channel dimensions at each encoder level (dim=48):
        L1: 48,  L2: 96,  L3: 192,  Latent: 384
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=None,
        num_refinement_blocks=4,
        heads=None,
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type='WithBias',
        prompt_len=5,
        prompt_size=32,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        lnt = layer_norm_type

        def _make_blocks(n, d, h):
            return nn.Sequential(*[
                TransformerBlock(d, h, ffn_expansion_factor, bias, lnt)
                for _ in range(n)
            ])

        # ---- Patch embedding ----
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # ---- Encoder ----
        self.encoder_l1 = _make_blocks(num_blocks[0], dim, heads[0])
        self.down1_2 = Downsample(dim)                         # -> dim*2

        self.encoder_l2 = _make_blocks(num_blocks[1], dim * 2, heads[1])
        self.down2_3 = Downsample(dim * 2)                     # -> dim*4

        self.encoder_l3 = _make_blocks(num_blocks[2], dim * 4, heads[2])
        self.down3_4 = Downsample(dim * 4)                     # -> dim*8

        # ---- Latent ----
        self.latent = _make_blocks(num_blocks[3], dim * 8, heads[3])

        # ---- Decoder level 3 ----
        self.up4_3 = Upsample(dim * 8)                        # -> dim*4
        self.reduce_chan_l3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.prompt3 = PromptGenBlock(dim * 4, prompt_len, prompt_size)
        self.noise_l3 = TransformerBlock(dim * 8, heads[2], ffn_expansion_factor, bias, lnt)
        self.reduce_noise_l3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_l3 = _make_blocks(num_blocks[2], dim * 4, heads[2])

        # ---- Decoder level 2 ----
        self.up3_2 = Upsample(dim * 4)                        # -> dim*2
        self.reduce_chan_l2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.prompt2 = PromptGenBlock(dim * 2, prompt_len, prompt_size)
        self.noise_l2 = TransformerBlock(dim * 4, heads[1], ffn_expansion_factor, bias, lnt)
        self.reduce_noise_l2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_l2 = _make_blocks(num_blocks[1], dim * 2, heads[1])

        # ---- Decoder level 1 ----
        self.up2_1 = Upsample(dim * 2)                        # -> dim
        self.reduce_chan_l1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.prompt1 = PromptGenBlock(dim, prompt_len, prompt_size)
        self.noise_l1 = TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, lnt)
        self.reduce_noise_l1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.decoder_l1 = _make_blocks(num_blocks[0], dim, heads[0])

        # ---- Refinement + output ----
        self.refinement = _make_blocks(num_refinement_blocks, dim, heads[0])
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp):
        # Encoder
        feat = self.patch_embed(inp)

        enc_l1 = self.encoder_l1(feat)
        enc_l2 = self.encoder_l2(self.down1_2(enc_l1))
        enc_l3 = self.encoder_l3(self.down2_3(enc_l2))
        latent = self.latent(self.down3_4(enc_l3))

        # Decoder level 3
        x = self.reduce_chan_l3(torch.cat([self.up4_3(latent), enc_l3], dim=1))
        p3 = self.prompt3(x)
        x = self.reduce_noise_l3(self.noise_l3(torch.cat([x, p3], dim=1)))
        x = self.decoder_l3(x)

        # Decoder level 2
        x = self.reduce_chan_l2(torch.cat([self.up3_2(x), enc_l2], dim=1))
        p2 = self.prompt2(x)
        x = self.reduce_noise_l2(self.noise_l2(torch.cat([x, p2], dim=1)))
        x = self.decoder_l2(x)

        # Decoder level 1
        x = self.reduce_chan_l1(torch.cat([self.up2_1(x), enc_l1], dim=1))
        p1 = self.prompt1(x)
        x = self.reduce_noise_l1(self.noise_l1(torch.cat([x, p1], dim=1)))
        x = self.decoder_l1(x)

        # Refinement + global residual
        x = self.refinement(x)
        return self.output(x) + inp
