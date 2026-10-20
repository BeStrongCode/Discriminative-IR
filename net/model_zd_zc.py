## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090
import math
import warnings



import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time
# from net.crossPromptAttention import Prompt
# from net.LearnFFT1 import LearnableRFFT2, LearnableIRFFT2

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    




class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # self.dropout = nn.Dropout(0.2)

        # self.is_dropout = is_dropout

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        # x = self.dropout(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.is_prompt = is_prompt
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        #
        # self.prompt = nn.Parameter(torch.ones(num_heads, dim//num_heads, 1))


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # if self.is_prompt:
        #     q = q + self.prompt
        #     k = k + self.prompt
        #     v = v + self.prompt


        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
##----------ChannelAttention Prompt Gen Module -----------------------
class ChannelAttn_PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(ChannelAttn_PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(lin_dim, lin_dim*2),
            nn.ReLU(),
            nn.Linear(lin_dim*2, lin_dim)
        )
        # self.scale = nn.Parameter(torch.zeros(1,lin_dim*2))

        # self.linear_layer1 = nn.Linear(lin_dim*2, lin_dim)


    def forward(self,x):
        B,C,H,W = x.shape
        emb_max = x.amax(dim=(-2,-1))
        emb_min = x.mean(dim=(-2,-1))
        emb_max = self.mlp(emb_max)
        emb_min = self.mlp(emb_min)
        emb = emb_max + emb_min

        # sigmoid gating
        prompt_weights = F.sigmoid(self.linear_layer(emb))


        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt,prompt_weights

##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        # self.mlp = nn.Sequential(
        #     nn.Linear(lin_dim, lin_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(lin_dim*2, lin_dim)
        # )
        # self.scale = nn.Parameter(torch.zeros(1,lin_dim*2))

        # self.linear_layer1 = nn.Linear(lin_dim*2, lin_dim)


    def forward(self,x):
        # B,C,H,W = x.shape
        # emb_max = x.amax(dim=(-2,-1))
        # # emb_min = x.mean(dim=(-2,-1))
        # # emb_max = self.mlp(emb_max)
        # # emb_min = self.mlp(emb_min)
        # # emb = emb_max + emb_min
        # # emb_min = x.mean(dim=(-2,-1))
        # # emb = torch.cat([emb_max,emb_min],dim=-1)
        #
        # # 训练使用gumbel_softmax，而测试使用softmax
        # prompt_weights = F.gumbel_softmax(self.linear_layer(emb_max),tau=1.5,hard=False,dim=1)
        # # prompt_weights = F.softmax(self.linear_layer(emb_max),dim=1)
        #
        # prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        # prompt = torch.sum(prompt,dim=1)
        # prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        # prompt = self.conv3x3(prompt)
        #
        # return prompt,prompt_weights
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

##########################################################################
##---------- Prompt Gen Module -----------------------
class Maxpool_gumbel_PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(Maxpool_gumbel_PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        # self.scale = nn.Parameter(torch.zeros(1,lin_dim*2))

        # self.linear_layer1 = nn.Linear(lin_dim*2, lin_dim)


    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.amax(dim=(-2,-1))
        prompt_weights = F.gumbel_softmax(self.linear_layer(emb),tau=1.5,hard=False,dim=1)
        # prompt_weights = F.softmax(self.linear_layer(emb),dim=1)

        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt,prompt_weights



class DualBranchDecomposition(nn.Module):
    """
    Dual-branch module for content / degradation feature decomposition.
    - Content branch: preserve spatial structure, degradation-invariant
    - Degradation branch: global, low-dimensional, degradation-discriminative
    """
    def __init__(self, in_channels, deg_dim=128):
        super().__init__()

        # -----------------------------
        # Degradation branch g_d
        # -----------------------------
        self.deg_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.deg_pool = nn.AdaptiveAvgPool2d(1)
        self.deg_fc = nn.Sequential(
            nn.Linear(in_channels // 2, deg_dim),
            nn.ReLU(inplace=True),
            nn.Linear(deg_dim, deg_dim)
        )

        # Optional: degradation classifier (only for Phase I)
        self.deg_classifier = nn.Linear(deg_dim, 1)  # set num_classes later

        # -----------------------------
        # Content branch g_c
        # -----------------------------
        self.content_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.content_norm = nn.BatchNorm2d(in_channels)

    def forward(self, feat):
        """
        Args:
            feat: encoder feature map [B, C, H, W]
        Returns:
            content_feat: [B, C, H, W]
            deg_feat: [B, D]
        """

        # ===== Degradation branch =====
        d = self.deg_conv(feat)            # [B, C/2, H, W]
        d = self.deg_pool(d).flatten(1)    # [B, C/2]
        deg_feat = self.deg_fc(d)          # [B, D]

        # ===== Content branch =====
        content_feat = self.content_conv(feat)
        content_feat = self.content_norm(content_feat)
        content_feat = content_feat + feat  # residual to preserve structure

        return content_feat, deg_feat

class FiLMModulation(nn.Module):
    """
    Degradation-guided FiLM modulation for content features
    """
    def __init__(self, content_channels, deg_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(deg_dim, content_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(content_channels * 2, content_channels * 2)
        )

    def forward(self, content_feat, deg_feat):
        """
        content_feat: [B, C, H, W]
        deg_feat:     [B, D]
        """
        B, C, H, W = content_feat.shape

        gamma_beta = self.mlp(deg_feat)        # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)

        return gamma * content_feat + beta
    
# class CrossAttention(nn.Module):
#     def __init__(self, dim, deg_dim, num_heads=8, qkv_bias=True):
#         super().__init__()
#         assert dim % num_heads == 0

#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.q_proj = nn.Linear(deg_dim, dim, bias=qkv_bias)
#         self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

#         self.out_proj = nn.Linear(dim, dim)

#     def forward(self, q_vec, kv_tokens):
#         """
#         q_vec:     [B, deg_dim]      (退化向量，1 token)
#         kv_tokens: [B, N, dim]       (内容特征 tokens)
#         """
#         B, N, C = kv_tokens.shape

#         # ---- Q: [B, 1, dim] ----
#         q = self.q_proj(q_vec).unsqueeze(1)

#         # ---- K,V: [B, N, dim] ----
#         k = self.k_proj(kv_tokens)
#         v = self.v_proj(kv_tokens)

#         # ---- multi-head ----
#         q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)

#         out = attn @ v
#         out = out.transpose(1, 2).contiguous().view(B, 1, C)
#         out = self.out_proj(out)

#         return out.squeeze(1)  # [B, dim]


# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, deg_dim, num_heads=8):
#         super().__init__()
#         self.attn = CrossAttention(dim, deg_dim, num_heads)

#         # 用 cross-attn 输出调制空间特征（比直接加好）
#         self.modulator = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Sigmoid()
#         )

#     def forward(self, deg_vec, content_feat):
#         """
#         deg_vec:       [B, deg_dim]
#         content_feat:  [B, C, H, W]
#         """
#         B, C, H, W = content_feat.shape

#         kv = content_feat.flatten(2).transpose(1, 2)  # [B, HW, C]

#         global_feat = self.attn(deg_vec, kv)  # [B, C]

#         gamma = self.modulator(global_feat).view(B, C, 1, 1)

#         return content_feat * gamma + content_feat


##########################################################################
##---------- Restormer -----------------------

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor



class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        #prompt1
        # self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
        # self.noise_level1 = TransformerBlock(dim=int(dim * 2 ** 1) + 64, num_heads=heads[2],ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        # self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1) + 64, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        #
        # #prompt2
        # self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
        # self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        # self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)



        # prompt3
        # self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=7, prompt_size=16, lin_dim=384)
        # # # self.prompt3 = ChannelAttn_PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)
        # # # self.fftprompt = LearnFFTPromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)
        # # # # self.channel_prompt = channel_prompt(ch=384)
        # # # # 576 512
        # self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        # self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2*2),kernel_size=1,bias=bias)

        self.DualBranchDecomposition = DualBranchDecomposition(in_channels=int(dim*2**3), deg_dim=128)

        # 内容特征与退化特征 交互模块
        self.FiLMModulation = FiLMModulation(content_channels=int(dim*2**3), deg_dim=128)
        # self.crossattention = CrossAttentionBlock(dim=int(dim*2**3), deg_dim=128, num_heads=8)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    def forward(self, inp_img):

        # encoder_block1
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # encoder_block2
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # encoder_block3
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # encoder_block4 h/8,w/8,8c
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # prompt3
        # dec3_param = self.prompt3(latent)
        # #
        # #
        # latent = torch.cat([latent, dec3_param], 1)
        # latent = self.noise_level3(latent)
        # latent = self.reduce_noise_level3(latent) # 384,16,16

        content_branch, deg_branch = self.DualBranchDecomposition(latent)
        
        # return content_branch, deg_branch
        # latent = self.FiLMModulation(content_branch, deg_branch)
        mod = self.FiLMModulation(content_branch, deg_branch)
        latent = latent + self.alpha * mod

        # # return latent


        #
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) # 192 32 32

        # dec2_param, weight2 = self.prompt2(out_dec_level3)
        # out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
        # out_dec_level3 = self.noise_level2(out_dec_level3)
        # out_dec_level3 = self.reduce_noise_level2(out_dec_level3)


        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) # 96 64 64

        # prompt1
        # dec1_param, weight3 = self.prompt1(out_dec_level2)
        # out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
        # out_dec_level2 = self.noise_level1(out_dec_level2)
        # out_dec_level2 = self.reduce_noise_level1(out_dec_level2)



        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1) # 96 128 128

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        # return out_dec_level1, latent, out_dec_level3, out_dec_level2, x4
        return out_dec_level1, latent


# model = Restormer()
# x = torch.randn(1, 3, 128, 128)
# y = model(x)
# print(y[0].shape)    