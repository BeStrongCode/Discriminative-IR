import numbers

import torch.nn as nn
# from net.model import LayerNorm
# from net.model import FeedForward
import torch
from einops import rearrange
import torch.nn.functional as F



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.dropout = nn.Dropout(0.2)

        # self.is_dropout = is_dropout

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.dropout(x)
        x = self.project_out(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

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

class ch_shuffle_high_text(nn.Module):
    def __init__(self, ch_dim,num_heads,LayerNorm_type,ffn_expansion_factor, bias,lin_ch=512):
        super(ch_shuffle_high_text, self).__init__()
        self.dim = ch_dim
        self.linear_layer1 = nn.Linear(lin_ch,lin_ch)
        self.linear_layer3 = nn.Linear(lin_ch,2*ch_dim)
# -----------------------------------------------------------------------
        self.conv1x1   = nn.Conv2d(ch_dim, 2*ch_dim, kernel_size=1, stride=1, padding=0) #
        self.conv_out   = nn.Conv2d(2*ch_dim, ch_dim, kernel_size=1, stride=1, padding=0) #
        self.norm1 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm2 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm3 = LayerNorm(ch_dim, LayerNorm_type)
        self.select_attn = Topm_CrossAttention_Restormer(ch_dim, num_heads, bias=False)
        self.ffn = FeedForward(ch_dim, ffn_expansion_factor, bias)
    def forward(self, img_featur, text_code):
        b,c,_,_ = img_featur.shape
        img_feature2 = img_featur
        text_code = self.linear_layer1(text_code)
        text_code = self.linear_layer3(text_code)
        soft_values, soft_indices = torch.topk(text_code, k=2*self.dim)
        img_featur = self.conv1x1(img_featur)
        shuffled_img = img_featur[torch.arange(b).unsqueeze(1), soft_indices, :, :] # shuffle

        q = self.conv_out(shuffled_img)
        att = self.select_attn(self.norm1(q),self.norm2(img_feature2))
        output = att + self.ffn(self.norm3(att))
        return output,img_feature2

class Topm_CrossAttention_Restormer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Topm_CrossAttention_Restormer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x_q, x_kv):
        b,c,h,w = x_q.shape
        q = self.q_dwconv(self.q(x_q))
        kv = self.kv_dwconv(self.kv(x_kv))
        k,v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        index = torch.topk(attn, k=int(C*9/10), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))
        attn4 = attn4.softmax(dim=-1)
        out4 = (attn4 @ v)
        out =  out4 * self.attn4
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)


    def forward(self,x):
        B,C,H,W = x.shape
        emb_max = x.amax(dim=(-2,-1))

        # 训练使用gumbel_softmax，而测试使用softmax
        prompt_weights = F.gumbel_softmax(self.linear_layer(emb_max),tau=1.5,hard=False,dim=1)
        # prompt_weights = F.softmax(self.linear_layer(emb),dim=1)

        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt,prompt_weights

class Prompt(nn.Module):
    def __init__(self,dim,ffn_expansion_factor,bias,LayerNorm_type='WithBias'):
        super(Prompt,self).__init__()
        self.pgm = PromptGenBlock(prompt_dim=384, prompt_len=5, prompt_size=16, lin_dim=384)

        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.conv_out   = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0) #
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.select_attn = Topm_CrossAttention_Restormer(dim, num_heads=8, bias=False)
    def forward(self, img_featur):
        b,c,_,_ = img_featur.shape
        img_feature2 = img_featur
        prompt,weights = self.pgm(img_featur)
        q = self.conv_out(prompt)
        att = self.select_attn(self.norm1(q),self.norm2(img_feature2))
        output = att + self.ffn(self.norm3(att))
        return output

if __name__ == '__main__':
    x = torch.rand(8,384,16,16)
    Prompt = Prompt(dim=384,ffn_expansion_factor=2.26,bias=False)
    output = Prompt(x)
    print(output.shape)
