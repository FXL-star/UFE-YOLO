import torch
import torch.nn as nn
from einops import rearrange
__all__ =['TSSA',"C2PSA_TSSA",'C2PSA_TSSA_DyT','C2f_TSSA','C3k2_TSSA']

from .block import C3

'''
来自ICLR2025顶会  适用于CV、NLP、时间序列预测所有任务通用--即插即用注意力模块
即插即用注意力模块： TSSA 

本文主要内容;
    注意力机制可以说是 Transformer 架构的关键区分因素，而 Transformer 近年来在各种任务上都展现了最先进的性能。
然而，Transformer 的注意力机制通常会带来较大的计算开销，其计算复杂度随着 token 数量呈二次增长。
在本研究中，我们提出了一种新的 Transformer 注意力机制，其计算复杂度随着 token 数量呈线性增长。

    我们的方法基于先前的研究，该研究表明，通过“白盒”架构设计，Transformer 风格的架构可以自然地生成，
其中网络的每一层被设计为执行最大编码率缩减（MCR²）目标的增量优化步骤。
具体而言，我们推导出 MCR² 目标的一种新的变分形式，并证明从该变分目标的展开梯度下降中，
可以得到一个新的注意力模块——Token Statistics Self-Attention（TSSA）。

    TSSA 具有线性的计算和存储复杂度，并且与传统的注意力架构完全不同，后者通常通过计算 token 之间的两两相似度来实现注意力机制。
我们的实验表明，在视觉、自然语言处理以及长序列任务上，仅仅用 TSSA 替换标准的自注意力，
就能够在计算成本显著降低的情况下，实现与传统 Transformer 相当的性能。
此外，我们的结果也对传统认知提出了挑战，即 Transformer 之所以成功，是否真的依赖于基于两两相似度的注意力机制。

TSSA注意力总结:
    TSSA 模块旨在提升注意力机制对局部和全局特征的建模能力，特别是针对视觉任务中不同区域信息的重要性差异。
    它通过引入 Token 统计信息（如均值、方差等）来增强自注意力机制，使得注意力分配更加合理，从而提高特征提取的精准度和网络的整体表现。

原理：
TSSA 结合了传统自注意力机制和 Token 统计特征，主要包括以下几个关键步骤：
    1. Token 统计特征提取：计算输入特征图中Token 的统计信息，如均值和方差，以获取全局和局部的统计分布。
    2. 注意力权重计算：将统计特征与输入特征结合，通过自注意力机制计算加权注意力分布，使得模型能够更好地关注关键区域。
    3. 自适应特征增强：基于计算得到的注意力权重，调整输入特征的分布，使得重要信息得到强化，抑制冗余或无关信息。
    4. 输出优化特征：经过 TSSA 处理后的特征更具表达力，同时保留了局部结构信息和全局关系，提高了模型对复杂场景的适应性。
TSSA 主要通过 Token 统计特征的引入来优化注意力机制，使得模型能够更加精准地学习图像中的关键特征，在各种视觉任务中（如分类、检测、分割等）都有潜在的应用价值

TSSA模块适用：图像分类、目标检测、图像分割、遥感语义分割、图像增强、图像去噪、暗光增强等CV所有任务；NLP所有任务; 时间序列预测所有任务通过即插即用模块。

'''

class TSSA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()

        self.heads = num_heads

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        B,C,H,W = x.size()
        x = x.reshape(B, C, -1).transpose(-1, -2)
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        b, h, N, d = w.shape

        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # b * h * n

        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out.view(B,C,H,W)
        return out

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=False, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class PSABlock_TSSA(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = TSSA(c, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""

        x = x + self.attn(x)if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class PSABlock_TSSA_DyT(PSABlock_TSSA):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.dyt1 = DynamicTanh(normalized_shape=c)
        self.dyt2 = DynamicTanh(normalized_shape=c)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x)) if self.add else self.attn(self.dyt1(x))
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x

class C2PSA_TSSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock_TSSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class C2PSA_TSSA_DyT(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock_TSSA_DyT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

# --------
class Bottleneck_TSSA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = TSSA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))



class C2f_TSSA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_TSSA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_TSSA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_TSSA(C2f_TSSA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_TSSA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
if __name__ == "__main__":
    #创建TSSA模块实例，32代表通道维度
    TSSA = TSSA(32)
    #   输入 B C H W, 输出 B C H W
    # 随机生成输入4维度张量：B, C, H, W
    input= torch.randn(1, 32, 32, 32)
    # 运行前向传递
    output = TSSA(input)
    # 输出输入图片张量和输出图片张量的形状
    print("CV_TSSA_input size:", input.size())
    print("CV_TSSA_Output size:", output.size())
