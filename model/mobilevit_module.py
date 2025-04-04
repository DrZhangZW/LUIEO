from typing import Optional, Tuple, Union, Dict
import math, torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from utils import logger


def get_normalization_layer(opts, num_features: int, norm_type: Optional[str] = None, num_groups: Optional[int] = None,
                            **kwargs):
    norm_type = getattr(opts, "model_normalization_name", "batch_norm") if norm_type is None else norm_type
    num_groups = getattr(opts, "model_normalization_groups", 1) if num_groups is None else num_groups
    momentum = getattr(opts, "model_normalization_momentum", 0.1)
    norm_layer = None
    norm_type = norm_type.lower()
    if norm_type in ['batch_norm', 'batch_norm_2d']:
        norm_layer = nn.BatchNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == 'batch_norm_1d':
        norm_layer = nn.BatchNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type in ['sync_batch_norm', 'sbn']:
        norm_layer = nn.SyncBatchNorm(num_features=num_features, momentum=momentum)
    elif norm_type in ['group_norm', 'gn']:
        num_groups = math.gcd(num_features, num_groups)
        norm_layer = nn.GroupNorm(num_channels=num_features, num_groups=num_groups)
    elif norm_type in ['instance_norm', 'instance_norm_2d']:
        norm_layer = nn.InstanceNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == "instance_norm_1d":
        norm_layer = nn.InstanceNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type in ['layer_norm', 'ln']:
        norm_layer = nn.LayerNorm(num_features)
    elif norm_type == 'identity':
        norm_layer = nn.Identity()
    return norm_layer


def get_activation_fn(act_type: str = 'swish', num_parameters: Optional[int] = -1, inplace: Optional[bool] = True,
                      negative_slope: Optional[float] = 0.1):
    act_layer = None
    if act_type == 'relu':
        act_layer = nn.ReLU(inplace=False)
    elif act_type == 'prelu':
        assert num_parameters >= 1
        act_layer = nn.PReLU(num_parameters=num_parameters)
    elif act_type == 'leaky_relu':
        act_layer = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif act_type == 'hard_sigmoid':
        act_layer = nn.Hardsigmoid(inplace=inplace)
    elif act_type == 'swish':
        act_layer = nn.SiLU()
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=inplace)
    elif act_type == 'hard_swish':
        act_layer = nn.Hardswish(inplace=inplace)
    return act_layer


def make_divisible(
        v: Union[float, int],
        divisor: Optional[int] = 8,
        min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(self, opts, in_channels: int, transformer_dim: int, ffn_dim: int,
                 n_transformer_blocks: int = 2, head_dim: int = 32, attn_dropout: float = 0.1,
                 dropout: float = 0.1, ffn_dropout: float = 0.1, patch_h: int = 8, patch_w: int = 8,
                 transformer_norm_layer: Optional[str] = "layer_norm", conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1, *args, **kwargs) -> None:
        super().__init__()
        conv_3x3_in = ConvLayer(
            opts=opts, in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True,
            groups=in_channels
        )  # bias=False, conv3x3+BN+Relu

        conv_1x1_in = ConvLayer(
            opts=opts, in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False
        )  # bias=False, conv1x1

        conv_1x1_out = ConvLayer(
            opts=opts, in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, use_norm=True, use_act=True
        )  # bias=False, conv1x1 +BN+Relu

        conv_3x3_out = None
        no_fusion = False
        # For MobileViTv3: input+global --> local+global
        if not no_fusion:
            # input_ch = tr_dim + in_ch
            conv_3x3_out = ConvLayer(
                opts=opts, in_channels=transformer_dim + in_channels, out_channels=in_channels,
                kernel_size=1, stride=1, use_norm=True, use_act=True
            )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)
        # conv+BN+Relu--> conv_1x1_in(adjust channels)
        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks
        global_rep = [
            TransformerEncoder_layer(opts=opts, embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx],
                                     num_heads=num_heads,
                                     attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout,
                                     transformer_norm_layer=transformer_norm_layer)
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=transformer_dim)
        )
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out  # bias=False, conv1x1 +BN+Relu
        self.fusion = conv_3x3_out  # bias=False, conv3x3 +BN+Relu

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.ffn_max_dim = ffn_dims[0]
        self.ffn_min_dim = ffn_dims[-1]

        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def forward(self, x: Tensor) -> Tensor:
        res = x
        # For MobileViTv3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        fm_conv = self.local_rep(x)
        # convert feature map to patches
        patches, info_dict = self.unfolding(fm_conv)
        # learn global representations
        patches = patches.permute(1, 0, 2)
        patches = self.global_rep(patches)
        patches = patches.permute(1, 0, 2)
        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)
        fm = self.conv_proj(fm)
        if self.fusion is not None:
            # For MobileViTv3: input+global --> local+global
            fm = self.fusion(
                torch.cat((fm_conv, fm), dim=1)
            )
        # For MobileViTv3: Skip connection
        fm = fm + res
        return fm

    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True
        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N
        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map


class TransformerEncoder_layer(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(P,N C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (int) : Number of heads in multi-head attention. Default: 8
        attn_dropout (float): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers. Default: 0.0

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(self, opts, embed_dim: int, ffn_latent_dim: int, num_heads: Optional[int] = 8,
                 attn_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.1, ffn_dropout: Optional[float] = 0.0,
                 transformer_norm_layer: Optional[str] = "layer_norm",
                 *args, **kwargs):
        super(TransformerEncoder_layer, self).__init__()
        # self.pre_norm_mha = nn.Sequential(
        #    get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
        #    nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, bias=True),
        #    nn.Dropout(p=dropout)
        # )
        self.pre_norm = get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, bias=True)
        self.dropout1 = nn.Dropout(dropout)
        # self.pre_norm_mha_test = nn.Sequential(
        #    get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
        #    nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, bias=True),
        #    nn.Dropout(p=dropout)
        # )
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            get_activation_fn(act_type=getattr(opts, "model_activation_name", "prelu"),
                              inplace=getattr(opts, "model_activation_inplace", False),
                              negative_slope=getattr(opts, "model_activation_neg_slope", 0.1),
                              num_parameters=1),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        # multi-head attention
        # query: has shape(query_nums,batches*patch_area,channel)
        # key,value : has shape (patches_nums,batches_patch_area,channel)
        res = x
        # test_x = self.pre_norm_mha_test(x)
        # x = self.pre_norm_mha(x)
        x = self.pre_norm(x)
        x = self.mha(query=x, key=x, value=x)[0]
        x = self.dropout1(x) + res
        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, opts, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 stride: Optional[int or tuple] = 1,
                 dilation: Optional[int or tuple] = 1, groups: Optional[int] = 1,
                 bias: Optional[bool] = False, padding_mode: Optional[str] = 'zeros',
                 use_norm: Optional[bool] = True, use_act: Optional[bool] = True,
                 *args, **kwargs
                 ) -> None:
        """
            Applies a 2D convolution over an input signal composed of several input planes.
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param kernel_size: kernel size
            :param stride: move the kernel by this amount during convolution operation
            :param dilation: Add zeros between kernel elements to increase the effective receptive field of the kernel.
            :param groups: Number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
            :param bias: Add bias or not
            :param padding_mode: Padding mode. Default is zeros
            :param use_norm: Use normalization layer after convolution layer or not. Default is True.
            :param use_act: Use activation layer after convolution layer/convolution layer followed by batch
            normalization or not. Default is True.
        """
        super(ConvLayer, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        if in_channels % groups != 0:
            logger.error('Input channels are not divisible by groups. {}%{} != 0 '.format(in_channels, groups))
        if out_channels % groups != 0:
            logger.error('Output channels are not divisible by groups. {}%{} != 0 '.format(out_channels, groups))

        block = nn.Sequential()

        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                               padding_mode=padding_mode)

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)

        self.act_name = None
        act_type = getattr(opts, "model_activation_name", "prelu")

        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model_activation_neg_slope", 0.1)
            inplace = getattr(opts, "model_activation_inplace", False)
            act_layer = get_activation_fn(act_type=act_type,
                                          inplace=inplace,
                                          negative_slope=neg_slope,
                                          num_parameters=out_channels)
            block.add_module(name="act", module=act_layer)

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(self, opts, in_channels: int, out_channels: int, stride: int, expand_ratio: Union[int, float],
                 dilation: int = 1) -> None:

        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 2)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                             module=ConvLayer(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                              use_act=True, use_norm=True))  # conv1x1+bn+relu
        block.add_module(
            name="conv_3x3",
            module=ConvLayer(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3,
                             groups=hidden_dim, use_act=True, use_norm=True))  # DW-conv3x3+bn+relu
        block.add_module(name="red_1x1",
                         module=ConvLayer(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                          use_act=False, use_norm=True))  # conv1x1+bn
        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation

    def forward(self, input, *args, **kwargs):
        if self.use_res_connect:
            return input + self.block(input)
        else:
            return self.block(input)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, opts, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvLayer(opts=opts, in_channels=c1, out_channels=c_, kernel_size=1, stride=1)
        self.cv2 = ConvLayer(opts=opts, in_channels=c_ * 4, out_channels=c2, kernel_size=1, stride=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
