import torch
from torch import nn, Tensor
from model.model_config import get_configuration
from model.mobilevit_module import ConvLayer, InvertedResidual, MobileViTBlock, SPPF
from utils import logger
from typing import Optional, Tuple, Dict
import math

class mobile_model(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        image_channels = 3
        # --------------------for yolov8n model-backbone---------
        self.backbone = backbone
        self.neck = neck
        self.head = head
        m = self.head.Detect_head  # Detect()
        s = 256  # 2x min stride
        m.inplace = True
        forward = lambda x, y: self.forward(x, y)
        _, _, x = forward(torch.zeros(1, image_channels, s, s), 'object_detection')
        m.stride = torch.tensor([s / x_tmp.shape[-2] for x_tmp in x])  # forward

        m.bias_init()  # only run once
        # Init weights, biases
        initialize_weights(self)

    def forward(self, x, task_name):
        outs = self.backbone(x)
        outs_decoder = self.neck(outs)
        outs_put = self.head(outs_decoder, task_name)
        return outs_put


class mobile_backbone(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        image_channels = 3
        out_channels = 16
        model_cfg = get_configuration(opts=opts)
        # We allow that using `output_stride` arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True
        self.dilation = 1
        self.conv_1 = ConvLayer(opts=opts, in_channels=image_channels, out_channels=out_channels,
                                kernel_size=3, stride=1, use_norm=True, use_act=True)  # downsaple
        self.layer_1, out_channels = self._make_layer(opts=opts, input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(opts=opts, input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(opts=opts, input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(opts=opts, input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(opts=opts, input_channel=out_channels, cfg=model_cfg["layer5"])
        self.SPPF = SPPF(opts=opts, c1=out_channels, c2=out_channels)
        self.num_channels = out_channels

    def forward(self, x: Tensor):
        outs = []
        x = self.conv_1(x)
        outs.append(x)
        x = self.layer_1(x)
        outs.append(x)
        x = self.layer_2(x)
        outs.append(x)
        x = self.layer_3(x)
        outs.append(x)
        x = self.layer_4(x)
        outs.append(x)
        x = self.layer_5(x)
        x = self.SPPF(x)
        outs.append(x)
        return outs

    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[
        nn.Sequential, int]:
        block_type = cfg.get("block_type", "../mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate)
        else:
            return self._make_mobilenet_layer(opts=opts, input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
            layer = InvertedResidual(opts=opts, in_channels=input_channel, out_channels=output_channels,
                                     stride=stride, expand_ratio=expand_ratio)
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[
        nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(opts=opts, in_channels=input_channel, out_channels=cfg.get("out_channels"),
                                     stride=stride, expand_ratio=cfg.get("mv_expand_ratio", 4), dilation=prev_dilation)
            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            logger.error("Transformer input dimension should be divisible by head dimension. "
                         "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(
            MobileViTBlock(opts=opts, in_channels=input_channel, transformer_dim=transformer_dim, ffn_dim=ffn_dim,
                           n_transformer_blocks=cfg.get("transformer_blocks", 1),
                           patch_h=cfg.get("patch_h", 2),
                           patch_w=cfg.get("patch_w", 2),
                           dropout=getattr(opts, "model_classification_mit_dropout", 0.1),
                           ffn_dropout=getattr(opts, "model_classification_mit_ffn_dropout", 0.0),
                           attn_dropout=getattr(opts, "model_classification_mit_attn_dropout", 0.1),
                           head_dim=head_dim,
                           no_fusion=getattr(opts, "model_classification_mit_no_fuse_local_global_features", False),
                           conv_ksize=getattr(opts, "model_classification_mit_conv_kernel_size", 3)))
        return nn.Sequential(*block), input_channel


class mobile_neck(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        # backbone feats: x0,x1,x2,x3,x4,x5
        self.dilation = 1
        model_cfg = get_configuration(opts=opts)
        # We allow that using `output_stride` arguments
        self.up = nn.Upsample(*[None, 2, 'nearest'])
        self.concat = Concat()
        self.proj_p5_p4 = ConvLayer(opts=opts,
                                    in_channels=model_cfg["layer5"]["out_channels"],
                                    out_channels=model_cfg["layer4"]["out_channels"],
                                    kernel_size=1, stride=1, use_norm=True, use_act=True)
        self.mv2_p5_p4, _ = self._make_layer(opts=opts, input_channel=model_cfg["layer4"]["out_channels"],
                                             cfg=model_cfg["layer4_1"])

        self.proj_p4_p3 = ConvLayer(opts=opts,
                                    in_channels=model_cfg["layer4"]["out_channels"],
                                    out_channels=model_cfg["layer3"]["out_channels"],
                                    kernel_size=1, stride=1, use_norm=True, use_act=True)
        self.mv2_p4_p3, _ = self._make_layer(opts=opts, input_channel=model_cfg["layer3"]["out_channels"],
                                             cfg=model_cfg["layer3_1"])

        self.proj_p3_p2 = ConvLayer(opts=opts,
                                    in_channels=model_cfg["layer3"]["out_channels"],
                                    out_channels=model_cfg["layer2"]["out_channels"],
                                    kernel_size=1, stride=1, use_norm=True, use_act=True)
        self.mv2_p3_p2, _ = self._make_layer(opts=opts, input_channel=model_cfg["layer2"]["out_channels"],
                                             cfg=model_cfg["layer2_1"])

        self.proj_p2_p1 = ConvLayer(opts=opts,
                                    in_channels=model_cfg["layer2"]["out_channels"],
                                    out_channels=model_cfg["layer1"]["out_channels"],
                                    kernel_size=1, stride=1, use_norm=True, use_act=True)
        self.mv2_p2_p1, _ = self._make_layer(opts=opts, input_channel=model_cfg["layer1"]["out_channels"],
                                             cfg=model_cfg["layer1_1"])

        self.proj_p1_p0 = ConvLayer(opts=opts,
                                    in_channels=model_cfg["layer1"]["out_channels"],
                                    out_channels=16,
                                    kernel_size=1, stride=1, use_norm=True, use_act=True)
        self.mv2_p1_p0, _ = self._make_layer(opts=opts, input_channel=model_cfg["layer1"]["out_channels"],
                                             cfg=model_cfg["layer0_1"])
        # self.mv2_p0_img = C2f(*[32,3,1,True])
        #self.mv2_p0_back = C2f(*[32, 3, 1, True])
        #self.mv2_p0_trans = C2f(*[32, 3, 1, True])
        layer0_img_1 = {"out_channels": 16, "expand_ratio": 1, "num_blocks": 2, "stride": 1, "block_type": "mv2"}
        block_img, block_trans, blcok_back = [], [], []
        block_img.append(self._make_layer(opts=opts, input_channel=16 + 16, cfg=layer0_img_1)[0])
        block_img.append(ConvLayer(opts=opts, in_channels=16, out_channels=3,
                                   kernel_size=1, stride=1, use_norm=True, use_act=True))
        self.mv2_p0_img = nn.Sequential(*block_img)

        blcok_back.append(self._make_layer(opts=opts, input_channel=16 + 16, cfg=layer0_img_1)[0])
        blcok_back.append(ConvLayer(opts=opts, in_channels=16, out_channels=3,
                                   kernel_size=1, stride=1, use_norm=True, use_act=True))
        self.mv2_p0_back = nn.Sequential(*blcok_back)

        block_trans.append(self._make_layer(opts=opts, input_channel=16 + 16, cfg=layer0_img_1)[0])
        block_trans.append(ConvLayer(opts=opts, in_channels=16, out_channels=3,
                                   kernel_size=1, stride=1, use_norm=True, use_act=True))
        self.mv2_p0_trans = nn.Sequential(*block_trans)


    def forward(self, x):
        # x: backbone features with [None,None,P3,P4,P5]
        # step 10, 11, 12
        P0, P1, P2, P3, P4, P5 = x[-6], x[-5], x[-4], x[-3], x[-2], x[-1]
        feat4 = P4 + self.proj_p5_p4(self.up(P5))  # self.proj_p5_p4(P4) +self.up(P5)
        feat4 = self.mv2_p5_p4(feat4)

        feat3 = P3 + self.proj_p4_p3(self.up(feat4))  # self.proj_p4_p3(P3) +self.up(P4)
        feat3 = self.mv2_p4_p3(feat3)

        feat2 = P2 + self.proj_p3_p2(self.up(feat3))
        feat2 = self.mv2_p3_p2(feat2)

        feat1 = P1 + self.proj_p2_p1(self.up(feat2))
        feat1 = self.mv2_p2_p1(feat1)

        feat0 = self.concat([P0, self.proj_p1_p0(self.up(feat1))])

        img = self.mv2_p0_img(feat0)
        back = self.mv2_p0_back(feat0)
        trans = self.mv2_p0_trans(feat0)


        outs = {'img': img, 'back': back, 'trans': trans, 'detect_x_small': feat2, 'detect_small': feat3,
                'detect_medium': feat4,
                'detect_large': P5}
        return outs

    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[
        nn.Sequential, int]:
        block_type = cfg.get("block_type", "../mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate)
        else:
            return self._make_mobilenet_layer(opts=opts, input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
            layer = InvertedResidual(opts=opts, in_channels=input_channel, out_channels=output_channels,
                                     stride=stride, expand_ratio=expand_ratio)
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[
        nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(opts=opts, in_channels=input_channel, out_channels=cfg.get("out_channels"),
                                     stride=stride, expand_ratio=cfg.get("mv_expand_ratio", 4), dilation=prev_dilation)
            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            logger.error("Transformer input dimension should be divisible by head dimension. "
                         "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(
            MobileViTBlock(opts=opts, in_channels=input_channel, transformer_dim=transformer_dim, ffn_dim=ffn_dim,
                           n_transformer_blocks=cfg.get("transformer_blocks", 1),
                           patch_h=cfg.get("patch_h", 2),
                           patch_w=cfg.get("patch_w", 2),
                           dropout=getattr(opts, "model_classification_mit_dropout", 0.1),
                           ffn_dropout=getattr(opts, "model_classification_mit_ffn_dropout", 0.0),
                           attn_dropout=getattr(opts, "model_classification_mit_attn_dropout", 0.1),
                           head_dim=head_dim,
                           no_fusion=getattr(opts, "model_classification_mit_no_fuse_local_global_features", False),
                           conv_ksize=getattr(opts, "model_classification_mit_conv_kernel_size", 3)))
        return nn.Sequential(*block), input_channel


class mobile_head(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.nc = 10
        # backbone feats: x0,x1,x2,x3,x4,x5
        self.dilation = 1
        self.concat = Concat()
        model_cfg = get_configuration(opts=opts)

        self.input_proj_3 = ConvLayer(opts=opts, in_channels=model_cfg["layer2_1"]["out_channels"],
                                      out_channels=model_cfg["layer3_1"]["out_channels"],
                                      kernel_size=3, stride=2, use_norm=True, use_act=True)  # downsaple
        self.final_conv_3, _ = self._make_layer(opts=opts, input_channel=2 * model_cfg["layer3_1"]["out_channels"],
                                                cfg=model_cfg["head3_1"])  # 18 (P4/16-medium)

        self.input_proj_4 = ConvLayer(opts=opts, in_channels=model_cfg["layer3_1"]["out_channels"],
                                      out_channels=model_cfg["layer4_1"]["out_channels"],
                                      kernel_size=3, stride=2, use_norm=True, use_act=True)  # downsaple
        self.final_conv_4, _ = self._make_layer(opts=opts, input_channel=2 * model_cfg["layer4_1"]["out_channels"],
                                                cfg=model_cfg["head4_1"])  # 18 (P4/16-medium)

        self.input_proj_5 = ConvLayer(opts=opts, in_channels=model_cfg["head4_1"]["out_channels"],
                                      out_channels=model_cfg["layer5"]["out_channels"],
                                      kernel_size=3, stride=2, use_norm=True, use_act=True)  # downsaple
        self.final_conv_5, _ = self._make_layer(opts=opts, input_channel=2 * model_cfg["layer5"]["out_channels"],
                                                cfg=model_cfg["head5_1"])  # (P5/32-large)

        self.Detect_head = Detect(self.nc, [model_cfg["layer2_1"]["out_channels"],
                                            model_cfg["head3_1"]["out_channels"],
                                            model_cfg["head4_1"]["out_channels"],
                                            model_cfg["head5_1"]["out_channels"]])

    def forward(self, x, task_name):
        # x = {'detect_small':feat3,'detect_medium':feat4,'detect_large':P5}
        # downsample: feat3: 80x80, feat4: 40x40, P5: 20x20
        if task_name == 'object_detection':
            x_small, feat3, feat4, feat5 = x['detect_x_small'], x['detect_small'], x['detect_medium'], x['detect_large']
            small = self.final_conv_3(self.concat([feat3, self.input_proj_3(x_small)]))
            medium = self.final_conv_4(self.concat([feat4, self.input_proj_4(small)]))
            large = self.final_conv_5(self.concat([feat5, self.input_proj_5(medium)]))
            result = [x_small, small, medium, large]
            y, y_loss = self.Detect_head(result)
            return {'img': x['img'], 'back': x['back'], 'trans': x['trans']}, y, y_loss
        else:
            return {'img': x['img'], 'back': x['back'], 'trans': x['trans']}

    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[
        nn.Sequential, int]:
        block_type = cfg.get("block_type", "../mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate)
        else:
            return self._make_mobilenet_layer(opts=opts, input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
            layer = InvertedResidual(opts=opts, in_channels=input_channel, out_channels=output_channels,
                                     stride=stride, expand_ratio=expand_ratio)
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[
        nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(opts=opts, in_channels=input_channel, out_channels=cfg.get("out_channels"),
                                     stride=stride, expand_ratio=cfg.get("mv_expand_ratio", 4), dilation=prev_dilation)
            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            logger.error("Transformer input dimension should be divisible by head dimension. "
                         "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(
            MobileViTBlock(opts=opts, in_channels=input_channel, transformer_dim=transformer_dim, ffn_dim=ffn_dim,
                           n_transformer_blocks=cfg.get("transformer_blocks", 1),
                           patch_h=cfg.get("patch_h", 2),
                           patch_w=cfg.get("patch_w", 2),
                           dropout=getattr(opts, "model_classification_mit_dropout", 0.1),
                           ffn_dropout=getattr(opts, "model_classification_mit_ffn_dropout", 0.0),
                           attn_dropout=getattr(opts, "model_classification_mit_attn_dropout", 0.1),
                           head_dim=head_dim,
                           no_fusion=getattr(opts, "model_classification_mit_no_fuse_local_global_features", False),
                           conv_ksize=getattr(opts, "model_classification_mit_conv_kernel_size", 3)))
        return nn.Sequential(*block), input_channel


def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #pass
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


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


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max =1# DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # if self.training:
        #    return x
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, x)
        # return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')  # if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


if __name__ == '__main__':
    import argparse
    from utils.common_utils import load_config_file

    parser = argparse.ArgumentParser(description='Training arguments', add_help=True)
    parser.add_argument('--common_config_file', type=str,
                        default=r'E:\new python\MobileVIT_WP2\mobilevitv3_small_multiserver.yaml',
                        help="Configuration file")
    opts = parser.parse_args()
    opts = load_config_file(opts)
    model_backbone = mobile_backbone(opts=opts)
    model_neck = mobile_neck(opts=opts)
    model_head = mobile_head(opts=opts)
    model = mobile_model(model_backbone, model_neck, model_head)
    x = torch.rand((2, 3, 128, 128))
