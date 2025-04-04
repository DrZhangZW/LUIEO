from typing import Dict
from utils import logger


def get_configuration(opts) -> Dict:
    mode = getattr(opts, "model_classification_mit_mode", "small")
    if mode is None:
        logger.error("Please specify mode")

    head_dim = getattr(opts, "model_classification_mit_head_dim", None)
    num_heads = getattr(opts, "model_classification_mit_number_heads", None)
    if head_dim is not None:
        if num_heads is not None:
            logger.error(
                "--model.classification.mit.head-dim and --model.classification.mit.number-heads "
                "are mutually exclusive."
            )
    elif num_heads is not None:
        if head_dim is not None:
            logger.error(
                "--model.classification.mit.head-dim and --model.classification.mit.number-heads "
                "are mutually exclusive."
            )
    mode = mode.lower()

    if mode == "small_v3":
        mv2_exp_mult = 2
        # image size 256x256
        config = {
            "layer1": {"out_channels": 32, "expand_ratio": mv2_exp_mult, "num_blocks": 3, "stride": 2,
                       "block_type": "mv2"},  # (Input,output) = (16,32) 128x128
            "layer2": {"out_channels": 64, "expand_ratio": mv2_exp_mult, "num_blocks": 3, "stride": 2,
                       "block_type": "mv2"},  # (Input,output) = (32,64) 56x56
            "layer3": {"out_channels": 128, "transformer_channels": 144, "ffn_dim": 288, "transformer_blocks": 2,
                       "patch_h": 2, "patch_w": 2,
                       "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": head_dim, "num_heads": num_heads,
                       "block_type": "mobilevit"},  # (Input,output) = (64,128) 28x28
            "layer4": {"out_channels": 256, "transformer_channels": 192, "ffn_dim": 384, "transformer_blocks": 4,
                       "patch_h": 2, "patch_w": 2,
                       "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": head_dim, "num_heads": num_heads,
                       "block_type": "mobilevit"},  # (Input,output) = (128,256) 14x14
            "layer5": {"out_channels": 320, "transformer_channels": 240, "ffn_dim": 480, "transformer_blocks": 3,
                       "patch_h": 2, "patch_w": 2,
                       "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": head_dim, "num_heads": num_heads,
                       "block_type": "mobilevit"},  # (Input,output) = (256,320) 7x7
            "last_layer_exp_factor": 4,
            "layer4_1": {"out_channels": 256, "transformer_channels": 192, "ffn_dim": 384, "transformer_blocks": 4,
                         "patch_h": 2, "patch_w": 2,
                         "stride": 1, "mv_expand_ratio": 1, "head_dim": head_dim, "num_heads": num_heads,
                         "block_type": "mobilevit"},  # (Input,output) = (256,256)  # 14x14
            "layer3_1": {"out_channels": 128, "transformer_channels": 144, "ffn_dim": 288, "transformer_blocks": 2,
                         "patch_h": 2, "patch_w": 2,
                         "stride": 1, "mv_expand_ratio": 1, "head_dim": head_dim, "num_heads": num_heads,
                         "block_type": "mobilevit"},  # (Input,output) = (128,128) 28x28
            "layer2_1": {"out_channels": 64, "expand_ratio": mv2_exp_mult, "num_blocks": 3, "stride": 1,
                         "block_type": "mv2"},  # (Input,output) = (64,64) 56x56
            "layer1_1": {"out_channels": 32, "expand_ratio": mv2_exp_mult, "num_blocks": 3, "stride": 1,
                         "block_type": "mv2"},  # (Input,output) = (32,32)128x128
            "layer0_1": {"out_channels": 16, "expand_ratio": mv2_exp_mult, "num_blocks": 3, "stride": 1,
                         "block_type": "mv2"},  # (Input,output) = (16,16) 128x128
            "layer0_img": {"out_channels": 3, "expand_ratio": 1, "num_blocks": 3, "stride": 1,
                         "block_type": "mv2"},  # (Input,output) = (16,16) 128x128
            "layer0_back": {"out_channels": 3, "expand_ratio": 1, "num_blocks": 3, "stride": 1,
                         "block_type": "mv2"},  # (Input,output) = (16,16) 128x128
            "layer0_trans": {"out_channels": 3, "expand_ratio": 1, "num_blocks": 3, "stride": 1,
                         "block_type": "mv2"},  # (Input,output) = (16,16) 128x128
            "head3_1": {"out_channels": 128, "expand_ratio": 1, "num_blocks": 2, "stride": 1,
                        "block_type": "mv2"},  # (Input=layer3_1,output) = (128x2+256,256) 128x128
            "head4_1": {"out_channels": 256, "expand_ratio": 1, "num_blocks": 2, "stride": 1,
                        "block_type": "mv2"},  # (Input=layer3_1,output) = (128x2+256,256) 128x128
            "head5_1": {"out_channels": 320, "expand_ratio": 1, "num_blocks": 2, "stride": 1,
                        "block_type": "mv2"},  # (Input,output) = (320+1024,1024) 128x128
        }
    else:
        raise NotImplementedError
    return config
