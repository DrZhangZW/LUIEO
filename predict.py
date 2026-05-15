import os
import torch
import argparse
import glob
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torch.cuda.amp import GradScaler
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# 你的依赖
from utils import logger, ops, detect_utils
from utils.common_utils import device_setup, load_config_file
from utils.checkpoint_utils import load_checkpoint
from model.mobilevit_model import mobile_model, mobile_backbone, mobile_neck, mobile_head
from dataloader_wp.detect_data_transform import T

# 预处理
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ===================== 【1:1 复制你自己的 postprocess_yolo】 =====================
def postprocess_yolo(preds, targets, conf, max_det=10):
    preds_yolo = preds.clone()
    target_sizes = torch.stack([t["size"] for t in targets], dim=0).detach().cpu()
    temp_sizes = torch.stack([t["size"] for t in targets], dim=0).detach().cpu()
    tgt_bbox, tgt_ids = [], []
    for v in targets:
        tgt_ids.append((v["labels"] - 1).detach().cpu())
        tgt_bbox.append(v["boxes"].detach().cpu())
    results_gt = detect_utils.PostProcess_gt(tgt_ids, tgt_bbox, temp_sizes, target_sizes)

    # 🔥 🔥 🔥 用你自己的 utils.ops！！！！！！
    temp_label = ops.non_max_suppression(preds_yolo, conf_thres=conf, iou_thres=0.5,
                                         labels=[], multi_label=True, agnostic=False, max_det=max_det)
    results = detect_utils.PostProcess_yolo_head(temp_label, max_det=300, temp_sizes=temp_sizes,
                                                 target_sizes=target_sizes, is_plotting=True)
    return results, results_gt


# ===================== 【你原版增强代码，一个字不动】 =====================
def run_predict(opts, model, save_dir="predict_output"):
    os.makedirs(save_dir, exist_ok=True)
    device = opts.dev_device
    img_dir = opts.dataset_root_path + opts.dataset_pred_under

    img_paths = glob.glob(os.path.join(img_dir, "*.*"))
    img_paths = [f for f in img_paths if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]

    model.train()

    for path in img_paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        name = Path(path).stem

        # ===================== 双图 + 翻转 trick（完全不动） =====================
        img_ori = transform(img).unsqueeze(0)
        img_flip = transform(img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)
        x = torch.cat([img_ori, img_flip], dim=0).to(device)

        with torch.no_grad():
            outs, pred_label_postprocess, pred_label = model(x, 'object_detection')

        # ===================== 增强图保存（完全不动） =====================
        enhanced = outs["img"]
        enhanced[1] = torch.flip(enhanced[1], dims=[-1])
        final_enhanced = enhanced.mean(0, keepdim=True).clamp(0, 1)
        enhanced_np = final_enhanced[0].permute(1, 2, 0).cpu().numpy() * 255
        enhanced_img = Image.fromarray(enhanced_np.astype('uint8'))
        enhanced_img = F.hflip(enhanced_img)
        enhanced_img.save(f"{save_dir}/{name}_enhanced.png")

        # ===================== 【1:1 复制你自己的画框逻辑】 =====================
        try:
            pred_yolo = pred_label_postprocess[0:1].detach()
            fake_targets = [{"size": torch.tensor([h, w]), "boxes": torch.zeros(1, 4), "labels": torch.zeros(1)}]
            results_yolo, _ = postprocess_yolo(pred_yolo, fake_targets, conf=0.2)

            res = results_yolo[0]
            boxes = res["boxes"][:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])

            det_img = np.array(enhanced_img)
            det_img = detect_utils.detect_view(
                det_img, boxes, res["labels"], scores=res["scores"],
                category_index=detect_utils.catid2name
            )
            Image.fromarray(det_img).save(f"{save_dir}/{name}_detect.png")

        except Exception as e:
            print("画框失败：", e)


        print(f"✅ 已保存: {name}")


# ===================== 主函数（完全不动） =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--common_config_file', default=ROOT / "mobilevitv3_small_multiserver.yaml")
    parser.add_argument("--dataset_root_path", default="E:/Python-master/dataset")
    parser.add_argument("--dataset_pred_under", default="/RUOD/val/images")
    parser.add_argument("--model_pretrained", default="./runs/train/exp1/weights/checkpoint648.pt")

    predict_opts = parser.parse_args()
    predict_opts = load_config_file(predict_opts)
    predict_opts = device_setup(predict_opts)
    setattr(predict_opts, "dev_device_id", None)
    setattr(predict_opts, "ddp_use_distributed", False)

    device = predict_opts.dev_device
    model = mobile_model(mobile_backbone(predict_opts), mobile_neck(predict_opts), mobile_head(predict_opts)).to(device)

    lr = 1e-4
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad], "lr": lr},
        {"params": [p for n, p in model.named_parameters() if "head" in n and p.requires_grad], "lr": lr},
    ]
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=4e-5)
    gradient_scalar = GradScaler(enabled=False)

    model, *_ = load_checkpoint(predict_opts, model, optimizer, gradient_scalar, None)
    run_predict(predict_opts, model)