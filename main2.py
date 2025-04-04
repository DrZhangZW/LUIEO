import torch
import argparse
from torch.cuda.amp import GradScaler
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
from train_code import Trainer
from utils import logger
from utils.common_utils import device_setup, increment_path, load_config_file
from utils.checkpoint_utils import EMA, load_checkpoint
from model.mobilevit_model import mobile_model, mobile_backbone, mobile_neck, mobile_head
from model.scheduler_cos import CosineScheduler
from dataloader_wp.Mobile_dataloader import get_train_val_dataset


def main(opts):
    device = getattr(opts, "dev_device", torch.device('cpu'))
    # -------------setting dataset and model-------------
    train_loader, train_sampler, val_loader,pred_loader = get_train_val_dataset(opts)
    backbone = mobile_backbone(opts)
    neck = mobile_neck(opts)
    head = mobile_head(opts)
    model = mobile_model(backbone, neck, head).to(device=device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # -------------setting learning rate and optim-------------
    lr = getattr(opts, "scheduler_lr", 1e-4)
    weight_decay = getattr(opts, "optim_weight_decay", 4e-5)  # 1e-4
    beta1 = getattr(opts, "optim_adamw_beta1", 0.9)
    beta2 = getattr(opts, "optim_adamw_beta2", 0.98)
    # param_dicts = [p for p in model.parameters() if p.requires_grad]
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad],
         "lr": lr, },
        {"params": [p for n, p in model.named_parameters() if "head" in n and p.requires_grad],
         "lr": lr, },
    ]
    optimizer = torch.optim.AdamW(param_dicts, betas=(beta1, beta2), weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,100)
    max_iterations = len(train_loader)
    setattr(opts, "scheduler_max_iterations", max_iterations)
    warn_iterations = int(max_iterations)
    setattr(opts, "scheduler_warmup_iterations", warn_iterations)
    lr_scheduler = CosineScheduler(opts)
    # ----------------setting ema model-----------------------
    model_ema = None
    use_ema = getattr(opts, "ema_enable", False)
    if use_ema:
        ema_momentum = getattr(opts, "ema_momentum", 0.0001)
        model_ema = EMA(model=model, ema_momentum=ema_momentum, device=device)
        logger.log('Using EMA')
    gradient_scalar = GradScaler(enabled=getattr(opts, "common_mixed_precision", True))
    pretrained = getattr(opts, "model_pretrained", None)
    start_epoch, start_iteration, best_metric = 0, 0, 2
    if pretrained != "None":
        model, optimizer, gradient_scalar, start_epoch, start_iteration, best_metric, model_ema = load_checkpoint(
            opts=opts,
            model=model,
            optimizer=optimizer,
            gradient_scalar=gradient_scalar,
            model_ema=model_ema)

    training_engine = Trainer(opts=opts,
                              model=model,
                              gradient_scalar=gradient_scalar,
                              training_loader=train_loader,
                              validation_loader=val_loader,
                              optimizer=optimizer,
                              scheduler=lr_scheduler,
                              start_epoch=start_epoch,
                              start_iteration=start_iteration,
                              best_metric=best_metric,
                              model_ema=model_ema)
    training_engine.val_detect(start_epoch)       #for validation or pred images with batch size =2
    training_engine.run(train_sampler)                                 #for training



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments', add_help=True)
    parser.add_argument('--common_config_file', type=str,
                        default=ROOT / "mobilevitv3_small_multiserver.yaml",
                        help="Configuration file")
    opts = parser.parse_args()
    opts = load_config_file(opts)
    opts = device_setup(opts)
    save_dir =   str(increment_path(Path(opts.common_project) / opts.common_name))

    num_gpus = getattr(opts, "dev_num_gpus", 1)
    world_size = getattr(opts, "ddp_world_size", -1)
    use_distributed = getattr(opts, "ddp_enable", False)
    if num_gpus <= 1:
        use_distributed = False
    setattr(opts, "dev_device_id", None)
    setattr(opts, "common_save_dir", save_dir)
    setattr(opts, "ddp_use_distributed", use_distributed)
    # for k, v in vars(opts).items():
    #    logger.info( (f'{k}={v}'))
    main(opts=opts)
