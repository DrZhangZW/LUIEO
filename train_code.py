import torch
import time
import gc
import numpy as np
from torch.cuda.amp import autocast
import matplotlib
import math
import cv2
from PIL import Image
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pathlib
from pathlib import Path
# =================================================
from utils import logger
from utils import ops
from utils import detect_utils
from utils.checkpoint_utils import save_checkpoint
from utils.metrics_utils import Statistics, metric_monitor
from dataloader_wp.Mobile_dataloader import ImageDataset
from model.loss_obj_yolo_used import v8DetectionLoss
from model.loss_enhance import depthLoss

DEFAULT_IMAGE_WIDTH = DEFAULT_IMAGE_HEIGHT = 256
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_LOG_FREQ = 500
DEFAULT_ITERATIONS = 30000
DEFAULT_EPOCHS = 300
DEFAULT_MAX_ITERATIONS = DEFAULT_MAX_EPOCHS = 100000


# =================================================
class Trainer:
    def __init__(self, opts, model,  gradient_scalar, validation_loader, training_loader, optimizer,
                  scheduler,
                 start_epoch: int = 0, start_iteration: int = 0, best_metric: float = 0.0, model_ema=None,
                 *args, **kwargs) -> None:
        self.opts = opts
        self.model = model

        self.gradient_scalar = gradient_scalar
        self.model_ema = model_ema
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.max_iterations = getattr(self.opts, "scheduler_max_iterations", DEFAULT_ITERATIONS)
        # ==============================create the gradient scalar-=============================
        self.training_loader = training_loader
        self.val_loader = validation_loader
        self.training_simulate = ImageDataset(opts, is_training=True)
        self.device = getattr(opts, "dev_device", torch.device("cpu"))
        # ===============================parameters================================================
        self.start_epoch = start_epoch
        self.best_metric = best_metric
        self.train_iterations = start_iteration
        self.log_freq = getattr(self.opts, "common_log_freq", DEFAULT_LOG_FREQ)
        self.accum_freq = getattr(self.opts, "common_accum_freq", 1)
        self.accum_after_epoch = getattr(self.opts, "common_accum_after_epoch", 0)
        self.mixed_precision_training = getattr(opts, "common_mixed_precision", False)
        # =============================Loss function==============
        self.obj_loss_yolo = v8DetectionLoss(opts, self.model)
        self.depth_loss = depthLoss(opts)
        self.classes_yolo = detect_utils.catid2name
        # =============================monitor=============================
        self.metric_names = getattr(opts, "stats_name", ['loss'])
        self.ckpt_metric = getattr(self.opts, "stats_checkpoint_metric", "loss")
        # =============================saved folder=======================
        common_save_dir = getattr(opts, "common_save_dir")
        self.loss_iter = '{}/{}'.format(common_save_dir, "loss_iter.txt")
        self.loss_epoch = '{}/{}'.format(common_save_dir, "loss_epoch.txt")
        self.train_result = '{}/{}'.format(common_save_dir, "weights")
        pathlib.Path(self.train_result).mkdir(parents=True, exist_ok=True)
        with open(self.loss_epoch, 'a') as f:
            f.write('test depth model')
            for k, v in vars(opts).items():
                f.write('\n' f'{k}={v}')

    def run(self, train_sampler):
        keep_k_best_ckpts = getattr(self.opts, "common_k_best_checkpoints", 5)
        ema_best_metric = self.best_metric
        is_ema_best = False
        self.max_epochs = getattr(self.opts, "scheduler_max_epochs", DEFAULT_EPOCHS)
        self.lr_list = []
        for epoch in range(self.start_epoch, self.max_epochs):
            train_sampler.update_scales(epoch=epoch)
            train_loss = self.train_epoch(epoch)
            self.val_detect(epoch)
            gc.collect()
            min_checkpoint_metric = getattr(self.opts, "stats_checkpoint_metric_min", True)
            if min_checkpoint_metric:
                is_best = train_loss <= self.best_metric
                self.best_metric = min(train_loss, self.best_metric)
                save_checkpoint(
                    iterations=self.train_iterations,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    best_metric=self.best_metric,
                    is_best=is_best,
                    save_dir=self.train_result,
                    model_ema=self.model_ema,
                    is_ema_best=is_ema_best,
                    ema_best_metric=ema_best_metric,
                    gradient_scalar=self.gradient_scalar,
                    max_ckpt_metric=min_checkpoint_metric,
                    k_best_checkpoints=keep_k_best_ckpts)
                logger.info('Checkpoints saved at: {}'.format(self.train_result), print_line=True)

    def val_detect(self, epoch,pred_dataloader=None):
        if pred_dataloader==None:
            pred_dataloader=self.val_loader
        time.sleep(2)  # To prevent possible deadlock during epoch transition
        epoch_start_time = time.time()
        batch_load_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        img_num = 0
        results_yolo_total, results_yolo_gt_total = [], []
        self.val_iterations = 0
        task_name = 'object_detection'
        train_stats = Statistics(metric_names=self.metric_names)
        for batch_id, batch in enumerate(pred_dataloader):
            #print(self.val_iterations)
            #if self.val_iterations > 200:
            #   break
            batch_load_toc = time.time() - batch_load_start
            #===================read clean, under_path,under,target=======================
            under_path,under,target_label = batch[1], batch[2], batch[3]
            batch_size = under.shape[0]
            img_num = img_num + batch_size
            print(f"{under.shape}")
            under = under.to(self.device)
            target_label = [{k: v.to(self.device) for k, v in t.items()} for t in target_label]
            #=====================Inference===================================
            time1 = time.time()
            with torch.no_grad():
                outs,pred_label_postprocess, pred_label = self.model(under, 'object_detection') #'object_detection''object_enhancement'
                #pred_label_postprocess, pred_label = self.model_obj(outs['img'])
            time2 = time.time()
            # outs = {'img': x['img']  'back':back,'trans':trans} 'post_process': y, 'detect_pred': x}
            #===================compute loss============================================
            loss_enhance, _ = self.depth_loss(result=outs, depth_gt=under, task=task_name)
            target_label_yolo = self.coco2yolo(target_label)
            loss_obj_yolo = self.obj_loss_yolo(pred_label, target_label_yolo)
            loss_dict = dict()
            loss_dict.update(loss_enhance)
            loss_dict.update(loss_obj_yolo)
            #===============view result=======================================
            pred_clean = outs['img']
            output = {"pred_clean": pred_clean.clamp(0, 1)}
            #====================print metrics =====================================
            metrics_obj = metric_monitor(pred_label=pred_label, target_label=target_label, loss=loss_dict,
                                         metric_names=self.metric_names)
            train_stats.update(metric_vals=metrics_obj, batch_time=batch_load_toc, n=batch_size)
            summary_str_obj = train_stats.iter_summary(epoch=epoch,
                                                       n_processed_samples=self.train_iterations,
                                                       total_samples=self.max_iterations,
                                                       learning_rate=0.0001,
                                                       elapsed_time=epoch_start_time)
            pred_yolo = pred_label_postprocess.detach()
            results_yolo, results_yolo_gt = self.postprocess_yolo(pred_yolo, target_label, conf=0.3)
            results_yolo_total.append(results_yolo)
            results_yolo_gt_total.append(results_yolo_gt)
            #====================plot figs=====================================
            plot_img = [under]
            plot_img += [v for _, v in output.items()]
            _ = self.save_result(mod='val',
                                 epoch=f'{self.val_iterations}',
                                 title='img_pred',
                                 plot_x=plot_img,
                                 plotting=False)

            _, pred_label_img,gt_label_img = self.save_obj_result(mod='val_yolo',
                                                                  epoch=f'{epoch}',
                                                                  title=f'{self.val_iterations}',
                                                                  under=under,
                                                                  clean=output["pred_clean"],
                                                                  results=results_yolo,
                                                                  results_gt=results_yolo_gt,
                                                                  classes=self.classes_yolo,
                                                                  plotting=False,
                                                                  save_img = True)
            """
            cv2.imshow("images", pred_label_img[:, :, ::-1])
            cv2.waitKey(1)
            cv2.imshow("gt_images", gt_label_img[:, :, ::-1])
            cv2.waitKey(1)
            """
            # ===================================================================
            if self.val_iterations % 50 == 0:
                with open(self.loss_iter, 'a') as f:
                    f.write('\n' f'{under.shape}: {summary_str_obj}')

            self.val_iterations = self.val_iterations + 1
            print("inference_time:", time2 - time1)
        print(f"one epoch images: {img_num}")
        if task_name == 'object_detection':
            logger.info('====================yolo_map======================')
            results_yolo_total = [item for sublist in results_yolo_total for item in sublist]
            results_yolo_gt_total = [item for sublist in results_yolo_gt_total for item in sublist]
            detect_utils.val_map_v8(self.train_result, results_yolo_total, results_yolo_gt_total, self.classes_yolo,
                                    self.loss_epoch)

    def train_epoch(self, epoch):
        time.sleep(2)  # To prevent possible deadlock during epoch transition
        epoch_start_time = time.time()
        batch_load_start = time.time()
        if epoch > self.accum_after_epoch or (self.max_epochs - 10) < epoch:
            accum_freq = 5
        else:
            accum_freq = 1
        train_stats = Statistics(metric_names=self.metric_names)
        self.model.train()
        self.optimizer.zero_grad()
        img_num = 0
        period = 80
        num_iter_task = math.floor(epoch / period)
        if epoch % period < (period // 2):
            task_name = 'object_detection'  # 'object_detection''object_enhancement'
            # update the learning rate
            lr_epoch = epoch - (period // 2) * num_iter_task
        else:
            task_name = 'object_enhancement'
            # update the learning rate
            lr_epoch = epoch - (period // 2) * (num_iter_task + 1)
        self.optimizer = self.scheduler.update_lr(optimizer=self.optimizer, epoch=lr_epoch,
                                                  curr_iter=self.train_iterations, task_name=task_name)
        results_yolo_total, results_yolo_gt_total = [], []

        for batch_id, batch in enumerate(self.training_loader):
            # if batch_id >5:
            #    break
            batch_load_toc = time.time() - batch_load_start
            clean, depth, under, target_label = batch[0], batch[1], batch[2], batch[3]

            batch_size = clean.shape[0]
            img_num = img_num + batch_size
            print(f"{clean.shape}")

            loss_dict = dict()
            with autocast(enabled=False):
                if task_name == 'object_detection':
                    target_label = [{k: v.to(self.device) for k, v in t.items()} for t in target_label]
                    under = under.to(self.device)
                    outs,pred_label_postprocess, pred_label = self.model(under,task_name)
                    loss_enhance, output = self.depth_loss(result=outs, depth_gt=under, task=task_name)
                    target_label_yolo = self.coco2yolo(target_label)
                    loss_obj_yolo = self.obj_loss_yolo(pred_label, target_label_yolo)
                    if epoch > 1:
                        losses = (loss_obj_yolo["loss_obj_yolo"]+ 0.005 * loss_enhance["loss_enhance"] ) / accum_freq#
                    else:
                        losses = (loss_obj_yolo["loss_obj_yolo"]) / accum_freq
                    loss_dict.update(loss_obj_yolo)
                    loss_dict.update(loss_enhance)
                if task_name == 'object_enhancement':
                    simulate_info = self.training_simulate.forward_random_parameters(clean, depth)
                    simulate_info = {k: v.to(self.device) for k, v in simulate_info.items()}
                    pred_label = self.model(simulate_info['simulate'], task_name)
                    # pred_label = {'img': x['img']  'back':back,'trans':trans}
                    loss_enhance, output = self.depth_loss(result=pred_label, depth_gt=simulate_info, task=task_name)
                    losses = (loss_enhance["loss_enhance"]) / accum_freq
                    loss_dict.update(loss_enhance)

            self.gradient_scalar.scale(losses).backward()
            if (batch_id + 1) % accum_freq == 0:
                # optimizer_step Perform a single step of the training optimizer with gradient clipping and EMA update.
                max_norm = 0
                if max_norm > 0:
                    self.gradient_scalar.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                # optimizer step
                self.gradient_scalar.step(optimizer=self.optimizer)
                # update the scale for next batch
                self.gradient_scalar.update()
                # set grads to zero
                self.optimizer.zero_grad()
                if self.model_ema is not None:
                    self.model_ema.update_parameters(self.model)
            lr = []
            for param_group in self.optimizer.param_groups:
                lr.append(param_group['lr'])
            metrics_obj = metric_monitor(pred_label=pred_label, target_label=target_label, loss=loss_dict,
                                         metric_names=self.metric_names)
            train_stats.update(metric_vals=metrics_obj, batch_time=batch_load_toc, n=batch_size)
            summary_str_obj = train_stats.iter_summary(epoch=epoch,
                                                       n_processed_samples=self.train_iterations,
                                                       total_samples=self.max_iterations,
                                                       learning_rate=lr,
                                                       elapsed_time=epoch_start_time)
            with open(self.loss_iter, 'a') as f:
                f.write('\n' f'{clean.shape}: {summary_str_obj}')
            if self.train_iterations % 2000 == 0:
                if task_name == 'object_detection':
                    pred_yolo = pred_label_postprocess.detach()
                    results_yolo, results_yolo_gt = self.postprocess_yolo(pred_yolo, target_label, conf=0.1)
                    results_yolo_total.append(results_yolo)
                    results_yolo_gt_total.append(results_yolo_gt)
                    # --------plot figure---------------
                    plot_img = [under]
                    plot_img += [v for _, v in output.items()]
                    _ = self.save_result(mod='train',
                                         epoch=f'{epoch}_{self.train_iterations}',
                                         title='under_pred_clean_trans_back_compute',
                                         plot_x=plot_img,
                                         plotting=True)

                    _, _, _ = self.save_obj_result(mod='train_yolo',
                                                   epoch=f'{epoch}_{self.train_iterations}',
                                                   title='under_yolo_pred_gt_label',
                                                   under=under,
                                                   clean=output['pred_clean'],
                                                   results=results_yolo,
                                                   results_gt=results_yolo_gt,
                                                   classes=self.classes_yolo,
                                                   plotting=True,
                                                   save_img=False)
                else:
                    plot_img = [simulate_info['simulate'], simulate_info['transmission'], simulate_info['backlight']]

                    plot_img += [v for _, v in output.items()]
                    _ = self.save_result(mod='train',
                                         epoch=f'{epoch}_{self.train_iterations}',
                                         title='simulate_pred_clean_trans_back',
                                         plot_x=plot_img,
                                         plotting=True)
            self.train_iterations = self.train_iterations + 1
            batch_load_start = time.time()
        print(f"one epoch images: {img_num}")
        with open(self.loss_epoch, 'a') as f:
            f.write('\n' f'{clean.shape}: {summary_str_obj}')
        self.lr_list.append(lr[0] if len(lr) > 1 else lr)
        if epoch > 5:
            self.plot_lr(self.lr_list)
        """
        if task_name == 'object_detection':
            logger.info('====================yolo_map======================')
            results_yolo_total = [item for sublist in results_yolo_total for item in sublist]
            results_yolo_gt_total = [item for sublist in results_yolo_gt_total for item in sublist]
            detect_utils.val_map_v8(self.train_result, results_yolo_total, results_yolo_gt_total, self.classes_yolo,
                                    self.loss_epoch)
        """
        return losses

    def coco2yolo(self, target_label):

        for tmp_label in target_label:
            nl = tmp_label['boxes'].shape[0]
            tmp_label['batch_idx'] = torch.zeros(nl).cuda()
            tmp_label['cls'] = (tmp_label['labels'][:, None] - 1)
            tmp_label['bboxes'] = tmp_label['boxes']
        new_batch = {}
        keys = target_label[0].keys()
        values = list(zip(*[list(b.values()) for b in target_label]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)

            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)

        return new_batch

    def postprocess_yolo(self, preds, targets, conf, max_det=10):
        preds_yolo = preds.clone()
        # results_gt: boxes: xyxy, scales to orig_size
        # temp_label: boxes like xywh for size of train image
        target_sizes = torch.stack([t["size"] for t in targets], dim=0).detach().cpu()
        temp_sizes = torch.stack([t["size"] for t in targets], dim=0).detach().cpu()
        tgt_bbox, tgt_ids = [], []
        for v in targets:
            tgt_ids.append((v["labels"] - 1).detach().cpu())
            tgt_bbox.append(v["boxes"].detach().cpu())
        results_gt = detect_utils.PostProcess_gt(tgt_ids, tgt_bbox, temp_sizes, target_sizes)

        temp_label = ops.non_max_suppression(preds_yolo, conf_thres=conf, iou_thres=0.5,
                                             labels=[], multi_label=True, agnostic=False, max_det=max_det)

        results = detect_utils.PostProcess_yolo_head(temp_label, max_det=300, temp_sizes=temp_sizes,
                                                     target_sizes=target_sizes, is_plotting=True)
        return results, results_gt

    def plot_lr(self, lr_list):
        x_list = list(range(len(lr_list)))
        fig = plt.figure()
        plt.plot(x_list, lr_list)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.show()
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        f = '{}/{}'.format(self.train_result, f'lr_list.png')
        plt.savefig(f, bbox_inches='tight', dpi=fig.dpi, pad_inches=0)
        plt.close()

    def save_result(self, mod, epoch, title, plot_x, plotting=True):
        # -------------------------------end finetunning and save some results----------------------------------------
        bs, c, h, w = plot_x[0].shape
        num_x = 5 if bs > 6 else bs
        num_y = len(plot_x)
        plot_x_new = []
        plot_x_new += [x.detach().cpu().numpy() for x in plot_x]

        if plotting:
            fig, ax = plt.subplots(num_x, num_y, figsize=(30, 15))
            plt.xticks([])
            plt.yticks([])
        plot_save = []
        for index_x in range(0, num_x):
            for index_y in range(0, num_y):
                tmp_img = self.imagetensor2view(plot_x_new[index_y][index_x])  #
                plot_save.append(tmp_img)
                if plotting:
                    if num_x == 1:
                        ax[index_y].imshow(tmp_img)
                    else:
                        ax[index_x][index_y].imshow(tmp_img)
            # plt.tight_layout(h_pad=0,w_pad=0)
        if plotting:
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
            f = '{}/{}'.format(self.train_result, f'{mod}_{title}_epoch_{epoch}.png')
            plt.title(title, x=-1.4, y=-0.6)
            plt.savefig(f, bbox_inches='tight', dpi=fig.dpi, pad_inches=0)
            plt.close()
        return plot_save

    def save_obj_result(self, mod, epoch, title, under, clean, results, results_gt, classes, plotting=True,save_img=True):
        temp_under = under.detach().cpu().numpy()
        temp_clean = clean.detach().cpu().numpy()
        # classes = detect_utils.categories
        num_img = 5 if len(results_gt) > 6 else len(results_gt)
        # fig, ax = plt.subplots(num_img, 3, figsize=(24, 24))
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        if plotting:
            fig, ax = plt.subplots(num_img, 3)
            plt.tight_layout()

        for index in range(0, num_img):
            temp_clean_view = self.imagetensor2view(temp_clean[index])
            temp_under_view = self.imagetensor2view(temp_under[index])
            #==========================================================================================
            #Due to the deformation of the training data including a horizontal flip, we need to flip it back.
            import torchvision.transforms.functional as F
            temp_clean_view = Image.fromarray(temp_clean_view, mode='RGB')
            temp_under_view = Image.fromarray(temp_under_view, mode='RGB')
            temp_clean_view = F.hflip(temp_clean_view)
            temp_under_view = F.hflip(temp_under_view)
            w, h = temp_clean_view.size
            if "boxes" in results[index]:
                boxes = results[index]["boxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                results[index]["boxes"] = boxes
            if "boxes" in results_gt[index]:
                boxes = results_gt[index]["boxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                results_gt[index]["boxes"] = boxes
            temp_clean_view = np.array(temp_clean_view)
            temp_under_view = np.array(temp_under_view)
            #============================================================================
            temp_pred_label = detect_utils.detect_view(temp_clean_view, results[index]['boxes'],
                                                       results[index]['labels'],
                                                       scores=results[index]['scores'], category_index=classes)

            temp_gt_label = detect_utils.detect_view(temp_under_view, results_gt[index]['boxes'],
                                                     results_gt[index]['labels'],
                                                     scores=results_gt[index]['scores'],
                                                     category_index=classes)
            if save_img:
                image = Image.fromarray(temp_clean_view, mode='RGB')
                image.save('{}/{}'.format(self.train_result,
                                          f'{mod}_iter_{title}_imgID_{index}_pred.png'))  # f'{Path(under_path[0]).stem}.png'
                image = Image.fromarray(temp_pred_label, mode='RGB')
                image.save('{}/{}'.format(self.train_result, f'{mod}_iter_{title}_imgID_{index}_pred_label.png'))
                image = Image.fromarray(temp_gt_label, mode='RGB')
                image.save('{}/{}'.format(self.train_result, f'{mod}_iter_{title}_imgID_{index}_pred_gt.png'))

            if plotting:
                if len(results) == 1:
                    ax[0].imshow(temp_under_view)
                    ax[1].imshow(temp_pred_label)
                    ax[2].imshow(temp_gt_label)
                else:
                    ax[index][0].imshow(temp_under_view)
                    ax[index][1].imshow(temp_pred_label)
                    ax[index][2].imshow(temp_gt_label)
        if plotting:
            f = '{}/{}'.format(self.train_result, f'{mod}_iter_{title}_epoch_{epoch}.png')
            plt.title(title, x=-1.4, y=-0.6)
            plt.savefig(f, bbox_inches='tight', dpi=fig.dpi, pad_inches=0)
            plt.close()

        return temp_clean_view, temp_pred_label, temp_gt_label

    def imagetensor2view(self, imagetesnor):
        channel = imagetesnor.shape[0]
        imagetensor = imagetesnor.transpose((1, 2, 0))
        imagetensor = (imagetensor * 255).astype(np.uint8)

        if channel == 1:
            return imagetensor.squeeze()
        else:
            return imagetensor
