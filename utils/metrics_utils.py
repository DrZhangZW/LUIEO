import sys, time
import numpy as np
import torch
from torch import Tensor
from torch import distributed as dist
from utils import logger
from typing import Optional, Dict, Union, Any, Tuple

class Statistics:
    def __init__(self, metric_names: Optional[list] = ['loss']) -> None:
        if len(metric_names) == 0:
            logger.error('Metric names list cannot be empty')
        # key is the metric name and value is the value
        metric_dict: Dict[str, Union[Any]] = {}
        metric_counters = {}
        for m_name in metric_names:
            metric_dict[m_name] = None
            metric_counters[m_name] = 0
        self.metric_dict = metric_dict
        self.supported_metrics = list(metric_dict.keys())
        self.metric_counters = metric_counters
        self.round_places = 4

        self.batch_time = 0
        self.batch_counter = 0

    def update(self, metric_vals: dict, batch_time: float, n: Optional[int] = 1) -> None:
        for k, v in metric_vals.items():
            if k in self.supported_metrics:
                if self.metric_dict[k] is None:
                    if k == "iou":
                        self.metric_dict[k] = {"inter": v["inter"] * n, "union": v["union"] * n}
                    else:
                        self.metric_dict[k] = v * n
                else:
                    if k == "iou":
                        self.metric_dict[k]["inter"] += (v["inter"] * n)
                        self.metric_dict[k]["union"] += (v["union"] * n)
                    else:
                        self.metric_dict[k] += v * n
                self.metric_counters[k] += n
        self.batch_time += batch_time
        self.batch_counter += 1

    def avg_statistics_all(self, sep=": ") -> list:
        metric_stats = []
        for k, v in self.metric_dict.items():
            if v is not None:
                counter = self.metric_counters[k]
                if k == "iou":
                    inter = (v["inter"] * 1.0) / counter
                    union = (v["union"] * 1.0) / counter
                    iou = inter / union
                    if isinstance(iou, torch.Tensor):
                        iou = iou.cpu().numpy()
                    # Converting iou from [0, 1] to [0, 100]
                    # other metrics are by default in [0, 100 range]
                    v_avg = np.mean(iou) * 100.0
                else:
                    v_avg = (v * 1.0) / counter
                v_avg = round(v_avg, self.round_places)
                metric_stats.append("{:<}{}{:.4f}".format(k, sep, v_avg))

        return metric_stats

    def avg_statistics(self, metric_name: str) -> float:
        avg_val = None
        if metric_name in self.supported_metrics:
            counter = self.metric_counters[metric_name]
            v = self.metric_dict[metric_name]

            if metric_name == "iou":
                inter = (v["inter"] * 1.0) / counter
                union = (v["union"] * 1.0) / counter
                iou = inter / union
                if isinstance(iou, torch.Tensor):
                    iou = iou.cpu().numpy()
                # Converting iou from [0, 1] to [0, 100]
                # other metrics are by default in [0, 100 range]
                avg_val = np.mean(iou) * 100.0
            else:
                avg_val = (v * 1.0) / counter

            avg_val = round(avg_val, self.round_places)
        return avg_val

    def iter_summary(self,
                     epoch: int,
                     n_processed_samples: int,
                     total_samples: int,
                     elapsed_time: float,
                     learning_rate: float or list):
        metric_stats = self.avg_statistics_all()
        el_time_str = "Elapsed time: {:5.2f}".format(time.time() - elapsed_time)
        if isinstance(learning_rate, float):
            lr_str = "LR: {:1.6f}".format(learning_rate)
        else:
            learning_rate = [round(lr, 6) for lr in learning_rate]
            lr_str = "LR: {}".format(learning_rate)
        epoch_str = "Epoch: {:3d} [{:8d}/{:8d}]".format(epoch, n_processed_samples, total_samples)
        batch_str = "Avg. batch load time: {:1.3f}".format(self.batch_time / self.batch_counter)

        stats_summary = [epoch_str]
        stats_summary.append(lr_str)
        stats_summary.extend(metric_stats)
        stats_summary.append(batch_str)
        stats_summary.append(el_time_str)

        summary_str = ", ".join(stats_summary)
        logger.log(summary_str)
        sys.stdout.flush()
        return summary_str

    def epoch_summary(self, epoch: int, stage: Optional[str] = "Training"):
        metric_stats = self.avg_statistics_all(sep="=")
        metric_stats_str = " || ".join(metric_stats)
        logger.log('*** {} summary for epoch {}'.format(stage.title(), epoch))
        print("\t {}".format(metric_stats_str))
        sys.stdout.flush()
        return metric_stats_str

def metric_monitor( pred_label: Tensor or Tuple[Tensor], target_label, loss: Tensor or float,
                   metric_names: list,
                   use_distributed: Optional[bool] = False):
    metric_vals = dict()
    if isinstance(loss, Dict):
        for k, v in loss.items():
            metric_vals[k] = tensor_to_python_float(v, is_distributed=False)
    else:
        loss = tensor_to_python_float(loss, is_distributed=False)
        metric_vals['loss'] = loss
    # if "loss" in metric_names:
    #    loss = tensor_to_python_float(loss["loss_depth"], is_distributed=False)
    #    metric_vals['loss'] = loss
    if "top1" in metric_names:
        top_1_acc, top_5_acc = top_k_accuracy(pred_label, target_label, top_k=(1, 5))
        top_1_acc = tensor_to_python_float(top_1_acc, is_distributed=use_distributed)
        metric_vals['top1'] = top_1_acc
        if "top5" in metric_names:
            top_5_acc = tensor_to_python_float(top_5_acc, is_distributed=use_distributed)
            metric_vals['top5'] = top_5_acc
    if "iou" in metric_names:
        inter, union = compute_miou_batch(prediction=pred_label, target=target_label)
        inter = tensor_to_python_float(inter, is_distributed=use_distributed)
        union = tensor_to_python_float(union, is_distributed=use_distributed)
        metric_vals['iou'] = {'inter': inter, 'union': union}

    return metric_vals
def compute_miou_batch(prediction: Union[Tuple[Tensor, Tensor], Tensor], target: Tensor,
                       epsilon: Optional[float] = 1e-7):
    if isinstance(prediction, Tuple) and len(prediction) == 2:
        mask = prediction[0]
        assert isinstance(mask, Tensor)
    elif isinstance(prediction, Tensor):
        mask = prediction
        assert isinstance(mask, Tensor)
    else:
        raise NotImplementedError(
            "For computing loss for segmentation task, we need prediction to be an instance of Tuple or Tensor")
    num_classes = mask.shape[1]
    pred_mask = torch.max(mask, dim=1)[1]
    assert pred_mask.dim() == 3, "Predicted mask tensor should be 3-dimensional (B x H x W)"

    pred_mask = pred_mask.byte()
    target = target.byte()

    # shift by 1 so that 255 is 0
    pred_mask += 1
    target += 1

    pred_mask = pred_mask * (target > 0)
    inter = pred_mask * (pred_mask == target)
    area_inter = torch.histc(inter.float(), bins=num_classes, min=1, max=num_classes)
    area_pred = torch.histc(pred_mask.float(), bins=num_classes, min=1, max=num_classes)
    area_mask = torch.histc(target.float(), bins=num_classes, min=1, max=num_classes)
    area_union = area_pred + area_mask - area_inter + epsilon
    return area_inter, area_union

def reduce_tensor(inp_tensor: torch.Tensor) -> torch.Tensor:
    size = float(dist.get_world_size())
    inp_tensor_clone = inp_tensor.clone()
    dist.barrier()
    dist.all_reduce(inp_tensor_clone, op=dist.ReduceOp.SUM)
    inp_tensor_clone /= size
    return inp_tensor_clone


def tensor_to_python_float(inp_tensor: Union[int, float, torch.Tensor],
                           is_distributed: bool) -> Union[int, float, np.ndarray]:
    if is_distributed and isinstance(inp_tensor, torch.Tensor):
        inp_tensor = reduce_tensor(inp_tensor=inp_tensor)

    if isinstance(inp_tensor, torch.Tensor) and inp_tensor.numel() > 1:
        # For IOU, we get a C-dimensional tensor (C - number of classes)
        # so, we convert here to a numpy array
        return inp_tensor.cpu().numpy()
    elif hasattr(inp_tensor, 'item'):
        return inp_tensor.item()
    elif isinstance(inp_tensor, (int, float)):
        return inp_tensor * 1.0
    else:
        raise NotImplementedError("The data type is not supported yet in tensor_to_python_float function")
def top_k_accuracy(output: Tensor, target: Tensor, top_k: Optional[tuple] = (1,)) -> list:
    maximum_k = max(top_k)
    batch_size = target.shape[0]

    _, pred = output.topk(maximum_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(
        target.reshape(1, -1).expand_as(pred)
    )

    results = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_k = correct_k.mul_(100.0 / batch_size)
        results.append(acc_k)
    return results