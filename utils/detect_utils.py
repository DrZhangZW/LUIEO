import os
from dataloader_wp import detect_data_transform as T
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from torchvision.ops.boxes import batched_nms,nms
from PIL import Image, ImageDraw, ImageFont,ImageColor
from utils import logger
from utils.box_ops import box_cxcywh_to_xyxy
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

categories  = ['N/A','holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle','jellyfish']
categories = {k: v for k, v in enumerate(categories) }
clsid2catid={0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10,10:11}
catid2name = { 0:'holothurian',1: 'echinus', 2:'scallop',
                3: 'starfish',4: 'fish', 5: 'corals',
                6: 'diver', 7: 'cuttlefish',
                8: 'turtle',9:'jellyfish'}
"""
categories_id = 0
for index in range(0,len(class_names)):
    categories_id = categories_id + 1
    categories.append({'id': categories_id,'name':class_names[index]} )
"""
#==================================labels processing=======================================
def PostProcess(out_logits, out_bbox, target_sizes,confidence=0,apply_nms=True,max_det=30):
    """ Perform the computation
    Parameters:
        out_logits: bs x num_queries x num_classes
        out_bbox: bs x num_querise x 4
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x w x h] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
            target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    image = Image.open('../input/mydesktop/MYDesktop3.jpg')
    plt.imshow(image)
    plt.show()
    """
    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    # convert to [x0, y0, x1, y1] format
    boxes = box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cpu()
    boxes = boxes * scale_fct[:, None, :]

    bs = prob.shape[0]
    results = []
    for index in range(0,bs):
        tmp_scores = scores[index]
        keep = tmp_scores>confidence
        tmp_scores = tmp_scores[keep]
        tmp_class = labels[index][keep]
        tmp_boxes =boxes[index][keep]
        if apply_nms:
            keep = batched_nms(tmp_boxes, tmp_scores, tmp_class, 0.5)
            tmp_scores = tmp_scores[keep]
            tmp_class = tmp_class[keep]
            tmp_boxes = tmp_boxes[keep]
        tmp_scores = tmp_scores[:max_det]
        tmp_class = tmp_class[:max_det]
        tmp_boxes = tmp_boxes[:max_det,:]
        results.append({'scores': tmp_scores, 'labels':tmp_class, 'boxes':tmp_boxes})
        """
        if apply_nms:
            top_scores, labels = scores.max(-1)
            keep = batched_nms(boxes, top_scores, labels, iou)
            scores, boxes = scores[keep], boxes[keep]
        """
    #results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results
def PostProcess_gt(out_logits, out_bbox, temp_sizes,target_sizes):
    """ Perform the computation
    Parameters:
        out_logits: list[image1_logits,image2_logits]
        out_bbox: list[image1_box,image2_box]
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x w x h] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
            target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    image = Image.open('../input/mydesktop/MYDesktop3.jpg')
    plt.imshow(image)
    plt.show()
    """
    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2
    prob = out_logits #F.softmax(out_logits, -1)
    scores, labels = prob, prob
    # convert to [x0, y0, x1, y1] format
    bs = len(prob)
    results = []
    for index in range(0, bs):
        tmp_scores = scores[index]
        tmp_scores = tmp_scores
        tmp_class = labels[index]
        boxes = box_cxcywh_to_xyxy(out_bbox[index])
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = temp_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cpu()
        boxes = boxes * scale_fct[index]

        shape = target_sizes[index].cpu().numpy()
        shape1=temp_sizes[index].cpu().numpy()
        scale_coords([shape1[0], shape1[1]], boxes, [shape[0], shape[1]])

        results.append({'scores': tmp_scores, 'labels': tmp_class, 'boxes': boxes})
    #results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results

def PostProcess_yolo_head(output, max_det=300,temp_sizes=None,target_sizes=None,is_plotting=True,apply_nms=True):
    """
    output boxes with form xyxy
    Convert model output to target format [batch_id, class_id, x, y, x, y, conf] for plotting.
    """
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        #box=box_cxcywh_to_xyxy(box)
        shape = target_sizes[i].cpu().numpy()
        shape1=temp_sizes[i].cpu().numpy()
        scale_coords([shape1[0], shape1[1]], box, [shape[0], shape[1]])
        if is_plotting:
            j = torch.full((conf.shape[0], 1), i)
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])  # filter wrong boxes
            j, cls, box, conf = j[keep], cls[keep], box[keep], conf[keep]
        #targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
        targets.append({'scores': conf.squeeze(dim=1), 'labels':cls.squeeze(dim=1), 'boxes':box})
    results = []
    for i, o in enumerate(targets):
        tmp_boxes= o['boxes']
        tmp_scores= o['scores']
        tmp_class= o['labels']
        if apply_nms:
            #keep = batched_nms(tmp_boxes, tmp_scores, tmp_class, 0.1)
            keep = nms(tmp_boxes, tmp_scores, 0.3)
            tmp_scores_nms = tmp_scores[keep]
            tmp_class_nms = tmp_class[keep]
            tmp_boxes_nms = tmp_boxes[keep]
            results.append({'scores': tmp_scores_nms, 'labels': tmp_class_nms, 'boxes': tmp_boxes_nms})

    return results

#==================================images object processing=======================================
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def detect_view(image,boxes,classes,scores,masks=None,category_index=None):
    """
    Args:
        image:  w x h x channel
        boxes: num_obj * 4
        classes: 1 x num_obj
        scores: 1 x num_obj
        masks: None
        category_index: class_id,class_name
    Returns:
    """
    annotator = Annotator(image, classes=category_index,pil=True)
    #boxes = scale_coords(img1_shape=image.shape[0:2],coords=boxes,img0_shape=image.shape[0:2])
    annotator.box_label(boxes, classes, scores)
    return annotator.result()

class Annotator:

    def __init__(self, im, line_width=None, font_size=None, font='utils/Arial.ttf', pil=True, classes=None):
        self.pil = pil
        self.classes = classes
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.lw = line_width or max(round(sum(self.im.size) / 2 * 0.003), 2)  # line width
            #self.font = check_font(font= font,size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
            try:
                self.font = ImageFont.truetype(font=font,
                                               size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
            except Exception:
                self.font = ImageFont.load_default()
        else:  # use cv2
            self.im = im
            self.lw = line_width or max(round(sum(self.im.shape) / 2 * 0.003), 2)  # line width

        # Pose
        #self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        #                 [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        #self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        #self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    def box_label(self, boxes, labels,scores):
        if self.pil:
            labels = np.float32(labels)
            colors = []
            for cls in labels:
                color_tmp = ImageColor.getrgb(STANDARD_COLORS[int(cls % len(STANDARD_COLORS))])
                colors.append(color_tmp)
            for box, cls, score, color in zip(boxes, labels, scores, colors):
                left, top, right, bottom = box
                class_score = f'{self.classes[int(cls)]}: {score.item():.4f}'
                self.draw.rectangle([left, top, right, bottom],width=self.lw, outline=color)
                try:
                    _, _, w, h = self.font.getbbox(class_score)  # text width, height
                except Exception:
                    w, h = self.font.getsize(class_score)  # text width, height
                outside = top - h >= 0 # label fits outside box
                self.draw.text((left, top - h if outside else top), class_score, fill=color, font=self.font)
                #plt.imshow(self.im)
                #plt.show()
                #self.draw.rectangle([left,top - h if outside else top,
                #                     left + w + 1,
                #                     top + 1 if outside else top + h + 1])#,fill=color)
        else:  # cv2
            labels = np.float32(labels)
            colors = []
            for cls in labels:
                color_tmp = ImageColor.getrgb(STANDARD_COLORS[int(cls % len(STANDARD_COLORS))])
                colors.append(color_tmp)
            for box, cls, score, color in zip(boxes, labels, scores, colors):
                left, top, right, bottom = box
                p1, p2 = (int(left), int(top)), (int(right), int(bottom))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
                tf = max(self.lw - 1, 1)  # font thickness
                class_score = f'{self.classes[int(cls)]}: {score}'
                w, h = cv2.getTextSize(class_score , 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, class_score, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, color,
                            thickness=tf, lineType=cv2.LINE_AA)
    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
#============================================================================================
def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

def val_map(save_dir,results,results_gt,target_sizes,classes):
    """
    Args:
        results: list [image1:'box': 'pred': 'class']
        results_gt:

    Returns:
    """

    nc = len(classes)
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    seen = 0
    s = ('%10s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    logger.info(s)
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    jdict, stats, ap, ap_class = [], [], [], []
    stats_v8 = []
    for index in range(0,len(results_gt)):
        seen = seen + 1
        boxes,labels,scores = results[index]['boxes'], results[index]['labels'],results[index]['scores']
        #boxes, labels, scores =results_gt[index]['boxes'], results_gt[index]['labels'], results_gt[index]['labels']
        boxes_gt,labels_gt = results_gt[index]['boxes'], results_gt[index]['labels']
        nl = len(labels_gt)
        tcls = labels_gt.tolist() if nl else []
        if len(boxes) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                stats_v8.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        boxes = scale_coords(target_sizes[index], boxes, target_sizes[index])  # native-space pred
        if nl:
            boxes_gt = scale_coords(target_sizes[index], boxes_gt, target_sizes[index])  # native-space pred
            labelsn = torch.cat((labels_gt.unsqueeze(dim=1), boxes_gt), 1)  # native-space labels
            predn = torch.cat((boxes,scores.unsqueeze(dim=1),labels.unsqueeze(dim=1)),1)
            correct = process_batch(predn, labelsn, iouv)
            correct_v8 = process_batch_v8(predn,labelsn, iouv)
        else:
            correct = torch.zeros(boxes.shape[0], niou, dtype=torch.bool)
            correct_v8 = torch.zeros(boxes.shape[0], niou, dtype=torch.bool)
        stats.append((correct.cpu(), scores.cpu(), labels.cpu(), tcls))  # (correct, conf, pcls, tcls)
        stats_v8.append((correct_v8.cpu(), scores.cpu(), labels.cpu(), tcls))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    stats_v8 = [np.concatenate(x, 0) for x in zip(*stats_v8)]  # to nump
    #if len(stats) and stats[0].any():
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=classes)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        p_v8, r_v8, ap_v8, f1_v8, ap_class_v8 = ap_per_class(*stats_v8, plot=True, save_dir=save_dir, names=classes)
        ap50_v8, ap_v8 = ap_v8[:, 0], ap_v8.mean(1)  # AP@0.5, AP@0.5:0.95
        mp_v8, mr_v8, map50_v8, map_v8 = p_v8.mean(), r_v8.mean(), ap50_v8.mean(), ap_v8.mean()
        nt_v8 = np.bincount(stats_v8[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    # Print results
    pf = '%10s' + '%11i' * 2 + '%11.3g' * 4  # print format
    logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    for i, c in enumerate(ap_class):
        logger.info(pf % (classes[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    # Print speedsp, r, f1, mp, mr, map50, map
    #t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #shape = (batch_size, 3, imgsz, imgsz)
    #logger.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
    training = True
    save_txt = False
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        logger.info(f"Results saved to {save_dir}{s}")

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps

def val_map_v8(save_dir,results,results_gt,classes,loss_epoch):
    """
    Args:
        results: list [image1:'box': 'pred': 'class'] xyxy
        results_gt:
    Returns:
    """
    nc = len(classes)
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    seen = 0
    s = ('%10s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    with open(loss_epoch, 'a') as f:
        f.write('\n' f'{s}')

    logger.info(s)
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    jdict, stats, ap, ap_class = [], [], [], []

    for index in range(0,len(results_gt)):
        seen = seen + 1
        boxes,labels,scores = results[index]['boxes'], results[index]['labels'],results[index]['scores']
        boxes_gt,labels_gt = results_gt[index]['boxes'], results_gt[index]['labels']
        nl, npr = labels_gt.shape[0], boxes.shape[0]  # number of labels, predictions

        tcls = labels_gt.tolist() if nl else []
        if npr == 0:
            if nl:
                correct_bboxes= torch.zeros(0, niou, dtype=torch.bool)
                stats.append((correct_bboxes, torch.Tensor(), torch.Tensor(), tcls))
            continue
        # Evaluate
        if nl:
            labelsn = torch.cat((labels_gt[:,None], boxes_gt), 1)  # native-space labels
            predn = torch.cat((boxes,scores[:,None],labels[:,None]),1)#(xyxy,conf,label)
            correct_bboxes = process_batch_v8(predn,labelsn, iouv)
        else:
            correct_bboxes = torch.zeros(boxes.shape[0], niou, dtype=torch.bool)
        stats.append((correct_bboxes.cpu(), scores.cpu(), labels.cpu(), tcls))  # (correct, conf, pcls, tcls)
    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    #stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    #if len(stats) and stats[0].any():
    if len(stats):
        #p1, r1, ap1, f11, ap_class1 = ap_per_class(*stats, plot=True, save_dir=save_dir, names=classes)
        p, r, ap, f1, ap_class = ap_per_class_v8(*stats,plot=True, save_dir=save_dir,
                                                  names=classes)[2:]

        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        #nt_per_class = np.bincount(stats[-1].astype(int), minlength=nc)
    else:
        nt = torch.zeros(1)
    # Print results
    pf = '%10s' + '%11i' * 2 + '%11.3g' * 4  # print format
    logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    with open(loss_epoch, 'a') as f:
        f.write('\n' '{}'.format(pf % ('all', seen, nt.sum(), mp, mr, map50, map)))

    for i, c in enumerate(ap_class):
        logger.info(pf % (classes[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        with open(loss_epoch, 'a') as f:
            f.write('\n' '{}'.format(pf % (classes[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) ))

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps
def process_batch(detections, labels,iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct
def process_batch_v8(detections, labels,iouv):
    """
    Return correct prediction matrix.
    Args:
        detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
            Each detection is of the format: x1, y1, x2, y2, conf, class.
        labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
            Each label is of the format: class, x1, y1, x2, y2.
        pred_classes (torch.Tensor): Predicted class indices of shape(N,).
        true_classes (torch.Tensor): Target class indices of shape(M,).
        iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth

    Returns:
        (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
    """
    #iou = box_iou(labels[:, 1:], detections[:, :4])
    iou = box_iou_v8(labels[:, 1:], detections[:, :4])
    pred_classes = detections[:, 5]
    true_classes = labels[:, 0]
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    correct_class = true_classes[:, None] == pred_classes
    for i, iouv in enumerate(iouv):
        x = torch.nonzero(iou.ge(iouv) & correct_class)  # IoU > threshold and classes match
        if x.shape[0]:
            # Concatenate [label, detect, iou]
            matches = torch.cat((x, iou[x[:, 0], x[:, 1]].unsqueeze(1)), 1).cpu().numpy()
            if x.shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
def box_iou_v8(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)



def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot and py != []:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index

    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def ap_per_class_v8(tp,conf,pred_cls, target_cls,plot=False, on_plot=None, save_dir=Path(),names=(),eps=1e-16, prefix=''):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts for each class.
            fp (np.ndarray): False positive counts for each class.
            p (np.ndarray): Precision values at each confidence threshold.
            r (np.ndarray): Recall values at each confidence threshold.
            f1 (np.ndarray): F1-score values at each confidence threshold.
            ap (np.ndarray): Average precision for each class at different IoU thresholds.
            unique_classes (np.ndarray): An array of unique classes that have data.

    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot and py != []:
        plot_pr_curve(px, py, ap,  Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r,  ap, f1,unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec
# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def build_val_map(opts,pred_label,target_label):
    classes = categories
    nc = len(classes)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    target_sizes = torch.stack([t["size"] for t in target_label], dim=0).cpu() # ["orig_size"] : height X width
    results_pred = PostProcess(pred_label['pred_logits'][-1].detach().cpu(),pred_label['pred_boxes'][-1].detach().cpu(),
                             target_sizes)
    tgt_bbox, tgt_ids = [], []
    for v in target_label:
        tgt_ids.append(v["labels"].detach().cpu())
        tgt_bbox.append(v["boxes"].detach().cpu())
    results_pred_gt = PostProcess_gt(tgt_ids, tgt_bbox, target_sizes,target_sizes)
    save_dir = getattr(opts,"common_save_dir")
    results, maps = val_map_v8(save_dir,results_pred,results_pred_gt,target_sizes,classes)
    #results, map= val_map(save_dir,results_pred,results_pred_gt,target_sizes,classes)
    # Update best mAP
    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
    return results, maps



