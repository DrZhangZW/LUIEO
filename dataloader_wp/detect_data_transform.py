import random,os
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
#from utils.box_ops import box_xyxy_to_cxcywh
#from utils.misc import interpolate
# needed due to empty tensor bug in pytorch and torchvision 0.5

from packaging import version
import torchvision
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)

def crop(image, target, region):
    if type(image) is tuple:
        cropped_image = tuple(F.crop(tmp_image, *region) for tmp_image in image)
    else:
        cropped_image = F.crop(image, *region)
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(cropped_image)
    target = target.copy()
    i, j, h, w = region
    # should we do something wrt the original size?

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        target["size"] = torch.tensor([h, w])
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    if type(image) is tuple:
        flipped_image = tuple(F.hflip(tmp_image) for tmp_image in image)
        w, h = image[0].size
    else:
        flipped_image = F.hflip(image)
        w, h = image.size
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(flipped_image[0])

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if type(image) is tuple:
        size = get_size(image[0].size, size, max_size)
        rescaled_image = tuple(F.resize(tmp_image, size)for tmp_image in image)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image[0].size, image[0].size))
        ratio_width, ratio_height = ratios
    else:
        size = get_size(image.size, size, max_size)
        rescaled_image = F.resize(image, size)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
        ratio_width, ratio_height = ratios
    if target is None:
        return rescaled_image, None
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area
    if "size" in target:
        h, w = size
        target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        if type(img) is tuple:
            #w = random.randint(self.min_size, min(img[0].width, self.max_size))
            #h = random.randint(self.min_size, min(img[0].height, self.max_size))
            #region = T.RandomCrop.get_params(img[0], [h, w])
            w = min(img[0].width, img[0].height)
            region = T.RandomCrop.get_params(img[0], [w, w])
        else:
            w= min(img.width,img.height)
            #w = random.randint(self.min_size, min(img.width, self.max_size))
            #h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [w, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=1):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        #size = random.choice(self.sizes)
        return resize(img, target, self.sizes, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms2(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        if type(img) is tuple:
            Tensor_image = tuple(F.to_tensor(tmp_image) for tmp_image in img)
        else:
            Tensor_image  =F.to_tensor(img)

        return Tensor_image, target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        #image = F.normalize(image, mean=self.mean, std=self.std)
        if target == []:
            return image, []
        else:
            target = target.copy()
            h, w = image.shape[-2:]
            thres = 3 * (h * w) * 0.01 * 0.01

            if "boxes" in target:
                area = target['area']
                boxes = target['boxes']
                labels = target['labels']
                keep = area>thres
                target['area'] = area[keep]
                target['boxes'] = boxes[keep]
                target['labels'] = labels[keep]

                boxes = target["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes
            return image, target



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

#==================================labels processing=======================================
def txt_to_coco(under_label_path,W,H):
    write_json_context = dict()
    write_json_context['images'] = []
    write_json_context['annotations'] = []
    img_context={}
    img_context['file_name']=under_label_path
    img_context['height']=H
    img_context['width']=W
    write_json_context['images'].append(img_context)
    if os.path.exists(under_label_path):
        with open(under_label_path, 'r') as f:
            lines = f.readlines()
        for j, line in enumerate(lines):
            bbox_dict = {}
            class_id, x, y, w, h = line.strip().split(' ')
            class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)
            xmin = (x - w / 2) * W
            ymin = (y - h / 2) * H
            xmax = (x + w / 2) * W
            ymax = (y + h / 2) * H
            w = w * W
            h = h * H
            # bbox_dict['id'] = i * 10000 + j  # bounding box的坐标信息
            # bbox_dict['image_id'] = i
            height, width = abs(ymax - ymin), abs(xmax - xmin)
            area = height * width
            bbox_dict['category_id'] = class_id  # or class_id
            bbox_dict['iscrowd'] = 0
            bbox_dict['area'] = area
            bbox_dict['bbox'] = [xmin, ymin, w, h]  # [xmin, ymin, w, h]
            bbox_dict['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
            write_json_context['annotations'].append(bbox_dict)
    else:
        bbox_dict = {}
        bbox_dict['category_id'] = 11  # or class_id
        bbox_dict['iscrowd'] = 0
        bbox_dict['area'] = 0
        bbox_dict['bbox'] = [0, 0, 0, 0]  # [xmin, ymin, w, h]
        write_json_context['annotations'].append(bbox_dict)
    return write_json_context

def ConvertCocotoMask(target):
    # boxes: [xmin, ymin,w,h] to [xmin, ymin,xmax,ymax]
    w, h = target['images'][0]['width'],target['images'][0]['height']
    #image_id = target["image_id"]
    #image_id = torch.tensor([image_id])
    anno = target["annotations"]
    anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)
    classes = [obj["category_id"] for obj in anno]
    classes = torch.tensor(classes, dtype=torch.int64)
    #if return_masks:
    #    segmentations = [obj["segmentation"] for obj in anno]
    #    masks = convert_coco_poly_to_mask(segmentations, h, w)
    keypoints = None
    if anno and "keypoints" in anno[0]:
        keypoints = [obj["keypoints"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    if keypoints is not None:
        keypoints = keypoints[keep]
    target = {}
    target["boxes"] = boxes
    target["labels"] = classes

    if keypoints is not None:
        target["keypoints"] = keypoints
    # for conversion to coco api
    area = torch.tensor([obj["area"] for obj in anno])
    iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
    target["area"] = area[keep]
    target["iscrowd"] = iscrowd[keep]
    target["orig_size"] = torch.as_tensor([ int(h),int(w)])
    target["size"] = torch.as_tensor([ int(h),int(w)])
    return target


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)
#  h, w = image.shape[-2:]
# boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)