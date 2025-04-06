import os, math
from scipy.io import loadmat
from utils import logger
from PIL import Image
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from pathlib import Path
import random
from typing import Optional, Tuple, Union
from dataloader_wp.detect_data_transform import txt_to_coco, ConvertCocotoMask

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
DEFAULT_IMAGE_WIDTH = DEFAULT_IMAGE_HEIGHT = 256
# self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
from dataloader_wp import detect_data_transform as T


def make_coco_transforms(train_and_val, sizes):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if train_and_val == True:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSizeCrop(0, 0),
            T.RandomResize(sizes=sizes),
            normalize,
        ])
    else:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSizeCrop(0, 0),
            T.RandomResize(sizes=sizes),
            normalize,
        ])

def get_train_val_dataset(opts):
    train_dataset = ImageDataset(opts, is_training=True)
    n_train_samples = len(train_dataset)
    train_sampler = VariableBatchSamper(opts=opts, n_data_samples=n_train_samples, is_training=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,  # Handled inside data sampler
                              # num_workers=getattr(opts, "dataset_workers", 1),
                              # pin_memory=getattr(opts, "dataset_pin_memory", False),
                              batch_sampler=train_sampler,
                              collate_fn=ImageDataset.collate_fn
                              )
    #==================================val loader: batchsize=2===============================================
    val_dataset = ImageDataset(opts, is_training=False)
    n_val_samples = len(val_dataset)
    val_sampler = VariableBatchSamper(opts=opts, n_data_samples=n_val_samples, is_training=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,  # Handled inside data sampler
                            # num_workers=getattr(opts, "dataset_workers", 1),
                            # pin_memory=getattr(opts, "dataset_pin_memory", False),
                            batch_sampler=val_sampler,
                            collate_fn=ImageDataset.collate_fn
                            )
 
    return train_loader, train_sampler, val_loader 


def make_divisible(v: Union[float, int], divisor: Optional[int] = 8,
                   min_value: Optional[Union[float, int]] = None, ) -> Union[float, int]:
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


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, opts, is_training: Optional[bool] = True):
        #================================================================
        self.opts = opts
        self.is_training = is_training
        #===================train data path====================================
        if self.is_training:
            self.clean_path = getattr(opts, "dataset_train_clean", None)
            self.depth_path = getattr(opts, "dataset_train_depth", None)
            self.under_path = getattr(opts, "dataset_train_under", None)
            self.under_label_path = getattr(opts, "dataset_train_under_label", None)
            #====================load data files===========================
            self.clean_data = self.load_flist(self.clean_path)
            self.depth_data = self.img2mat(self.clean_data, self.depth_path)

            self.under_data = self.load_flist(self.under_path)
            self.under_label = self.img2label_paths(self.under_data, self.under_label_path)
            #=============================================================
        else:
            self.val_under_path = getattr(opts, "dataset_val_under", None)
            self.val_under_label_path = getattr(opts, "dataset_val_under_label", None)
            #===================load data files=======================================
            self.under_data_val = self.load_flist(self.val_under_path)
            self.under_data_val_label = self.img2label_paths(self.under_data_val,self.val_under_label_path)
        t=1

    def __len__(self):
        if self.is_training:
            data_num = len(self.under_data)  # if len(self.clean_data) > len(self.under_data) else len(self.under_data)
        else:
            data_num = len(self.under_data_val)
        return data_num

    @staticmethod
    def collate_fn(batch):
        clean, depth, under, targets = zip(*batch)
        try:
            cleans = torch.stack(clean, 0)
            depths = torch.stack(depth, 0)
        except:
            cleans, depths = [], depth
        unders = torch.stack(under, 0)
        datas = tuple([cleans, depths, unders, targets])
        return datas

    def __getitem__(self, batch_indexs_tup: Tuple):
        crop_size_h, crop_size_w, img_index = batch_indexs_tup
        self._transforms = make_coco_transforms(True, (crop_size_h, crop_size_w))
        clean, depth, under, target = self.load_item(img_index)
        return clean, depth, under, target

    def load_item(self, img_index):
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input clean image, depth,underwater image
        nx4 boxes form [x1, y1, x2, y2] to [x, y, w, h] normalized where x1y1=top-left, x1y2=bottom-right
        """
        #=========================setting the Image_ID of under, clean, depth====================================
        if self.is_training:
            under_index = img_index
            clean_index = int(np.random.random() * len(self.clean_data))
            under = Image.open(self.under_data[under_index]).convert('RGB')
            clean = Image.open(self.clean_data[clean_index]).convert('RGB')
            # ===================load depth========================================
            if os.path.exists(self.depth_data[clean_index]):
                depth_path = self.depth_data[clean_index]
                # print(depth_path)
                depth = np.array(loadmat(depth_path)['dph']).astype(np.float32)
                depth = Image.fromarray(depth)
            else:
                depth = None
            #======================read label=============================
            under_w_ori, under_h_ori = under.size[0:2]  # Image.open shape wXh
            under_label_path = self.under_label[under_index]
        else:
            under_index = img_index
            under_val_path = self.under_data_val[under_index]
            under = Image.open(under_val_path).convert('RGB')
            #======================read label=============================
            under_w_ori, under_h_ori = under.size[0:2]  # Image.open shape wXh
            under_label_path = self.under_data_val_label[under_index]
        #=========================setting the image label====================================
        target = txt_to_coco(under_label_path, under_w_ori, under_h_ori)  # boxes: [xmin, ymin,w,h]
        target = ConvertCocotoMask(target)  # boxes: [xmin, ymin,xmax,ymax]
        """
        from utils import detect_utils
        temp_pred_label = detect_utils.detect_view(under, target['boxes'], target['labels'],
                                                   scores=target['labels'], category_index=detect_utils.categories)
        import cv2
        cv2.imshow('temp_pred',temp_pred_label)
        cv2.waitKey(1)
        """
        #======================transforms images======================
        under, target = self._transforms(under, target)
        if self.is_training:
            if depth == None:
                clean = self._transforms(clean, [])
            else:
                (clean, depth), _ = self._transforms((clean, depth), [])
        else:
            clean,depth = 1,under_val_path
        return clean, depth, under, target

    def read_label(self, path):
        if os.path.exists(path):
            with open(path) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                l = np.array(l, dtype=np.float32)
            if len(l):
                assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
        else:
            l = None
        return l

    def img2img(self, list, path):
        f = []
        # f += [path + os.sep + str(Path(x).name) for x in list]
        f += ['{}/{}'.format(path, str(Path(x).name)) for x in list]
        return f

    def img2mat(self, list, path):
        f = []
        # f += [path + os.sep + str(Path(x).stem) + '.mat' for x in list]
        f += ['{}/{}'.format(path, str(Path(x).stem) + '.mat') for x in list]
        return f

    def img2label_paths(self, list, path):
        # Define label paths as a function of image paths
        # sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        # return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in list]
        f = []
        # f += [path + os.sep + str(Path(x).stem) + '.txt' for x in list]
        f += ['{}/{}'.format(path, str(Path(x).stem) + '.txt') for x in list]
        return f

    def load_flist(self, flist):
        f = []
        path = flist
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
            else:
                raise Exception(f'{p} does not exist')

        f2 = []
        f2 += [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
        return f2

    def choose_eta(self, eta_r, eta_g, eta_b, num_water, num_pic):
        eta = []
        for i in range(0, num_pic):
            water_type1 = int(np.random.random() * num_water)
            water_type2 = int(np.random.random() * num_water)
            eta_rI, eta_gI, eta_bI = torch.tensor([[[eta_r[water_type1]]]]), torch.tensor(
                [[[eta_g[water_type1]]]]), torch.tensor([[[eta_b[water_type1]]]])
            eta1 = torch.stack([eta_rI, eta_gI, eta_bI], axis=3)
            eta_rI, eta_gI, eta_bI = torch.tensor([[[eta_r[water_type2]]]]), torch.tensor(
                [[[eta_g[water_type2]]]]), torch.tensor([[[eta_b[water_type2]]]])
            eta2 = torch.stack([eta_rI, eta_gI, eta_bI], axis=3)
            weight = torch.from_numpy(np.random.uniform(0, 1, 1))
            eta_w = weight * eta1 + (1 - weight) * eta2
            eta.append(eta_w.permute((0, 3, 1, 2)))
        eta = torch.cat(eta, 0)
        # eta = eta.numpy()
        return eta

    def forward_random_parameters(self, image, depth):  # x:NCHW, ex:e^(-d(x)), N,1,H,W, beta:N,1,1,1
        # eta_r = np.array([0.30420412, 0.30474395, 0.35592191, 0.32493874, 0.55091001,0.42493874, 0.55874165, 0.13039252 , 0.10760831, 0.15963731])#1-4 green  5-7 blue 8-10 red+green = yellow
        # eta_g = np.array([0.11727661, 0.05999663, 0.11227639, 0.15305673, 0.14385827, 0.12305673, 0.0518615, 0.18667714, 0.1567016, 0.205724217])
        # eta_b = np.array([0.1488851, 0.30099538, 0.38412464, 0.25060999, 0.01387215, 0.055799195, 0.0591001, 0.5539252 , 0.60103   , 0.733602  ])
        # N_r = np.array([0.805,0.804,0.83,0.8,0.75,0.75,0.71,0.67,0.62,0.55])
        # N_g = np.array([0.961,0.955,0.95,0.925,0.885,0.885,0.82,0.73,0.61,0.46])
        # N_b = np.array([0.982,0.975,0.968,0.94,0.89,0.875,0.8,0.67,0.5,0.29])
        # eta_r1 =  np.array([0.216913, 0.21815601, 0.18632958, 0.22314355, 0.28768207,0.28768207, 0.34249031, 0.40047757, 0.4780358 , 0.597837  ])#1-6 green+blue=cyan 7 8 green 9 10red+green = yellow
        # eta_g1 = np.array([0.03978087, 0.04604394, 0.05129329, 0.07796154, 0.12216763, 0.12216763, 0.19845094, 0.31471074, 0.49429632, 0.77652879])
        # eta_b1 = np.array([0.01816397, 0.02531781, 0.03252319, 0.0618754 , 0.11653382,0.13353139, 0.22314355, 0.40047757, 0.69314718, 1.23787436])
        # eta_r2 = np.array([0.09420412, 0.09474395, 0.08092191, 0.09691001, 0.12493874,0.12493874, 0.14874165, 0.1739252 , 0.20760831, 0.25963731])#1-6 green+blue=cyan 7 8 green 9 10red+green = yellow
        # eta_g2 = np.array([0.01727661, 0.01999663, 0.02227639, 0.03385827, 0.05305673,0.05305673, 0.08618615, 0.13667714, 0.21467016, 0.33724217])
        # eta_b2 = np.array([0.00788851, 0.01099538, 0.01412464, 0.02687215, 0.05060999, 0.05799195, 0.09691001, 0.1739252 , 0.30103   , 0.537602  ])
        # 18 water type   1-4 green 5-8 green+blue 9-11 blue 12-18 red+green=yellow, the large eta, the more severe the degration.
        eta_r = np.array([0.30420412, 0.30474395, 0.35592191, 0.32493874,
                          0.216913, 0.21815601, 0.18632958, 0.22314355,
                          0.55091001, 0.42493874, 0.55874165,
                          0.13039252, 0.10760831, 0.15963731, 0.4780358, 0.597837, 0.20760831, 0.25963731])
        eta_g = np.array([0.11727661, 0.05999663, 0.11227639, 0.15305673,
                          0.03978087, 0.04604394, 0.05129329, 0.07796154,
                          0.14385827, 0.12305673, 0.0518615,
                          0.18667714, 0.1567016, 0.205724217, 0.49429632, 0.77652879, 0.21467016, 0.33724217])
        eta_b = np.array([0.1488851, 0.30099538, 0.38412464, 0.25060999,
                          0.01816397, 0.02531781, 0.03252319, 0.0618754,
                          0.01387215, 0.055799195, 0.0591001,
                          0.5539252, 0.60103, 0.733602, 0.69314718, 1.23787436, 0.30103, 0.537602])
        depth = depth.detach().cpu()
        image = image.detach().cpu()
        water_depth = 10 * torch.rand((image.shape[0], 1, 1, 1))
        # ---------------------------------------------------------------------------------------------------------------
        img_num = image.shape
        eta = self.choose_eta(eta_r, eta_g, eta_b, len(eta_r), img_num[0])
        # -------------------------------------------------------
        t = torch.exp(-depth * eta)

        # L_t1 =np.exp(  np.multiply(-water_depth, eta) )
        # L_t2 = np.exp(np.multiply(-1, np.multiply(depth + Z_b, eta)))
        L_t1 = torch.exp(-(water_depth + depth - depth) * eta)
        # L_t2 = torch.exp(-(depth + Z_b) * eta)
        L_t2 = torch.exp(-depth * eta)

        x, y = torch.meshgrid(torch.linspace(0, depth.shape[3] - 1, depth.shape[3]),
                              torch.linspace(0, depth.shape[2] - 1, depth.shape[2]), indexing='ij')

        sigma = torch.rand(((image.shape[0], 1, 1, 1))) * t.shape[2]
        z_l = torch.randn((image.shape[0], 1, 1, 1))
        r_l = torch.rand(((image.shape[0], 1, 1, 1)))
        L_t3, v = get_art_light(x, y, eta, depth, sigma,
                                x_c=torch.floor(torch.rand(((image.shape[0], 1, 1, 1))) * x.shape[1]),
                                y_c=torch.floor(torch.rand(((image.shape[0], 1, 1, 1))) * x.shape[0]),
                                L_art=torch.rand(((image.shape[0], 1, 1, 1))).clip(0.7, 1),
                                Z_l=z_l, r_l=r_l)

        use_art = torch.rand((image.shape[0], 1, 1, 1)) > 0.5
        water_illum = torch.zeros((image.shape[0], 3, 1, 1))
        for index in range(image.shape[0]):
            water_illum_temp = torch.rand((1, 3, 1, 1)) if use_art[index] else torch.rand((1, 2, 1, 1))
            water_illum_temp = water_illum_temp / torch.sum(water_illum_temp, dim=1, keepdim=True)
            if use_art[index]:
                water_illum[index] = water_illum_temp
            else:
                water_illum[index, 0:2] = water_illum_temp

        L_t = water_illum[:, 0].unsqueeze(dim=3) * L_t1 + water_illum[:, 1].unsqueeze(dim=3) * L_t2 + water_illum[:,
                                                                                                      2].unsqueeze(
            dim=3) * L_t3

        # L_t =L_t1 + L_t2  # L_t3
        # --------------------------------------------------------------
        I = image * t * L_t + L_t * (1 - t)
        """
        from matplotlib import pyplot as plt
        num_img = 5 if len(I) > 6 else len(I)
        fig, ax = plt.subplots(num_img, 5, figsize=(6, 6))
        if len(I) == 1:
            [ax[0].imshow(I[i].transpose((1, 2, 0))) for i, _ in enumerate(I[0: num_img])]
            [ax[1].imshow(L_t[i].transpose((1, 2, 0))) for i, _ in enumerate(L_t[0: num_img])]
            [ax[2].imshow(t[i].transpose((1, 2, 0))) for i, _ in enumerate(t[0: num_img])]
            [ax[3].imshow(t[i].transpose((1, 2, 0))) for i, _ in enumerate(t[0: num_img])]
            [ax[4].imshow(L_t[i].transpose((1, 2, 0))) for i, _ in enumerate(L_t[0: num_img])]
        else:
            [ax[i][0].imshow(I[i].permute((1, 2, 0))) for i, _ in enumerate(I[0: num_img])]
            [ax[i][1].imshow(L_t1[i].permute((1, 2, 0))) for i, _ in enumerate(L_t1[0: num_img])]
            [ax[i][2].imshow(L_t3[i].permute((1, 2, 0))) for i, _ in enumerate(L_t2[0: num_img])]
            [ax[i][3].imshow(L_t[i].permute((1, 2, 0))) for i, _ in enumerate(L_t[0: num_img])]
            [ax[i][4].imshow(t[i].permute((1, 2, 0))) for i, _ in enumerate(t[0: num_img])]

        plt.title('clean-->L_t1-->L_t3--->L_t--->t',  x=-1.4, y=-0.6)
        plt.show()
        """
        output = {"clean": image.float(), "simulate": I.float(), "backlight": L_t.float(), "transmission": t.float(),
                  "depth": depth.float(), "water_type": eta.float(), "water_depth": water_depth.float()}
        return output

class ImageDataset_val(torch.utils.data.Dataset):
    def __init__(self, opts, is_training: Optional[bool] = False):
        self.opts = opts
        self.is_training = is_training
        self.val_under_path = getattr(opts, "dataset_val_under", None)
        self.val_under_label_path = getattr(opts, "dataset_val_under_label", None)

        self.crop_size_w = getattr(opts, "sampler_vbs_crop_size_width", DEFAULT_IMAGE_WIDTH)
        self.crop_size_h = getattr(opts, "sampler_vbs_crop_size_height", DEFAULT_IMAGE_HEIGHT)
        setattr(opts, "sampler_crop_size_width", self.crop_size_w)
        setattr(opts, "sampler_crop_size_height", self.crop_size_h)

        self.under_data = self.load_flist(self.val_under_path)
        self.under_label = self.img2label_paths(self.under_data, self.val_under_label_path)

    def __len__(self):
        return len(self.under_data)

    @staticmethod
    def collate_fn(batch):
        under, under_path, targets = zip(*batch)
        if len(under)==1:
            under = (under[0], 0.1*under[0])
            targets = (targets[0], targets[0])
        unders = torch.stack(under, 0)
        return tuple([under_path, under_path, unders, targets])

    def __getitem__(self, index):
        # crop_size_h, crop_size_w, img_index = batch_indexs_tup
        # self._transforms=make_coco_transforms(self.is_training, (crop_size_h, crop_size_w))
        under, under_path, target = self.load_item(index)
        return under, under_path, target

    def load_item(self, img_index):
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input clean image, depth,underwater image
        nx4 boxes form [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
        """
        under_index = img_index
        under_path = self.under_data[under_index]
        under = Image.open(under_path).convert('RGB')
        under_w_ori, under_h_ori = under.size[0:2]  # Image.open shape wXh

        under_label_path = self.under_label[under_index]
        target = txt_to_coco(under_label_path, under_w_ori, under_h_ori)  # boxes: [xmin, ymin,w,h]
        target = ConvertCocotoMask(target)  # boxes: [xmin, ymin,xmax,ymax]

        stride = 32
        if min(under_w_ori, under_h_ori) >= 512:
            img_size = 512
        else:
            img_size = min(under_w_ori, under_h_ori)

        r1 = img_size / max(under_w_ori, under_h_ori)
        r2 = img_size / max(under_w_ori, under_h_ori)
        new_w = math.floor((under_w_ori * r1) / stride) * stride
        new_h = math.floor((under_h_ori * r2) / stride) * stride
        new_size = (int(new_w), int(new_h))## (int(608), int(608))
        """
        from utils import detect_utils
        temp_pred_label = detect_utils.detect_view(under, target['boxes'], target['labels'],
                                                   scores=target['labels'], category_index=detect_utils.categories)
        import cv2
        cv2.imshow('temp_pred',temp_pred_label)
        cv2.waitKey(1)
        """
        image_transform = make_coco_transforms(False, (new_size))
        under, target = image_transform(under, target)

        return under, under_path, target

    def read_label(self, path):
        if os.path.exists(path):
            with open(path) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                l = np.array(l, dtype=np.float32)
            if len(l):
                assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
        else:
            l = None
        return l

    def img2img(self, list, path):
        f = []
        # f += [path + os.sep + str(Path(x).name) for x in list]
        f += ['{}/{}'.format(path, str(Path(x).name)) for x in list]
        return f

    def img2mat(self, list, path):
        f = []
        # f += [path + os.sep + str(Path(x).stem) + '.mat' for x in list]
        f += ['{}/{}'.format(path, str(Path(x).stem) + '.mat') for x in list]
        return f

    def img2label_paths(self, list, path):
        # Define label paths as a function of image paths
        # sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        # return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in list]
        f = []
        # f += [path + os.sep + str(Path(x).stem) + '.txt' for x in list]
        f += ['{}/{}'.format(path, str(Path(x).stem) + '.txt') for x in list]
        return f

    def load_flist(self, flist):
        f = []
        path = flist
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
            else:
                raise Exception(f'{p} does not exist')

        f2 = []
        f2 += [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
        return f2
def get_art_light(x1, y1, eta, depth, sigma, x_c, y_c, L_art=0.8, r_l=0.3, Z_l=0.3):
    v = L_art * torch.exp(-1.0 / (2 * sigma ** 2) * ((x1 - x_c) ** 2 + (y1 - y_c) ** 2))
    D = Z_l ** 2 + (0 ** 2) * ((x1 - x_c) ** 2 + (y1 - y_c) ** 2)
    D = torch.sqrt(D)
    D_decay = torch.exp(-(D + depth) * eta)
    art_light = D_decay * v
    return art_light, v


class VariableBatchSamper(Sampler):
    def __init__(self, opts, n_data_samples: int, is_training: Optional[bool] = False):
        """
        :param opts: arguments
        :param n_data_samples: number of data samples in the dataset
        :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        n_gpus: int = max(1, torch.cuda.device_count())
        batch_size_gpu0: int = getattr(opts, "dataset_train_batch_size0", 32) if is_training \
            else getattr(opts, "dataset_val_batch_size0", 32)
        n_samples_per_gpu = int(math.ceil(n_data_samples * 1.0 / n_gpus))
        total_size = n_samples_per_gpu * n_gpus
        indexes = [idx for idx in range(n_data_samples)]
        # This ensures that we can divide the batches evenly across GPUs
        indexes += indexes[:(total_size - n_data_samples)]
        assert total_size == len(indexes)
        self.img_indices = indexes
        self.n_samples = total_size
        self.batch_size_gpu0 = batch_size_gpu0
        self.n_gpus = n_gpus
        self.shuffle = True if is_training else False
        self.epoch = 0

        crop_size_w: int = getattr(opts, "sampler_vbs_crop_size_width", DEFAULT_IMAGE_WIDTH)
        crop_size_h: int = getattr(opts, "sampler_vbs_crop_size_height", DEFAULT_IMAGE_HEIGHT)

        min_crop_size_w: int = getattr(opts, "sampler_vbs_min_crop_size_width", 160)
        max_crop_size_w: int = getattr(opts, "sampler_vbs_max_crop_size_width", 320)

        min_crop_size_h: int = getattr(opts, "sampler_vbs_min_crop_size_height", 160)
        max_crop_size_h: int = getattr(opts, "sampler_vbs_max_crop_size_height", 320)

        scale_inc: bool = getattr(opts, "sampler_vbs_scale_inc", False)
        scale_ep_intervals: list or int = getattr(opts, "sampler_vbs_ep_intervals", [40])
        scale_inc_factor: float = getattr(opts, "sampler_vbs_scale_inc_factor", 0.25)

        check_scale_div_factor: int = getattr(opts, "sampler_vbs_check_scale", 32)
        max_img_scales: int = getattr(opts, "sampler_vbs_max_n_scales", 10)
        if isinstance(scale_ep_intervals, int):
            scale_ep_intervals = [scale_ep_intervals]

        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

        self.scale_inc_factor = scale_inc_factor
        self.scale_ep_intervals = scale_ep_intervals

        self.max_img_scales = max_img_scales
        self.check_scale_div_factor = check_scale_div_factor
        self.scale_inc = scale_inc

        if is_training:
            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.n_gpus,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h)
        else:
            self.img_batch_tuples = [(crop_size_h, crop_size_w, self.batch_size_gpu0)]

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        #if self.shuffle:
        random.shuffle(self.img_indices)
        random.shuffle(self.img_batch_tuples)
        start_index = 0
        while start_index < self.n_samples:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)
            end_index = min(start_index + batch_size, self.n_samples)
            # print(f'end_index:{end_index}')
            batch_ids = self.img_indices[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if len(batch_ids) != batch_size:
                batch_ids += self.img_indices[:(batch_size - n_batch_samples)]
            start_index += batch_size
            if len(batch_ids) > 0:
                batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
                # print(batch)
                yield batch


    def update_scales(self, epoch, *args, **kwargs):
        if epoch in self.scale_ep_intervals and self.scale_inc:
            self.min_crop_size_w += int(self.min_crop_size_w * self.scale_inc_factor)
            self.max_crop_size_w += int(self.max_crop_size_w * self.scale_inc_factor)

            self.min_crop_size_h += int(self.min_crop_size_h * self.scale_inc_factor)
            self.max_crop_size_h += int(self.max_crop_size_h * self.scale_inc_factor)

            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.n_gpus,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h)
            logger.log('Scales updated in {}'.format(self.__class__.__name__))
            logger.log("New scales: {}".format(self.img_batch_tuples))


def _image_batch_pairs(crop_size_w: int,
                       crop_size_h: int,
                       batch_size_gpu0: int,
                       n_gpus: int,
                       max_scales: Optional[float] = 5,
                       check_scale_div_factor: Optional[int] = 32,
                       min_crop_size_w: Optional[int] = 160,
                       max_crop_size_w: Optional[int] = 320,
                       min_crop_size_h: Optional[int] = 160,
                       max_crop_size_h: Optional[int] = 320,
                       *args, **kwargs) -> list:
    """
        This function creates batch and image size pairs.  For a given batch size and image size, different image sizes
        are generated and batch size is adjusted so that GPU memory can be utilized efficiently.

    :param crop_size_w: Base Image width (e.g., 224)
    :param crop_size_h: Base Image height (e.g., 224)
    :param batch_size_gpu0: Batch size on GPU 0 for base image
    :param n_gpus: Number of available GPUs
    :param max_scales: Number of scales. How many image sizes that we want to generate between min and max scale factors.
    :param check_scale_div_factor: Check if image scales are divisible by this factor.
    :param min_crop_size_w: Min. crop size along width
    :param max_crop_size_w: Max. crop size along width
    :param min_crop_size_h: Min. crop size along height
    :param max_crop_size_h: Max. crop size along height
    :param args:
    :param kwargs:
    :return: a sorted list of tuples. Each index is of the form (h, w, batch_size)
    """
    width_dims = list(np.linspace(min_crop_size_w, max_crop_size_w, max_scales))
    if crop_size_w not in width_dims:
        width_dims.append(crop_size_w)
    height_dims = list(np.linspace(min_crop_size_h, max_crop_size_h, max_scales))
    if crop_size_h not in height_dims:
        height_dims.append(crop_size_h)
    image_scales = set()
    for h, w in zip(height_dims, width_dims):
        # ensure that sampled sizes are divisible by check_scale_div_factor
        # This is important in some cases where input undergoes a fixed number of down-sampling stages
        # for instance, in ImageNet training, CNNs usually have 5 downsampling stages, which downsamples the
        # input image of resolution 224x224 to 7x7 size
        h = make_divisible(h, check_scale_div_factor)
        w = make_divisible(w, check_scale_div_factor)
        image_scales.add((h, w))
    image_scales = list(image_scales)
    img_batch_tuples = set()
    n_elements = crop_size_w * crop_size_h * batch_size_gpu0
    for (crop_h, crop_y) in image_scales:
        # compute the batch size for sampled image resolutions with respect to the base resolution
        _bsz = max(batch_size_gpu0, int(round(n_elements / (crop_h * crop_y), 2)))
        _bsz = make_divisible(_bsz, n_gpus)
        img_batch_tuples.add((crop_h, crop_y, _bsz))
    img_batch_tuples = list(img_batch_tuples)
    return sorted(img_batch_tuples)


import argparse
from utils.common_utils import load_config_file

if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    parser = argparse.ArgumentParser(description='Training arguments', add_help=True)
    parser.add_argument('--common_config_file', type=str,
                        default=ROOT / "mobilevitv3_small_multiserver.yaml",
                        help="Configuration file")
    opts = parser.parse_args()
    opts = load_config_file(opts)

    train_dataset = ImageDataset(opts, is_training=True)
    n_train_samples = len(train_dataset)
    train_sampler = VariableBatchSamper(opts=opts, n_data_samples=n_train_samples, is_training=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,  # Handled inside data sampler
                              num_workers=getattr(opts, "dataset_workers", 1),
                              pin_memory=getattr(opts, "dataset_pin_memory", False),
                              batch_sampler=train_sampler,
                              collate_fn=ImageDataset.collate_fn
                              )
    for batch_id, batch in enumerate(train_loader):
        # move data to device
        X_s, depth, under, target = batch[0], batch[1], batch[2], batch[3]
