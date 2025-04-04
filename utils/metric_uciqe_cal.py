# model validation metrics
import os
import glob
import math
import numpy as np
import cv2
import argparse
from pathlib import Path
import pathlib
from PIL import Image
import torchvision.transforms.functional as F
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


def get_uciqe(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # RGB转为HSV
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180  # 色度的标准差
    mu = np.mean(S) / 255  # 饱和度的平均值
    n, m = np.shape(V)
    number = math.floor(n * m / 100)  # 所需像素的个数
    Maxsum, Minsum = 0, 0
    V1, V2 = V / 255, V / 255

    for i in range(1, number + 1):
        Maxvalue = np.amax(np.amax(V1))
        x, y = np.where(V1 == Maxvalue)
        Maxsum = Maxsum + V1[x[0], y[0]]
        V1[x[0], y[0]] = 0
    top = Maxsum / number
    for i in range(1, number + 1):
        Minvalue = np.amin(np.amin(V2))
        X, Y = np.where(V2 == Minvalue)
        Minsum = Minsum + V2[X[0], Y[0]]
        V2[X[0], Y[0]] = 1

    bottom = Minsum / number

    conl = top - bottom
    ###对比度
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2575 * mu

    print(uciqe,delta, conl, mu)
    return uciqe,delta, conl, mu


def get_uciqe2(image):
    # $\mbox{UCIQE}= c_1\times \sigma_c+c_2\times con_l+c_3\times \mu_s$
    # with $c_1=0.4680$, $c_2=0.2745$ and $c_3=0.2575$ as in \cite{yang2015underwater}.
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # RGB转为HSV
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180
    # 色度的标准差
    mu = np.mean(S) / 255  # 饱和度的平均值
    # 求亮度对比值
    n, m = np.shape(V)
    number = np.floor(n * m / 100)  # 所需像素的个数
    number = number.astype(np.int16)
    v = V.flatten() / 255
    v.sort()
    bottom = np.sum(v[:number]) / number
    v = -v
    v.sort()
    v = -v
    top = np.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu

    #print(uciqe,delta, conl, mu)
    return uciqe, delta, conl, mu


def getuiqm(img):
    # $\mbox{UIQM} = c_1\times \mbox{UICM} + c_2\times \mbox {UISM} + c_3\times \mbox{UIconM}$
    # with $c_1=0.0282$, $c_2=0.2953$, $c_3=3.5753$
    r, b, g = cv2.split(img)
    Uicm = uicm(img)
    EME_r = EME(r, 8)
    EME_b = EME(b, 8)
    EME_g = EME(g, 8)
    Uism = 0.299 * EME_r + 0.144 * EME_b + 0.557 * EME_g
    Uiconm = UICONM(img, 8)
    uiqm = 0.282 * Uicm + 0.2953 * Uism +0.6765* Uiconm # 色彩测量，清晰度，水下图像对比
    # uiqm = 0.299 * Uicm + 0.144 * Uism + 0.587 * Uiconm
    #uiqm = 0.0282 * Uicm + 0.2953 * Uism + 3.5753 * Uiconm
    #uiqm = 0.0282 * Uicm + 0.2953 * Uism + 3.5753 * Uiconm
    #print(uiqm,Uicm, Uism, Uiconm)
    return uiqm, Uicm, Uism, Uiconm


def uicm(img):
    b, r, g = cv2.split(img)
    RG = r - g
    YB = (r + g) / 2 - b
    m, n, o = np.shape(img)  # img为三维 rbg为二维
    K = m * n
    alpha_L = 0.1
    alpha_R = 0.1  ##参数α 可调
    T_alpha_L = math.ceil(alpha_L * K)  # 向上取整
    T_alpha_R = math.floor(alpha_R * K)  # 向下取整

    RG_list = RG.flatten()
    RG_list = sorted(RG_list)
    sum_RG = 0
    for i in range(T_alpha_L + 1, K - T_alpha_R):
        sum_RG = sum_RG + RG_list[i]
    U_RG = sum_RG / (K - T_alpha_R - T_alpha_L)
    squ_RG = 0
    for i in range(K):
        squ_RG = squ_RG + np.square(RG_list[i] - U_RG)
    sigma2_RG = squ_RG / K

    YB_list = YB.flatten()
    YB_list = sorted(YB_list)
    sum_YB = 0
    for i in range(T_alpha_L + 1, K - T_alpha_R):
        sum_YB = sum_YB + YB_list[i]
    U_YB = sum_YB / (K - T_alpha_R - T_alpha_L)
    squ_YB = 0
    for i in range(K):
        squ_YB = squ_YB + np.square(YB_list[i] - U_YB)
    sigma2_YB = squ_YB / K

    Uicm = -0.0268 * np.sqrt(np.square(U_RG) + np.square(U_YB)) + 0.1586 * np.sqrt(sigma2_RG + sigma2_YB)
    return Uicm


def EME(rbg, L):
    m, n = np.shape(rbg)  # 横向为n列 纵向为m行
    number_m = math.floor(m / L)
    number_n = math.floor(n / L)
    # A1 = np.zeros((L, L))
    m1 = 0
    E = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1 + L, n1:n1 + L]
            rbg_min = np.amin(np.amin(A1))
            rbg_max = np.amax(np.amax(A1))

            if rbg_min > 0:
                rbg_ratio = rbg_max / rbg_min
            else:
                rbg_ratio = rbg_max  ###
            E = E + np.log(rbg_ratio + 1e-5)

            n1 = n1 + L
        m1 = m1 + L
    E_sum = 2 * E / (number_m * number_n)
    return E_sum


def UICONM(rbg, L):  # wrong
    m, n, o = np.shape(rbg)  # 横向为n列 纵向为m行
    number_m = math.floor(m / L)
    number_n = math.floor(n / L)
    A1 = np.zeros((L, L))  # 全0矩阵
    m1 = 0
    logAMEE = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1 + L, n1:n1 + L]
            rbg_min = int(np.amin(np.amin(A1)))
            rbg_max = int(np.amax(np.amax(A1)))
            plip_add = rbg_max + rbg_min - rbg_max * rbg_min / 1026
            if 1026 - rbg_min > 0:
                plip_del = 1026 * (rbg_max - rbg_min) / (1026 - rbg_min)
                if plip_del > 0 and plip_add > 0:
                    local_a = plip_del / plip_add
                    local_b = math.log(plip_del / plip_add)
                    phi = local_a * local_b
                    logAMEE = logAMEE + phi
            n1 = n1 + L
        m1 = m1 + L
    logAMEE = 1026 - 1026 * ((1 - logAMEE / 1026) ** (1 / (number_n * number_m)))
    return logAMEE


def load_images(path):
    f = []
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            # f = list(p.rglob('*.*'))  # pathlib
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += sorted([x.replace('./', parent) if x.startswith('./') else x for x in t])  # local to global path
                # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{p} does not exist')
    img_files = []
    img_files += [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
    return img_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./runs/uciqe/uwcycle')
    parser.add_argument('--save-dir', default='./runs/uciqe/uwcycle')
    opt = parser.parse_args()
    uiqm1, uciqe1 = [], []

    uiqm1_uciqe1  = '{}/{}'.format(opt.save_dir, "uiqm_uciqe.txt")

    with open(uiqm1_uciqe1, 'a') as f:
        f.write(f'Uicm+Uism+Uiconmuciqe=uiqm')
        f.write(' ')
        f.write(f'delta+conl+mu=uciqe')
        f.write('\r\n')
    img_files  = load_images(opt.data_dir)
    for idx in range(0,len(img_files)):
        img_file = img_files[idx]
        img_name= (img_files[idx].split(os.sep)[-1]).split('.')[0]
        under = Image.open(img_file).convert('RGB')
        under = under.resize((300,300))
        #under.show()
        under = np.array(under)

        uiqm_test,Uicm, Uism, Uiconm=getuiqm(under)
        #uciqe_test,_,_,_ = get_uciqe(under)
        uciqe_test2, delta, conl, mu =get_uciqe2(under)
        uiqm1.append(uiqm_test)
        uciqe1.append(uciqe_test2)
        with open(uiqm1_uciqe1, 'a') as f:
            f.write(f'{img_name}')
            f.write(f'{round(Uicm, 4)}+{round(Uism, 4)}+{round(Uiconm, 4)}+={round(uiqm_test, 4)}')
            f.write(' ')
            f.write(f'{round(delta, 4)}+{round(conl, 4)}+{round(mu, 4)}+={round(uciqe_test2, 4)}')
            f.write(' ')
            f.write('\r\n')

    with open(uiqm1_uciqe1, 'a') as f:
            f.write(f'uiqm_avg')
            f.write(' ')
            f.write(f'uciqe_avg')
            f.write(' ')
            f.write('\r\n')
            f.write(f'{round(sum(uiqm1) / len(uiqm1), 4)}')
            f.write(' ')
            f.write(f'{round(sum(uciqe1) / len(uciqe1), 4)}')
            f.write(' ')





