import torch
import torch.nn as nn
import numpy as np


class depthLoss:
    def __init__(self, opts):
        super(depthLoss, self).__init__()
        self.classify = getattr(opts, 'model_classify_or_not', False)
        self.train_cfg = dict(aux_loss=getattr(opts, "loss_depth_aux_loss", True),
                              aux_index=getattr(opts, "loss_depth_aux_index", [2, 4]),
                              aux_weight=getattr(opts, "loss_depth_aux_weight", [1 / 4, 1 / 4]),
                              main_index=getattr(opts, "loss_depth_main_index", [2, 4]))
        self.device = getattr(opts, "dev_device", torch.device("cpu"))
        self.align_corners = True
        self.sigloss = SigLoss(max_depth=1)
        self.gradloss = Loss_grad_normal()

    def __call__(self, result, depth_gt, task='object_detection'):  # 'object_enhancement'
        if task == 'object_detection':
            losses, output = self.loss_objectiont(result, depth_gt)
        else:
            losses, output = self.loss_enhancement(result, depth_gt)

        return losses, output

    def loss_enhancement(self, result, depth_gt):
        pred_clean, pred_back, pred_trans = result['img'], result['back'], result['trans']

        clean, simulates, lights, transmissions, depths, water_types, water_depths = (depth_gt['clean'],
                                                                                      depth_gt['simulate'],
                                                                                      depth_gt['backlight'],
                                                                                      depth_gt['transmission'],
                                                                                      depth_gt['depth'],
                                                                                      depth_gt['water_type'],
                                                                                      depth_gt['water_depth'])

        #compute_clean = ((simulates - pred_back * (1 - pred_trans)) / (pred_trans  + 0.000001)).clamp(0, 1)
        compute_clean = ((simulates - pred_back* (1 - pred_trans)) / ( pred_back*pred_trans + 0.000001)).clamp(0, 1)
        compute_I = (pred_clean * pred_trans*pred_back + (1-pred_trans)*pred_back).clamp(0, 1)
        #compute_back = (simulates / (pred_clean * pred_trans + 1 - pred_trans + 0.0001)).clamp(0, 1)
        pred_trans = (pred_trans.clamp(0, 1))
        pred_back = (pred_back.clamp(0, 1))
        pred_clean = (pred_clean.clamp(0, 1))

        loss_clean = nn.functional.l1_loss(pred_clean, clean)
        loss_clean_phy = nn.functional.l1_loss(pred_trans, transmissions) +nn.functional.l1_loss(pred_back, lights)

        loss_trans, loss_trans_grad, loss_trans_normal = self.gradloss(pred_trans, transmissions)
        loss_trans_sig = self.sigloss(pred_trans, transmissions)

        loss_light, loss_light_grad, loss_light_normal = self.gradloss(pred_back, lights)
        loss_light_sig = self.sigloss(pred_back, lights)

        loss_enhance = loss_clean + 0.5 * loss_clean_phy +\
                       (0.5 * loss_trans + 0.5 * loss_trans_grad + 0.5 * loss_trans_normal + 0.1 * loss_trans_sig + \
                       0.5 * loss_light + 0.5 * loss_light_grad + 0.5 * loss_light_normal + 0.1 * loss_light_sig)

        losses = dict()
        losses["loss_enhance"] = loss_enhance
        losses["loss_clean"] = loss_clean
        losses["loss_clean_phy"] = loss_clean_phy

        losses["loss_trans"] = loss_trans
        losses["loss_trans_grad"] = loss_trans_grad
        losses["loss_trans_normal"] = loss_trans_normal
        losses["loss_trans_sig"] = loss_trans_sig

        losses["loss_light"] = loss_light
        losses["loss_light_grad"] = loss_light_grad
        losses["loss_light_normal"] = loss_light_normal
        losses["loss_light_sig"] = loss_light_sig

        output = {"pred_clean": pred_clean, "pred_trans": pred_trans, "pred_light": pred_back,
                  "compute_clean": compute_clean, "compute_I": compute_I}

        return losses, output

    def loss_objectiont(self, result, depth_gt):
        pred_clean, pred_back, pred_trans = result['img'], result['back'], result['trans']
        under = depth_gt

        pred_trans = pred_trans.clamp(0, 1)
        pred_back = pred_back.clamp(0, 1)
        pred_clean = pred_clean.clamp(0, 1)
        compute_clean = ((under - pred_back*(1-pred_trans)) / (pred_back*pred_trans  + 0.000001)).clamp(0, 1)
        compute_trans = 0.5*((under - pred_back) / (pred_clean * pred_back - pred_back + 0.0001)).clamp(0, 1)
        compute_back = 0.5*(under / (pred_clean * pred_trans + 1 - pred_trans + 0.0001)).clamp(0, 1)
        compute_I = (pred_clean * pred_trans*pred_back + (1-pred_trans)*pred_back).clamp(0, 1)

        loss_clean_phy = nn.functional.l1_loss(compute_clean, pred_clean)

        loss_trans, loss_trans_grad, loss_trans_normal = self.gradloss(pred_trans, compute_trans)
        loss_trans_sig = self.sigloss(pred_trans, compute_trans)

        loss_light, loss_light_grad, loss_light_normal = self.gradloss(pred_back, compute_back)
        loss_light_sig = self.sigloss(pred_back, compute_back)

        loss_enhance = 0.1 * loss_clean_phy + \
                       (0.05 * loss_trans + 0.05 * loss_trans_grad + 0.05 * loss_trans_normal + 0.01 * loss_trans_sig +
                        0.05 * loss_light + 0.05 * loss_light_grad + 0.05 * loss_light_normal + 0.01 * loss_light_sig)

        losses = dict()
        losses["loss_enhance"] = loss_enhance
        losses["loss_clean_phy"] = loss_clean_phy

        losses["loss_trans"] = loss_trans
        losses["loss_trans_grad"] = loss_trans_grad
        losses["loss_trans_normal"] = loss_trans_normal
        losses["loss_trans_sig"] = loss_trans_sig

        losses["loss_light"] = loss_light
        losses["loss_light_grad"] = loss_light_grad
        losses["loss_light_normal"] = loss_light_normal
        losses["loss_light_sig"] = loss_light_sig

        output = {"pred_clean": pred_clean, "pred_trans": pred_trans, "pred_light": pred_back,
                  "compute_clean": compute_clean, "compute_I": compute_I}
        return losses, output


class SigLoss(nn.Module):
    """SigLoss.
        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.
    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self, valid_mask=True, max_depth=None, warm_up=False, warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask

        self.max_depth = max_depth
        self.eps = 0.001  # avoid grad explode
        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def forward(self, input, target):
        loss = torch.zeros(1).cuda()  # loss_depth, loss_grad,loss_normal
        ch = input.size(1)
        for index in range(0, ch + 1):
            if index == ch:
                tmp_input, tmp_gt = input, target
            else:
                tmp_input = input[:, index][:, None, :, :]
                tmp_gt = target[:, index][:, None, :, :]
            loss = loss + self.sigloss(tmp_input, tmp_gt)
        return loss

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)
        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2
        return torch.sqrt(Dg)


# loss_depth+loss_grad+loss_normal

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        bs, c, h, w = x.shape
        out_x = []
        out_y = []
        for index in range(0, c):
            tmp_ch = x[:, index][:, None, :, :]
            out = self.edge_conv(tmp_ch)
            out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
            grad_dx = out[:, 0, :, :].contiguous().view_as(tmp_ch)
            grad_dy = out[:, 1, :, :].contiguous().view_as(tmp_ch)
            out_x.append(grad_dx)
            out_y.append(grad_dy)
            # self.save_result('train', f'{index}', 'under_pred_clean', tmp_ch, grad_dx, grad_dy)

        grad_x = torch.cat(out_x, 1)
        grad_y = torch.cat(out_y, 1)
        out_x.append(grad_x)
        out_y.append(grad_y)
        # out = {'grad_x':grad_x ,'grad_y':grad_y }
        # self.save_result('train', f'{index + 1}', 'under_pred_clean', x, grad_dx, grad_dy)

        return out_x, out_y

    def save_result(self, mod, epoch, title, x_qry, y_qry, outputs):
        from matplotlib import pyplot as plt
        # -------------------------------end finetunning and save some results----------------------------------------
        temp_input = x_qry
        temp_label = y_qry
        temp_out = outputs

        temp_input = temp_input.detach().cpu().numpy()
        temp_label = temp_label.detach().cpu().numpy()
        temp_out = temp_out.detach().cpu().numpy()
        num_img = 5 if len(temp_out) > 6 else len(temp_out)
        fig, ax = plt.subplots(num_img, 3, figsize=(24, 24))
        for index in range(0, num_img):
            temp_input_view = self.imagetensor2view(temp_input[index])
            temp_label_view = self.imagetensor2view(temp_label[index])
            temp_out_view = self.imagetensor2view(temp_out[index])
            if len(temp_out) == 1:
                ax[0].imshow(temp_input_view)
                ax[1].imshow(temp_label_view)
                ax[2].imshow(temp_out_view)
            else:
                ax[index][0].imshow(temp_input_view)
                ax[index][1].imshow(temp_label_view)
                ax[index][2].imshow(temp_out_view)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        f = '{}/{}'.format('./runs/train', f'{mod}_{title}_epoch_{epoch}.png')
        plt.title(title, x=-1.4, y=-0.6)

        plt.savefig(f, bbox_inches='tight', dpi=fig.dpi, pad_inches=0)
        plt.close()

    def imagetensor2view(self, imagetesnor):
        channel = imagetesnor.shape[0]
        imagetensor = imagetesnor.transpose((1, 2, 0))
        imagetensor = (imagetensor * 255).astype(np.uint8)

        if channel == 1:
            return imagetensor.squeeze()
        else:
            return imagetensor


class Loss_grad_normal:
    def __init__(self):
        super(Loss_grad_normal, self).__init__()
        self.get_gradient = Sobel().cuda()

    def __call__(self, pred, gt):
        bs, c, h, w = pred.shape
        ones = torch.ones(pred.size(0), c, pred.size(2), pred.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        ones_ch = torch.ones(pred.size(0), 1, pred.size(2), pred.size(3)).float().cuda()
        ones_ch = torch.autograd.Variable(ones_ch)
        cos = nn.CosineSimilarity(dim=1, eps=0)

        gt_grad_dx, gt_grad_dy = self.get_gradient(gt)
        pred_grad_dx, pred_grad_dy = self.get_gradient(pred)

        loss = torch.zeros(3).cuda()  # loss_depth, loss_grad,loss_normal
        for index in range(0, len(gt_grad_dx)):
            tmp_gt_dx, tmp_gt_dy = gt_grad_dx[index], gt_grad_dy[index]
            tmp_pred_dx, tmp_pred_dy = pred_grad_dx[index], pred_grad_dy[index]
            if index == len(gt_grad_dx) - 1:
                tmp_pred, tmp_gt = pred, gt
            else:
                tmp_pred, tmp_gt = pred[:, index][:, None, :, :], gt[:, index][:, None, :, :]
            if c == 1:
                tmp_pred_normal = torch.cat((-tmp_pred_dx, -tmp_pred_dy, ones_ch), 1)
                tmp_gt_normal = torch.cat((-tmp_gt_dx, -tmp_gt_dy, ones_ch), 1)
            else:
                tmp_pred_normal = torch.cat((-tmp_pred_dx, -tmp_pred_dy, ones), 1)
                tmp_gt_normal = torch.cat((-tmp_gt_dx, -tmp_gt_dy, ones), 1)
            #tmp_loss_depth = nn.functional.l1_loss(tmp_pred,tmp_gt)
            tmp_loss_depth = torch.log(torch.abs(tmp_pred - tmp_gt) + 1).mean()

            tmp_loss_dx = torch.log(torch.abs(tmp_pred_dx - tmp_gt_dx) + 1).mean()
            tmp_loss_dy = torch.log(torch.abs(tmp_pred_dy - tmp_gt_dy) + 1).mean()
            tmp_loss_grad = (tmp_loss_dx + tmp_loss_dy)

            tmp_loss_normal = torch.abs(1 - cos(tmp_pred_normal, tmp_gt_normal)).mean()
            loss[0] = loss[0] + tmp_loss_depth
            loss[1] = loss[1] + tmp_loss_grad
            loss[2] = loss[2] + tmp_loss_normal

        return loss[0], loss[1], loss[2]
