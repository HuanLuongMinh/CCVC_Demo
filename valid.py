import argparse
from itertools import cycle
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def _predict(model, img, mode, args, cfg):
    """Softmax maps (B, C, H, W) of branch 1, branch 2 and of the averaged logits."""
    if mode == 'sliding_window':
        grid = args.crop_size
        b, _, h, w = img.shape
        final_1 = torch.zeros(b, cfg['nclass'], h, w, device=img.device)
        final_2 = torch.zeros(b, cfg['nclass'], h, w, device=img.device)
        final_ave = torch.zeros(b, cfg['nclass'], h, w, device=img.device)
        count = torch.zeros(b, 1, h, w, device=img.device)
        row = 0
        while row < h:
            col = 0
            while col < w:
                logits = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                pred1 = logits['pred1']
                pred2 = logits['pred2']
                pred_ave = (pred1 + pred2) / 2

                final_1[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred1.softmax(dim=1)
                final_2[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred2.softmax(dim=1)
                final_ave[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred_ave.softmax(dim=1)
                count[:, :, row: min(h, row + grid), col: min(w, col + grid)] += 1

                col += int(grid * 2 / 3)

            row += int(grid * 2 / 3)

        # dividing by the coverage count does not change the argmax (upstream) but gives probabilities
        return final_1 / count, final_2 / count, final_ave / count

    logits = model(img)
    pred1 = logits['pred1']
    pred2 = logits['pred2']
    pred_ave = (pred1 + pred2) / 2
    return pred1.softmax(dim=1), pred2.softmax(dim=1), pred_ave.softmax(dim=1)


def _denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((img.cpu() * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()


def evaluate(local_rank, model, loader, mode, args, cfg, vis_ids=None, gamma=0.9):
    """Upstream evaluation (mIoU of branch 1, branch 2 and of the averaged logits) plus, for the OEM runs:
    the conflict ratio on val, the confidence/accuracy of branch 1 and the predictions of the vis images."""
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    device = getattr(args, 'device', torch.device('cuda', max(local_rank, 0)))
    intersection_meter1 = AverageMeter()
    union_meter1 = AverageMeter()
    intersection_meter2 = AverageMeter()
    union_meter2 = AverageMeter()
    intersection_meter_ave = AverageMeter()
    union_meter_ave = AverageMeter()

    different_pred = 0
    total_pred = 0
    # pixel counts for the confidence / reliability curve of branch 1 (ignore pixels excluded)
    n_valid = n_conf = correct_conf = correct_nonconf = 0
    vis_ids = set(vis_ids or [])
    vis = []

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.to(device)

            if mode == 'center_crop':
                h, w = img.shape[-2:]
                start_h, start_w = (h - args.crop_size) // 2, (w - args.crop_size) // 2
                img = img[:, :, start_h:start_h + args.crop_size, start_w:start_w + args.crop_size]
                mask = mask[:, start_h:start_h + args.crop_size, start_w:start_w + args.crop_size]

            prob1, prob2, prob_ave = _predict(model, img, mode, args, cfg)
            conf1, pred1 = prob1.max(dim=1)
            pred2 = prob2.argmax(dim=1)
            pred_ave = prob_ave.argmax(dim=1)

            total_pred += img.shape[0]
            mask_dev = mask.to(device)
            valid = mask_dev != 255
            different_pred += ((pred1 != pred2) & valid).sum().item()
            n_valid += valid.sum().item()
            confident = (conf1 > gamma) & valid
            n_conf += confident.sum().item()
            correct = (pred1 == mask_dev) & valid
            correct_conf += (correct & confident).sum().item()
            correct_nonconf += (correct & valid & ~confident).sum().item()

            intersection1, union1, target1 = \
                intersectionAndUnion(pred1.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            intersection2, union2, target2 = \
                intersectionAndUnion(pred2.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            intersection_ave, union_ave, target_ave = \
                intersectionAndUnion(pred_ave.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            intersection_meter1.update(intersection1)
            union_meter1.update(union1)
            intersection_meter2.update(intersection2)
            union_meter2.update(union2)
            intersection_meter_ave.update(intersection_ave)
            union_meter_ave.update(union_ave)

            if id[0] in vis_ids:
                present = union1 > 0
                vis.append({'name': os.path.basename(id[0].split(' ')[0]),
                            'image': _denormalize(img[0]),
                            'pred1': pred1[0].cpu().numpy().astype(np.uint8),
                            'pred2': pred2[0].cpu().numpy().astype(np.uint8),
                            'gt': mask[0].numpy().astype(np.uint8),
                            'miou1': float(np.mean(intersection1[present] / union1[present]) * 100.0)
                            if present.any() else float('nan')})

    iou_class1 = intersection_meter1.sum / (union_meter1.sum + 1e-10)
    mIOU1 = np.mean(iou_class1) * 100.0
    iou_class2 = intersection_meter2.sum / (union_meter2.sum + 1e-10)
    mIOU2 = np.mean(iou_class2) * 100.0
    iou_class_ave = intersection_meter_ave.sum / (union_meter_ave.sum + 1e-10)
    mIOU_ave = np.mean(iou_class_ave) * 100.0

    result = {}

    result['IOU1'] = mIOU1
    result['IOU2'] = mIOU2
    result['IOU_ave'] = mIOU_ave
    result['iou_class1'] = iou_class1
    result['iou_class2'] = iou_class2
    result['iou_class_ave'] = iou_class_ave
    result['n_images'] = total_pred
    result['val_conflict_ratio'] = different_pred / max(n_valid, 1)
    n_nonconf = n_valid - n_conf
    result['confidence'] = {
        'n_valid': n_valid,
        'conf_ratio': n_conf / max(n_valid, 1),
        'acc_conf': correct_conf / n_conf if n_conf else float('nan'),
        'acc_nonconf': correct_nonconf / n_nonconf if n_nonconf else float('nan'),
        'acc_all': (correct_conf + correct_nonconf) / max(n_valid, 1),
    }
    result['vis'] = sorted(vis, key=lambda v: v['name'])

    return result


# if __name__ == '__main__':
#     evaluate()
