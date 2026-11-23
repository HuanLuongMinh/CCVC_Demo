"""Evaluation of CCVC-DAPN (spec 3.3, 5.2): valid.evaluate + agreement metrics.

evaluate_dapn(...) has the signature of valid.evaluate and returns the same keys, computed the same way from
the same predictions (valid._predict is imported, not copied), plus:

    agree_wrong_ratio  #(y1 = y2 != GT) / #valid pixels       ("agreed but wrong")
    agree_right_ratio  #(y1 = y2 = GT)  / #valid
    conflict_ratio     #(y1 != y2)      / #valid               (= val_conflict_ratio)
    err_b1             #(y1 != GT)      / #valid
    agree_wrong_class  {class: #(GT = c, y1 = y2 != c) / #(GT = c)}
    confusion_b1             (nclass x nclass) rows GT, columns branch-1 prediction, pixel counts
    confusion_agree_wrong    same, restricted to agreed-but-wrong pixels (prediction = y1 = y2)
"""
import os

import numpy as np
import torch

from valid import _predict, _denormalize
from util.utils import AverageMeter, intersectionAndUnion

AGREE_KEYS = ('agree_wrong_ratio', 'agree_right_ratio', 'conflict_ratio', 'err_b1')


def evaluate_dapn(local_rank, model, loader, mode, args, cfg, vis_ids=None, gamma=0.9):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    device = getattr(args, 'device', torch.device('cuda', max(local_rank, 0)))
    nclass = cfg['nclass']
    intersection_meter1 = AverageMeter()
    union_meter1 = AverageMeter()
    intersection_meter2 = AverageMeter()
    union_meter2 = AverageMeter()
    intersection_meter_ave = AverageMeter()
    union_meter_ave = AverageMeter()

    different_pred = 0
    total_pred = 0
    n_valid = n_conf = correct_conf = correct_nonconf = 0
    n_agree_wrong = n_agree_right = n_err1 = 0
    conf_b1 = torch.zeros(nclass * nclass, dtype=torch.float64)
    conf_aw = torch.zeros(nclass * nclass, dtype=torch.float64)
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

            # ---- agreement metrics (spec 5.2)
            agree = (pred1 == pred2) & valid
            agree_wrong = agree & (pred1 != mask_dev)
            n_agree_wrong += agree_wrong.sum().item()
            n_agree_right += (agree & (pred1 == mask_dev)).sum().item()
            n_err1 += ((pred1 != mask_dev) & valid).sum().item()
            gt_v, p1_v = mask_dev[valid].long(), pred1[valid].long()
            conf_b1 += torch.bincount(gt_v * nclass + p1_v, minlength=nclass * nclass).double().cpu()
            gt_aw, p_aw = mask_dev[agree_wrong].long(), pred1[agree_wrong].long()
            conf_aw += torch.bincount(gt_aw * nclass + p_aw, minlength=nclass * nclass).double().cpu()

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

    nv = max(n_valid, 1)
    cb1 = conf_b1.view(nclass, nclass).numpy()
    caw = conf_aw.view(nclass, nclass).numpy()
    names = cfg.get('class_names', ['c%d' % c for c in range(nclass)])
    gt_count = cb1.sum(axis=1)
    result['agree_wrong_ratio'] = n_agree_wrong / nv
    result['agree_right_ratio'] = n_agree_right / nv
    result['conflict_ratio'] = different_pred / nv
    result['err_b1'] = n_err1 / nv
    result['agree_wrong_class'] = {n: (float(caw[c].sum() / gt_count[c]) if gt_count[c] else float('nan'))
                                   for c, n in enumerate(names)}
    result['confusion_b1'] = cb1.astype(np.int64)
    result['confusion_agree_wrong'] = caw.astype(np.int64)
    return result


def write_confusion(path, matrix, class_names):
    """CSV: first column GT class, then one column per predicted class (pixel counts)."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write('gt\\pred,' + ','.join(c.replace(' ', '_') for c in class_names) + '\n')
        for c, row in zip(class_names, np.asarray(matrix)):
            f.write(c.replace(' ', '_') + ',' + ','.join(str(int(v)) for v in row) + '\n')
