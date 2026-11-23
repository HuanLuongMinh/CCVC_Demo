"""DAPG loss for the two CCVC sub-networks (spec 4.4, 4.5). Glue code written for CCVC-DAPN (not copied).

    loss, stats = compute_dapg(net, passes, args)

passes: {'l': (logits, labeled_img_mask), 'u_strong': (logits, mask), 'u_raw': (logits, mask)}, in this order.
For every pass and every branch b that has a DA module (da1 on branch 1, da2 on branch 2):

    L_pass = 1/2 [dapg_one(f1, da1) + dapg_one(f2, da2)]      (dapg_branches both)
           = dapg_one(f1, da1)                                 (dapg_branches b1)
    L^l = L_pass(labeled);  L^u = 1/2 [L_pass(strong) + L_pass(raw)]
    L_DAPG = 1/2 (L^l + L^u)  (dapg_on lu)   |   L^l  (dapg_on l: the unlabeled passes are not computed)

Each branch only sees its own feature and its own DA module: no DAPG gradient crosses branches.
Computed in fp32 with autocast off. Pixels whose (nearest-downsampled) mask is 255 are dropped (padding;
for labeled images also class 0 'unknown'). Fewer valid pixels than max_groups -> pass skipped (loss 0).
Non-finite features, or a NaN raised by DAPGLoss -> loss 0, warning, counted in stats['nan_skips'].
A skipped pass still returns 0 * (sum of the DA parameters) so every DA parameter keeps a gradient (DDP).
"""
import logging

import torch
import torch.nn.functional as F

from model.dapn.dapg_loss import DAPGLoss

PASSES = ('l', 'u_strong', 'u_raw')


def make_loss_fn(args):
    return DAPGLoss(margin=args.dapg_margin, lambda_inter=args.dapg_lambda_inter,
                    lambda_quality=args.dapg_lambda_quality)


def _zero(da, feat=None):
    z = sum(p.sum() for p in da.parameters()) * 0.0
    if feat is not None:
        z = z + feat.sum() * 0.0
    return z


def dapg_one(feat, da, loss_fn, mask):
    """feat (B, C, h, w), mask (B, H, W) -> (loss, stats or None, skip reason or None)."""
    with torch.autocast(device_type=feat.device.type, enabled=False):
        feat = feat.float()
        if not torch.isfinite(feat).all():
            return _zero(da), None, 'nan'
        valid = F.interpolate(mask[:, None].float(), size=feat.shape[-2:], mode='nearest')[:, 0] != 255
        f = feat.permute(0, 2, 3, 1)[valid]                       # (N, C)
        if f.shape[0] < da.max_groups:
            return _zero(da, feat), None, 'few'
        try:
            assign, proto, quality = da(f.t().reshape(1, -1, f.shape[0], 1))   # original 4-D API
            loss, d = loss_fn(f, assign, proto, quality)
        except ValueError as e:                                   # DAPGLoss raises on NaN
            logging.getLogger('dapn').warning('[DAPG] %s -> loss 0 for this pass', e)
            return _zero(da), None, 'nan'
        stats = {'loss': float(loss.detach()), 'intra': float(d['loss_intra'].detach()),
                 'inter': float(d['loss_inter'].detach()), 'quality': float(d['loss_quality'].detach()),
                 'n_valid_proto': int(proto.shape[0]), 'q_mean': float(quality.detach().mean()),
                 'n_pixels': int(f.shape[0])}
        return loss, stats, None


def compute_dapg(net, passes, args, loss_fn):
    das = net.da_modules()
    key = 'feature%d' if args.dapg_pos == 'feat' else 'c4_%d'
    pass_loss, per_branch, calls = {}, {b: [] for b in das}, []
    nan_skips = few_skips = 0
    branch_loss = {b: [] for b in das}
    for name in PASSES:
        if name not in passes or (name != 'l' and args.dapg_on == 'l'):
            continue
        logits, mask = passes[name]
        vals = []
        for b, da in das.items():
            loss, st, skip = dapg_one(logits[key % b], da, loss_fn, mask)
            calls.append((name, b))
            nan_skips += skip == 'nan'
            few_skips += skip == 'few'
            if st is not None:
                per_branch[b].append(st)
            branch_loss[b].append(float(loss.detach()))
            vals.append(loss)
        pass_loss[name] = sum(vals) / len(vals)

    zero = pass_loss['l'] * 0.0
    L_l = pass_loss['l']
    if args.dapg_on == 'lu':
        L_u = (pass_loss['u_strong'] + pass_loss['u_raw']) / 2
        L = (L_l + L_u) / 2
    else:
        L_u = zero
        L = L_l
    stats = {'L_dapg': float(L.detach()), 'L_dapg_l': float(L_l.detach()), 'L_dapg_u': float(L_u.detach()),
             'nan_skips': nan_skips, 'few_skips': few_skips, 'calls': calls,
             'pass_loss': {k: float(v.detach()) for k, v in pass_loss.items()}}
    for b in (1, 2):
        sts = per_branch.get(b, [])
        stats['L_dapg_b%d' % b] = (sum(branch_loss[b]) / len(branch_loss[b])) if branch_loss.get(b) else None
        for k in ('intra', 'inter', 'quality', 'n_valid_proto', 'q_mean'):
            stats['%s_b%d' % (k, b)] = (sum(s[k] for s in sts) / len(sts)) if sts else None
    return L, stats
