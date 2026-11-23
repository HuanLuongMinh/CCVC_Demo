"""Is "agreed but wrong" located in entangled feature clusters? (spec 5.4, RQ0/RQ1; no training)

    python tools/diagnose_entanglement.py --ckpt <run_dir>/final.pth --branch 1|2
        [--k 64 --iters 20 --sample 200000 --seed 0 --n-seeds 3] [--part-root <dir with val.txt>] [--data-root <OEM>]
        [--out-dir <dir>] [--device cuda|cpu]

Works for origin and DAPN checkpoints (model rebuilt as in tools/eval_ckpt.py; da1/da2 unused).

1. Forward every val image (full image, eval mode, AMP as in training). Take the feature f of the chosen branch
   (logits['feature<b>']: 304 channels at 1/4 resolution; branch 2 after mapping, dropout off), the GT and the
   predictions y1, y2 (argmax of the full-resolution logits). GT, y1, y2 and a GT-boundary flag are brought to
   the feature grid by nearest sampling; pixels with GT 255 are dropped.
   Boundary flag: the full-resolution GT pixel has, within --boundary-radius (3) pixels, a valid pixel of
   another class.
2. Sample --sample pixels (the same quota from every image, then a uniform subsample), seed --seed.
3. EM with the formula of DynamicAnchorModule (cosine, tau 0.1, no quality gate, no EMA): K centres initialised
   from K random sampled pixels, then --iters steps of E: A = softmax(clamp(cos(f, P) / tau, -50, 50)),
   M: P = normalize(A^T f / sum A) (norm clamped to [0.1, 10] as in the module). One run per EM seed
   (--seed, --seed+1, ...; --n-seeds runs).
4. Second pass over val: every valid pixel -> cluster argmax cos(f, P). Cluster purity = share of its majority
   GT class over ALL its pixels; "mixed" cluster: purity < --mixed (0.6).
5. Three groups: agree-right (y1 = y2 = GT), agree-wrong (y1 = y2 != GT), conflict (y1 != y2). Per group:
   share of pixels, mean purity of their clusters, share in mixed clusters, share within the GT boundary band;
   agree-wrong only: share in a pure cluster whose majority class is the (wrong) predicted class.
   Overall and per GT class; mean +- std (ddof 1) over the EM seeds.

Outputs: diagnose_b<n>.json, diagnose_b<n>_per_class.csv next to the checkpoint (or --out-dir; read-only
checkpoint directory -> <repo>/../exp/oem_dapn/diagnose/).
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

GROUPS = ('agree_right', 'agree_wrong', 'conflict')
EPS = 1e-6


# ------------------------------------------------------------------------------------------------ data
def boundary_map(mask, radius, nclass):
    """(B, H, W) GT with 255 -> bool (B, H, W): a valid pixel of another class within `radius` pixels."""
    if radius <= 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    m = mask.float()[:, None]
    valid = m != 255
    k = 2 * radius + 1
    hi = F.max_pool2d(torch.where(valid, m, torch.full_like(m, -1.0)), k, stride=1, padding=radius)
    lo = -F.max_pool2d(-torch.where(valid, m, torch.full_like(m, float(nclass + 10))), k, stride=1, padding=radius)
    return (((hi != m) | (lo != m)) & valid)[:, 0]


def to_grid(x, size):
    """Nearest sampling of an integer / bool map (B, H, W) onto the feature grid."""
    return F.interpolate(x[:, None].float(), size=size, mode='nearest')[:, 0]


def val_pixels(model, loader, branch, device, nclass, radius, max_images=0):
    """Yields (f (N, C) float32, gt (N,), y1 (N,), y2 (N,), boundary (N,)) of the valid pixels of each image."""
    model.eval()
    with torch.no_grad():
        for k, (img, mask, _) in enumerate(loader):
            if max_images and k >= max_images:
                break
            img = img.to(device)
            mask = mask.to(device).long()
            logits = model(img)
            feat = logits['feature%d' % branch].float()
            h, w = feat.shape[-2:]
            y1, y2 = logits['pred1'].argmax(1), logits['pred2'].argmax(1)
            bnd = boundary_map(mask, radius, nclass)
            gt_g, y1_g, y2_g, b_g = (to_grid(t, (h, w)).long() for t in (mask, y1, y2, bnd))
            valid = gt_g != 255
            f = feat.permute(0, 2, 3, 1)[valid]
            yield f, gt_g[valid], y1_g[valid], y2_g[valid], b_g[valid].bool()


# ------------------------------------------------------------------------------------------------ EM
def sample_features(pixel_iter, n_images, sample, seed):
    g = torch.Generator().manual_seed(seed)
    quota = int(math.ceil(sample / max(n_images, 1)))
    parts = []
    for f, *_ in pixel_iter:
        idx = torch.randperm(f.shape[0], generator=g)[:quota].to(f.device)
        parts.append(f[idx].cpu())
    X = torch.cat(parts) if parts else torch.zeros(0, 1)
    if X.shape[0] > sample:
        X = X[torch.randperm(X.shape[0], generator=g)[:sample]]
    return X


def fit_em(X, k, iters, tau, seed, device):
    """EM of DynamicAnchorModule.forward (no quality gate / EMA) from k random sampled pixels."""
    X = X.to(device).float()
    g = torch.Generator().manual_seed(seed)
    if X.shape[0] < k:
        raise ValueError('only %d sampled pixels for k=%d clusters' % (X.shape[0], k))
    Xn = F.normalize(X, dim=1)
    Xn = torch.where(torch.isnan(Xn).any(dim=1, keepdim=True), torch.zeros_like(Xn), Xn)
    P = Xn[torch.randperm(X.shape[0], generator=g)[:k].to(device)].clone()
    for _ in range(iters):
        A = torch.softmax(torch.clamp(Xn @ P.t() / tau, min=-50, max=50), dim=1)
        size = A.sum(dim=0).clamp(min=EPS * 100)
        Pn = (A.t() @ X) / size[:, None]
        norms = torch.clamp(torch.norm(Pn, dim=1, keepdim=True), min=0.1, max=10.0)
        P = F.normalize(Pn / norms, dim=1)
    return P


def accumulate(pixel_iter, centres, nclass):
    """counts[s]: (3 groups, K, nclass GT, nclass y1, 2 boundary) for every centre set s."""
    K = centres[0].shape[0]
    counts = [torch.zeros(3 * K * nclass * nclass * 2, dtype=torch.float64) for _ in centres]
    for f, gt, y1, y2, b in pixel_iter:
        grp = torch.where(y1 != y2, 2, torch.where(y1 == gt, 0, 1))
        fn = F.normalize(f.float(), dim=1)
        for s, P in enumerate(centres):
            cl = (fn @ P.to(fn.device).t()).argmax(dim=1)
            idx = (((grp * K + cl) * nclass + gt) * nclass + y1) * 2 + b.long()
            counts[s] += torch.bincount(idx, minlength=counts[s].numel()).double().cpu()
    return [c.view(3, K, nclass, nclass, 2) for c in counts]


# ------------------------------------------------------------------------------------------------ metrics
def metrics(T, mixed_thr, gt_class=None):
    """Metrics of one count tensor T (3, K, C, C, 2); gt_class restricts the groups to GT = c (purity of the
    clusters is always computed over all their pixels)."""
    cl_gt = T.sum(dim=(0, 3, 4))                            # (K, C) every pixel of the cluster
    size = cl_gt.sum(dim=1)
    purity = torch.where(size > 0, cl_gt.max(dim=1)[0] / size.clamp(min=1), torch.full_like(size, float('nan')))
    maj = cl_gt.argmax(dim=1)
    pure = (purity >= mixed_thr) & (size > 0)
    mixed = (purity < mixed_thr) & (size > 0)
    Tg = T if gt_class is None else T[:, :, gt_class:gt_class + 1]
    n_all = float(Tg.sum())
    out = {'n_pixels': n_all, 'n_clusters_used': int((size > 0).sum()),
           'mean_purity_all_pixels': float((size * torch.nan_to_num(purity)).sum() / size.sum().clamp(min=1))
           if gt_class is None else float('nan')}
    for g, name in enumerate(GROUPS):
        Ng = Tg[g]                                          # (K, C', C, 2)
        n = float(Ng.sum())
        per_k = Ng.sum(dim=(1, 2, 3))
        d = {'share': n / n_all if n_all else float('nan'), 'n': n}
        if n > 0:
            d['mean_purity'] = float((per_k * torch.nan_to_num(purity)).sum() / n)
            d['in_mixed_cluster'] = float(per_k[mixed].sum() / n)
            d['near_boundary'] = float(Ng[..., 1].sum() / n)
            if name == 'agree_wrong':
                C = T.shape[2]
                hit = torch.zeros(Ng.shape[0], C, dtype=torch.bool)
                hit[torch.arange(Ng.shape[0]), maj] = True
                hit &= pure[:, None]                        # (K, y1): pure cluster whose majority = prediction
                by_pred = Ng.sum(dim=(1, 3))                # (K, y1)
                d['in_pure_cluster_of_predicted_class'] = float(by_pred[hit].sum() / n)
        else:
            d.update({'mean_purity': float('nan'), 'in_mixed_cluster': float('nan'), 'near_boundary': float('nan')})
            if name == 'agree_wrong':
                d['in_pure_cluster_of_predicted_class'] = float('nan')
        out[name] = d
    return out


def mean_std(values):
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return {'mean': float('nan'), 'std': float('nan')}
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if v.size > 1 else 0.0}


def flatten(m):
    out = {}
    for k, v in m.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out['%s.%s' % (k, k2)] = v2
        else:
            out[k] = v
    return out


def summarise(counts, mixed_thr, class_names):
    per_seed = [flatten(metrics(T, mixed_thr)) for T in counts]
    overall = {k: mean_std([p[k] for p in per_seed]) for k in per_seed[0]}
    per_class = {}
    for c, name in enumerate(class_names):
        ps = [flatten(metrics(T, mixed_thr, gt_class=c)) for T in counts]
        per_class[name] = {k: mean_std([p[k] for p in ps]) for k in ps[0] if k != 'mean_purity_all_pixels'}
    return per_seed, overall, per_class


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--branch', type=int, default=1, choices=[1, 2])
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--sample', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-seeds', '--n_seeds', type=int, default=3)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--mixed', type=float, default=0.6)
    ap.add_argument('--boundary-radius', '--boundary_radius', type=int, default=3)
    ap.add_argument('--config', default='')
    ap.add_argument('--part-root', '--part_root', default='')
    ap.add_argument('--val-list', '--val_list', default='')
    ap.add_argument('--data-root', '--data_root', default='')
    ap.add_argument('--cfg', default=os.path.join(ROOT, 'configs', 'oem.yaml'))
    ap.add_argument('--out-dir', '--out_dir', default='')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num-workers', '--num_workers', type=int, default=2)
    ap.add_argument('--max-images', '--max_images', type=int, default=0, help='tests: first n val images only')
    a = ap.parse_args()

    import yaml
    from torch.utils.data import DataLoader
    from dataset.aug_semi2 import SemiDataset
    from eval_ckpt import build_model, load_state

    run_dir = os.path.dirname(os.path.abspath(a.ckpt))
    conf = json.load(open(a.config or os.path.join(run_dir, 'config.json'), encoding='utf-8'))
    cfg = yaml.load(open(a.cfg, encoding='utf-8'), Loader=yaml.Loader)
    cfg['data_root'] = a.data_root or conf.get('data_root')
    val = a.val_list or (os.path.join(a.part_root, 'val.txt') if a.part_root else conf.get('val_list'))
    for what, p in (('data root', cfg['data_root']), ('val list', val)):
        if not p or not os.path.exists(p):
            sys.exit('[diagnose] ERROR: %s not found: %r (use --data-root / --part-root)' % (what, p))
    device = torch.device(a.device)
    nclass, names = cfg['nclass'], cfg['class_names']

    net = build_model(conf, cfg, device)
    kind = load_state(net, a.ckpt)
    net.to(device)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', val_id_path=val)
    loader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=a.num_workers, drop_last=False)
    n_img = min(len(valset), a.max_images) if a.max_images else len(valset)
    print('[diagnose] %s checkpoint %s | branch %d | %d val images | K %d | EM %d iters x %d seeds | sample %d | %s'
          % (kind, a.ckpt, a.branch, n_img, a.k, a.iters, a.n_seeds, a.sample, device), flush=True)

    t0 = time.time()
    it = lambda: val_pixels(net, loader, a.branch, device, nclass, a.boundary_radius, a.max_images)  # noqa: E731
    X = sample_features(it(), n_img, a.sample, a.seed)
    print('[diagnose] pass 1: %d sampled pixels x %d channels (%.0f s)' % (X.shape[0], X.shape[1], time.time() - t0),
          flush=True)
    seeds = [a.seed + s for s in range(a.n_seeds)]
    centres = [fit_em(X, a.k, a.iters, a.tau, s, device) for s in seeds]
    counts = accumulate(it(), centres, nclass)
    per_seed, overall, per_class = summarise(counts, a.mixed, names)
    print('[diagnose] pass 2: every valid pixel assigned (%.0f s)' % (time.time() - t0), flush=True)

    out_dir = a.out_dir or (run_dir if os.access(run_dir, os.W_OK) else
                            os.path.join(os.path.dirname(ROOT), 'exp', 'oem_dapn', 'diagnose',
                                         '%s_%s_seed%s' % (conf.get('split'), conf.get('method'), conf.get('seed'))))
    os.makedirs(out_dir, exist_ok=True)
    js = os.path.join(out_dir, 'diagnose_b%d.json' % a.branch)
    res = {'ckpt': os.path.abspath(a.ckpt), 'checkpoint_kind': kind, 'code_version': conf.get('code_version', 'origin'),
           'method': conf.get('method'), 'split': conf.get('split'), 'seed': conf.get('seed'), 'branch': a.branch,
           'settings': {'k': a.k, 'em_iters': a.iters, 'tau': a.tau, 'sample': a.sample, 'sample_seed': a.seed,
                        'em_seeds': seeds, 'mixed_threshold': a.mixed, 'boundary_radius_px_full_res': a.boundary_radius,
                        'n_val_images': n_img, 'feature': 'logits[feature%d], 1/4 resolution' % a.branch,
                        'std': 'ddof=1 over EM seeds'},
           'n_sampled': int(X.shape[0]), 'time_s': time.time() - t0,
           'overall': overall, 'per_seed': per_seed}
    with open(js, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    pc = os.path.join(out_dir, 'diagnose_b%d_per_class.csv' % a.branch)
    with open(pc, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'metric', 'mean', 'std'])
        for c in names:
            for k, v in per_class[c].items():
                w.writerow([c, k, '%.6g' % v['mean'], '%.6g' % v['std']])

    o = overall
    print('[diagnose] clusters used %.1f / %d | mean purity (all pixels) %.3f' % (
        o['n_clusters_used']['mean'], a.k, o['mean_purity_all_pixels']['mean']))
    print('[diagnose] %-12s %8s %14s %14s %14s' % ('group', 'share', 'mean purity', 'in mixed', 'near boundary'))
    for g in GROUPS:
        print('[diagnose] %-12s %8.4f %7.3f+-%.3f %7.3f+-%.3f %7.3f+-%.3f' % (
            g, o[g + '.share']['mean'], o[g + '.mean_purity']['mean'], o[g + '.mean_purity']['std'],
            o[g + '.in_mixed_cluster']['mean'], o[g + '.in_mixed_cluster']['std'],
            o[g + '.near_boundary']['mean'], o[g + '.near_boundary']['std']))
    print('[diagnose] agree_wrong in a pure cluster of the predicted class: %.3f+-%.3f' % (
        o['agree_wrong.in_pure_cluster_of_predicted_class']['mean'],
        o['agree_wrong.in_pure_cluster_of_predicted_class']['std']))
    print('[diagnose] -> %s, %s' % (js, pc))


if __name__ == '__main__':
    main()
