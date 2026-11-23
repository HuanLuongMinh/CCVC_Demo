"""Unit tests of the DAPN path on CPU (spec 9.1 tests 8-12 + DAPN on both sub-networks).

    python tools/tests_dapn/test_dapn_unit.py

Prints PASS/FAIL per check; exit code = number of failed checks. Random tensors, crop 64; the backbone is
built without pretrained weights.
"""
import os
import re
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.optim import SGD  # noqa: E402

import model.backbone.resnet as resnet  # noqa: E402
from model.semseg.deeplabv3plus_dapn import DiscrepancyDeepLabV3PlusDAPN  # noqa: E402
from model.dapn.ccvc_dapg import compute_dapg, dapg_one, make_loss_fn  # noqa: E402

torch.set_num_threads(2)
CFG = yaml.load(open(os.path.join(ROOT, 'configs', 'oem.yaml')), Loader=yaml.Loader)
FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, (' (%s)' % detail) if detail else ''), flush=True)
    if not cond:
        FAILS.append(name)


def args_for(**kw):
    a = SimpleNamespace(backbone='resnet101', mode_mapping='else', use_MLP=True, use_norm=True, use_dropout=True,
                        dropout=0.5, amp=False, seed=0, use_dapg=True, dapg_on='lu', dapg_branches='both',
                        dapg_pos='feat', w_dapg=0.2, da_max_groups=64, da_temperature=0.1, da_num_iters=3,
                        da_init='xavier', da_min_quality=0.1, da_ema_decay=0.0, dapg_margin=0.3,
                        dapg_lambda_inter=0.5, dapg_lambda_quality=0.1, base_lr=0.001, lr_backbone=1.0,
                        lr_network=10.0)
    a.__dict__.update(kw)
    return a


def build(a):
    orig = resnet.__dict__['resnet101']
    resnet.__dict__['resnet101'] = lambda pretrained, **kw: orig(False, **kw)
    try:
        torch.manual_seed(0)
        return DiscrepancyDeepLabV3PlusDAPN(a, CFG)
    finally:
        resnet.__dict__['resnet101'] = orig


def batch(seed, pad=True):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(2, 3, 64, 64, generator=g)
    lab = torch.randint(0, 8, (2, 64, 64), generator=g)
    if pad:
        lab[:, 48:, :] = 255                     # padding rows
        lab[0, :8, :8] = 255                     # 'unknown'
    return img, lab


def passes(net, seed=0):
    net.train()
    (li, lm), (ui, um), (ri, rm) = batch(seed), batch(seed + 1), batch(seed + 2)
    return {'l': (net(li), lm), 'u_strong': (net(ui), um), 'u_raw': (net(ri), rm)}


def params_of(module):
    return [p for p in module.parameters() if p.requires_grad]


def grads_nonzero(params, ps):
    gs = torch.autograd.grad(params, ps, retain_graph=True, allow_unused=True)
    return [g is not None and float(g.abs().sum()) > 0 for g in gs]


def main():
    a = args_for()
    net = build(a)
    lf = make_loss_fn(a)

    # ---------- DAPN on BOTH sub-networks ----------
    das = net.da_modules()
    check('ccvc_dapn: da1 on branch 1 and da2 on branch 2', sorted(das) == [1, 2] and das[1] is net.da1 and
          das[2] is net.da2 and das[1] is not das[2])
    keys = {k.split('.')[0] for k in net.state_dict() if k.startswith('da')}
    check('ccvc_dapn: checkpoint (state_dict) holds da1.* and da2.*', keys == {'da1', 'da2'}, sorted(keys))
    check('da1 / da2: separate prototype sets (64 x 304 each)',
          net.da1.prototypes.shape == (64, 304) and net.da2.prototypes.shape == (64, 304) and
          net.da1.prototypes.data_ptr() != net.da2.prototypes.data_ptr())

    P = passes(net)
    L, st = compute_dapg(net, P, a, lf)
    calls = st['calls']
    check('ccvc_dapn: DA called on 3 passes x 2 branches', sorted(calls) == sorted(
        [(p, b) for p in ('l', 'u_strong', 'u_raw') for b in (1, 2)]), calls)
    check('ccvc_dapn: L^l > 0 and L^u > 0', st['L_dapg_l'] > 0 and st['L_dapg_u'] > 0,
          'L^l %.4f L^u %.4f' % (st['L_dapg_l'], st['L_dapg_u']))
    check('ccvc_dapn: L_dapg_b1 > 0 and L_dapg_b2 > 0 (labeled and unlabeled passes)',
          st['L_dapg_b1'] > 0 and st['L_dapg_b2'] > 0 and all(v > 0 for v in st['pass_loss'].values()), st['pass_loss'])
    ref = ((st['pass_loss']['l']) + (st['pass_loss']['u_strong'] + st['pass_loss']['u_raw']) / 2) / 2
    check('L_DAPG = 1/2 (L^l + L^u), L^u = 1/2 (strong + raw)', abs(ref - st['L_dapg']) < 1e-6)

    # gradients: per branch isolation, DA parameters receive gradients
    b1 = params_of(net.branch1)
    b2 = params_of(net.branch2)
    loss_b = {}
    for b in (1, 2):
        vals = [dapg_one(P[p][0]['feature%d' % b], getattr(net, 'da%d' % b), lf, P[p][1])[0] for p in P]
        loss_b[b] = sum(vals) / 3
    g11 = grads_nonzero(loss_b[1], b1)
    g12 = grads_nonzero(loss_b[1], b2)
    g22 = grads_nonzero(loss_b[2], b2)
    g21 = grads_nonzero(loss_b[2], b1)
    check('DAPG of branch 1 reaches branch-1 parameters', any(g11))
    check('DAPG of branch 1 does not touch branch-2 parameters', not any(g12))
    check('DAPG of branch 2 reaches branch-2 parameters', any(g22))
    check('DAPG of branch 2 does not touch branch-1 parameters', not any(g21))
    check('DAPG of branch 1 does not touch da2; branch 2 does not touch da1',
          not any(grads_nonzero(loss_b[1], params_of(net.da2))) and not any(grads_nonzero(loss_b[2], params_of(net.da1))))
    net.zero_grad()
    L.backward()
    for b in (1, 2):
        da = getattr(net, 'da%d' % b)
        ok = da.prototypes.grad is not None and float(da.prototypes.grad.abs().sum()) > 0 and \
            all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in da.quality_net.parameters())
        check('da%d.prototypes and da%d.quality_net receive gradients' % (b, b), ok)

    # optimizer: the expression of CCVC_aug_dapn.py (unchanged origin code) puts da* in the lr x10 group
    src = open(os.path.join(ROOT, 'CCVC_aug_dapn.py'), encoding='utf-8').read()
    opt_src = re.search(r'    optimizer = SGD\(.*?weight_decay=1e-4\)\n', src, re.S).group(0)
    check('optimizer code of CCVC_aug_dapn.py == origin CCVC_aug.py',
          opt_src in open(os.path.join(ROOT, 'CCVC_aug.py'), encoding='utf-8').read())
    model, args = net, a
    optimizer = eval(opt_src.strip().split('=', 1)[1], {'SGD': SGD, 'model': model, 'args': args})
    grp = {id(p): k for k, g in enumerate(optimizer.param_groups) for p in g['params']}
    da_params = params_of(net.da1) + params_of(net.da2)
    check('optimizer: every da1/da2 parameter in group 2 (lr base x lr_network = x10)',
          all(grp.get(id(p)) == 2 for p in da_params) and
          abs(optimizer.param_groups[2]['lr'] - a.base_lr * a.lr_network) < 1e-12,
          'lr %g' % optimizer.param_groups[2]['lr'])

    # ---------- presets ----------
    al = args_for(dapg_on='l')
    _, stl = compute_dapg(net, P, al, lf)
    check('ccvc_dapn_l: L^u = 0 and DA not called on unlabeled passes', stl['L_dapg_u'] == 0.0 and
          all(p == 'l' for p, _ in stl['calls']) and sorted(b for _, b in stl['calls']) == [1, 2], stl['calls'])
    ab1 = args_for(dapg_branches='b1')
    net_b1 = build(ab1)
    check('ccvc_dapn_b1: da1 only, no da2', hasattr(net_b1, 'da1') and not hasattr(net_b1, 'da2'))
    _, stb = compute_dapg(net_b1, passes(net_b1), ab1, lf)
    check('ccvc_dapn_b1: DA only on branch 1', all(b == 1 for _, b in stb['calls']) and stb['L_dapg_b2'] is None)
    # ---------- --dapg_pos c4 (spec 4.3, test 17) ----------
    from model.semseg.deeplabv3plus import Discrepancy_DeepLabV3Plus
    ac4 = args_for(dapg_pos='c4')
    net_c4 = build(ac4)
    check('c4: da1 / da2 on layer4 (2048 channels)', net_c4.da1.prototypes.shape == (64, 2048) and
          net_c4.da2.prototypes.shape == (64, 2048))
    x = torch.randn(2, 3, 160, 160, generator=torch.Generator().manual_seed(5))
    net_c4.train()
    torch.manual_seed(3)
    o_c4 = net_c4(x)
    torch.manual_seed(3)
    o_or = Discrepancy_DeepLabV3Plus.forward(net_c4, x)
    check('c4, training: pred/feature identical to the origin forward, plus c4_1 / c4_2',
          all(torch.equal(o_c4[k], o_or[k]) for k in o_or) and o_c4['c4_1'].shape[1] == 2048 and
          o_c4['c4_2'].shape[1] == 2048)
    net_c4.eval()
    with torch.no_grad():
        e_c4, e_or = net_c4(x), Discrepancy_DeepLabV3Plus.forward(net_c4, x)
    check('c4, evaluation: exactly the origin outputs (no c4 keys)', sorted(e_c4) == sorted(e_or) and
          all(torch.equal(e_c4[k], e_or[k]) for k in e_or))
    net_c4.train()
    pc4 = {}
    for name, s in (('l', 20), ('u_strong', 21), ('u_raw', 22)):
        g = torch.Generator().manual_seed(s)
        img, lab = torch.randn(2, 3, 160, 160, generator=g), torch.randint(0, 8, (2, 160, 160), generator=g)
        pc4[name] = (net_c4(img), lab)
    Lc4, stc4 = compute_dapg(net_c4, pc4, ac4, make_loss_fn(ac4))
    Lc4.backward()
    check('c4: DAPG on both branches, every pass computed, grads on da1 and da2',
          stc4['L_dapg_b1'] > 0 and stc4['L_dapg_b2'] > 0 and stc4['few_skips'] == 0 and
          net_c4.da1.prototypes.grad is not None and net_c4.da2.prototypes.grad is not None,
          'L %.4f' % float(Lc4))
    del net_c4, pc4, o_c4, o_or

    aoff = args_for(use_dapg=False)
    net_off = build(aoff)
    check('ccvc_dapn_off: no DA module', not any(k.startswith('da') for k in net_off.state_dict()))

    # ---------- masks, skips, NaN, fp32 under autocast ----------
    feat = P['l'][0]['feature1'].detach().requires_grad_(True)     # the graph above was freed by backward
    lab = P['l'][1]
    _, s1, _ = dapg_one(feat, net.da1, lf, lab)
    exp_valid = int((torch.nn.functional.interpolate(lab[:, None].float(), size=feat.shape[-2:],
                                                     mode='nearest')[:, 0] != 255).sum())
    check('mask: pixels labelled 255 (padding / unknown) excluded', s1['n_pixels'] == exp_valid and
          exp_valid < feat.shape[0] * feat.shape[2] * feat.shape[3], '%d of %d' % (exp_valid, feat[:, 0].numel()))
    full = torch.full_like(lab, 255)
    l0, s0, why = dapg_one(feat, net.da1, lf, full)
    check('mask all 255 -> pass skipped, loss 0, no error', why == 'few' and float(l0) == 0.0 and s0 is None)
    net.zero_grad()
    l0.backward(retain_graph=True)
    check('skipped pass keeps a (zero) gradient on every DA parameter (DDP)',
          all(p.grad is not None for p in net.da1.parameters()))
    bad = feat.detach().clone()
    bad[0, 0, 0, 0] = float('nan')
    ln, sn, why = dapg_one(bad, net.da1, lf, lab)
    check('NaN feature -> loss 0, counted as nan skip', why == 'nan' and float(ln) == 0.0)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        lh, sh, _ = dapg_one(feat.detach().to(torch.bfloat16), net.da1, lf, lab)
    check('DAPG computed in fp32 under autocast (half input)', lh.dtype == torch.float32 and sh is not None,
          str(lh.dtype))

    # ---------- evaluation ----------
    import valid
    from valid_dapn import evaluate_dapn
    net_off.eval()
    loader = []
    for k in range(3):
        img, lab = batch(10 + k)
        loader.append((img[:1], lab[:1], ['img%d.tif lab%d.tif' % (k, k)]))
    eargs = SimpleNamespace(device=torch.device('cpu'), crop_size=64)
    torch.manual_seed(0)
    r0 = valid.evaluate(0, net_off, loader, 'original', eargs, CFG)
    torch.manual_seed(0)
    r1 = evaluate_dapn(0, net_off, loader, 'original', eargs, CFG)
    check('valid.evaluate and evaluate_dapn: same three mIoU and conflict ratio',
          all(r0[k] == r1[k] for k in ('IOU1', 'IOU2', 'IOU_ave', 'val_conflict_ratio')),
          '%.4f %.4f %.4f' % (r1['IOU1'], r1['IOU2'], r1['IOU_ave']))
    s = r1['agree_wrong_ratio'] + r1['agree_right_ratio'] + r1['conflict_ratio']
    check('agree_wrong + agree_right + conflict = 1', abs(s - 1) < 1e-9, '%.12f' % s)
    check('err_b1 = 1 - acc_all of branch 1', abs(r1['err_b1'] - (1 - r1['confidence']['acc_all'])) < 1e-9)
    cb = r1['confusion_b1']
    check('confusion_b1: 8x8, total = valid pixels, diagonal = correct pixels',
          cb.shape == (8, 8) and int(cb.sum()) == r1['confidence']['n_valid'] and
          int(np.trace(cb)) == round(r1['confidence']['acc_all'] * r1['confidence']['n_valid']))
    caw = r1['confusion_agree_wrong']
    check('confusion_agree_wrong: zero diagonal, total = agree_wrong pixels',
          int(np.trace(caw)) == 0 and abs(caw.sum() / r1['confidence']['n_valid'] - r1['agree_wrong_ratio']) < 1e-9)

    print('test_dapn_unit: %d failed' % len(FAILS))
    sys.exit(len(FAILS))


if __name__ == '__main__':
    main()
