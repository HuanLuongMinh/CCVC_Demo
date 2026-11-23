"""Spec 4.1: the copied DAPN modules compute exactly what the original files compute.

    python tools/tests_dapn/test_dapn_equivalence.py [--orig_dir <dir with the two original files>]

Without --orig_dir the two files are downloaded from the pinned commit (DAPN_SOURCE.json) into a
temporary directory and their sha256 is checked. Minimal stubs of mmcv.runner.BaseModule and
mmseg.models.builder.MODELS/LOSSES let the original files import. Same seed + same tensors ->
assign, proto, quality, loss components and gradients must agree within 1e-6.
"""
import argparse, hashlib, importlib.util, json, os, sys, tempfile, types, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import torch
from torch import nn


def stub_mm():
    class _Reg(object):
        def register_module(self, *a, **k):
            return lambda c: c
    class BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()
    for name in ('mmcv', 'mmcv.runner', 'mmseg', 'mmseg.models', 'mmseg.models.builder',
                 'mmseg.models.losses', 'mmseg.models.utils'):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['mmcv.runner'].BaseModule = BaseModule
    sys.modules['mmseg.models.builder'].MODELS = _Reg()
    sys.modules['mmseg.models.builder'].LOSSES = _Reg()
    sys.modules['mmseg.models.losses'].__path__ = []
    sys.modules['mmseg.models.utils'].__path__ = []


def load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


def fetch(orig_dir):
    src = json.load(open(os.path.join(ROOT, 'DAPN_SOURCE.json')))
    paths = {}
    for rel, info in src['files'].items():
        p = os.path.join(orig_dir, os.path.basename(rel))
        if not os.path.exists(p):
            url = 'https://raw.githubusercontent.com/%s/%s/%s' % (
                src['repo'].split('github.com/')[1], src['commit'], rel)
            urllib.request.urlretrieve(url, p)
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        assert h == info['sha256'], 'original %s: sha256 %s != %s' % (rel, h, info['sha256'])
        paths[os.path.basename(rel)] = p
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig_dir', default='')
    a = ap.parse_args()
    tmp = None
    if not a.orig_dir:
        tmp = tempfile.TemporaryDirectory()
        a.orig_dir = tmp.name
    paths = fetch(a.orig_dir)
    stub_mm()
    o_da = load(paths['dynamic_anchor.py'], 'orig_dynamic_anchor')
    o_loss = load(paths['dapg_loss.py'], 'mmseg.models.losses.orig_dapg_loss')
    from model.dapn import DynamicAnchorModule, DAPGLoss

    kw = dict(feature_dim=304, max_groups=64, temperature=0.1, num_iters=3, init_method='xavier',
              min_quality=0.1, use_quality_gate=True, use_mask_predictor=False, ema_decay=0.0)
    worst = 0.0
    for seed, (n, scale) in enumerate([(500, 1.0), (2000, 3.0), (64, 0.2)]):
        outs = []
        for DA, L in ((o_da.DynamicAnchorModule, o_loss.DAPGLoss), (DynamicAnchorModule, DAPGLoss)):
            torch.manual_seed(100 + seed)
            da = DA(**kw)
            lf = L(margin=0.3, lambda_inter=0.5, lambda_quality=0.1)
            torch.manual_seed(7 + seed)
            f = (torch.randn(n, 304) * scale).requires_grad_(True)
            assign, proto, quality = da(f.t().reshape(1, 304, n, 1))
            loss, d = lf(f, assign, proto, quality)
            loss.backward()
            outs.append([assign, proto, quality, loss, d['loss_intra'], d['loss_inter'], d['loss_quality'],
                         f.grad, da.prototypes.grad] + [p.grad for p in da.quality_net.parameters()])
        for x, y in zip(*outs):
            assert x.shape == y.shape, (x.shape, y.shape)
            worst = max(worst, float((x.detach() - y.detach()).abs().max()) if x.numel() else 0.0)
    ok = worst <= 1e-6
    print('%s  DAPN copy == original (assign/proto/quality/loss/grad), max abs diff %.3g' % (
        'PASS' if ok else 'FAIL', worst))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
