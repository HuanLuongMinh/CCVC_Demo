"""Evaluate a saved checkpoint with the agreement metrics (spec 5.3) - origin or DAPN checkpoint.

    python tools/eval_ckpt.py --ckpt <run_dir>/final.pth [--config <run_dir>/config.json]
                              [--part-root <dir with val.txt> | --val-list <val.txt>] [--data-root <OEM root>]
                              [--out <run_dir>/eval_ckpt.json] [--device cuda|cpu] [--tol 0.01]

- The model is rebuilt from config.json of the run (same backbone / mapping / dropout settings, AMP as in
  training) and evaluated with valid_dapn.evaluate_dapn (same predictions as valid.evaluate), in the mode of
  final_eval.json ('original' by default).
- Origin checkpoints (no da* keys) and DAPN checkpoints load alike: only keys da1.* / da2.* may be unused
  (the DA modules play no role at evaluation); any other missing / unexpected key is an error.
- Writes eval_ckpt.json (three mIoU, agree_wrong_ratio, agree_right_ratio, conflict_ratio, err_b1, per class)
  and eval_ckpt_confusion_{b1,agree_wrong}.csv. The three mIoU must match final_eval.json within --tol
  (default 0.01), otherwise exit code 1.
- No pretrained weights are needed (the backbone is built without loading resnet101.pth).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import model.backbone.resnet as resnet  # noqa: E402
from dataset.aug_semi2 import SemiDataset  # noqa: E402
from model.semseg.deeplabv3plus_dapn import DiscrepancyDeepLabV3PlusDAPN  # noqa: E402
from valid_dapn import evaluate_dapn, write_confusion, AGREE_KEYS  # noqa: E402


def build_model(conf, cfg, device):
    a = SimpleNamespace(backbone=conf.get('backbone', 'resnet101'), mode_mapping=conf.get('mode_mapping', 'else'),
                        use_MLP=conf.get('use_MLP', True), use_norm=conf.get('use_norm', True),
                        use_dropout=conf.get('use_dropout', True), dropout=conf.get('dropout', 0.5),
                        amp=bool(conf.get('amp', True)) and device.type == 'cuda', use_dapg=False, seed=0)
    arch = a.backbone
    orig = resnet.__dict__[arch]
    resnet.__dict__[arch] = lambda pretrained, **kw: orig(False, **kw)     # weights come from the checkpoint
    try:
        return DiscrepancyDeepLabV3PlusDAPN(a, cfg)
    finally:
        resnet.__dict__[arch] = orig


def load_state(net, ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck and 'optimizer' in ck else ck
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    res = net.load_state_dict(sd, strict=False)
    unexpected = [k for k in res.unexpected_keys if not k.startswith(('da1.', 'da2.'))]
    if res.missing_keys or unexpected:
        raise RuntimeError('checkpoint does not fit the model: missing %s, unexpected %s' % (
            res.missing_keys[:10], unexpected[:10]))
    da = sorted({k.split('.')[0] for k in res.unexpected_keys})
    return 'dapn (%s ignored at evaluation)' % ', '.join(da) if da else 'origin'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default='')
    ap.add_argument('--final-eval', '--final_eval', default='')
    ap.add_argument('--part-root', '--part_root', default='')
    ap.add_argument('--val-list', '--val_list', default='')
    ap.add_argument('--data-root', '--data_root', default='')
    ap.add_argument('--cfg', default=os.path.join(ROOT, 'configs', 'oem.yaml'))
    ap.add_argument('--out', default='')
    ap.add_argument('--mode', default='')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num-workers', '--num_workers', type=int, default=2)
    ap.add_argument('--tol', type=float, default=0.01)
    a = ap.parse_args()

    run_dir = os.path.dirname(os.path.abspath(a.ckpt))
    conf = json.load(open(a.config or os.path.join(run_dir, 'config.json'), encoding='utf-8'))
    fe_path = a.final_eval or os.path.join(run_dir, 'final_eval.json')
    final = json.load(open(fe_path, encoding='utf-8')) if os.path.exists(fe_path) else None
    cfg = yaml.load(open(a.cfg, encoding='utf-8'), Loader=yaml.Loader)
    cfg['data_root'] = a.data_root or conf.get('data_root')
    val = a.val_list or (os.path.join(a.part_root, 'val.txt') if a.part_root else conf.get('val_list'))
    for what, p in (('data root', cfg['data_root']), ('val list', val)):
        if not p or not os.path.exists(p):
            sys.exit('[eval_ckpt] ERROR: %s not found: %r (use --data-root / --part-root)' % (what, p))
    mode = a.mode or (final or {}).get('eval_mode') or 'original'
    device = torch.device(a.device)

    net = build_model(conf, cfg, device)
    kind = load_state(net, a.ckpt)
    net.to(device)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', val_id_path=val)
    loader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=a.num_workers, drop_last=False)
    args = SimpleNamespace(device=device, crop_size=conf.get('crop_size', 512))
    print('[eval_ckpt] %s checkpoint %s | %d val images | mode %s | device %s | AMP %s' % (
        kind, a.ckpt, len(valset), mode, device, net.use_amp), flush=True)
    t0 = time.time()
    res = evaluate_dapn(0, net, loader, mode, args, cfg, gamma=conf.get('conf_threshold', 0.9))

    names = cfg.get('class_names')
    out = {'ckpt': os.path.abspath(a.ckpt), 'checkpoint_kind': kind, 'code_version': conf.get('code_version', 'origin'),
           'method': conf.get('method'), 'split': conf.get('split'), 'seed': conf.get('seed'), 'eval_mode': mode,
           'n_val': len(valset), 'eval_time_s': time.time() - t0,
           'miou_b1': res['IOU1'], 'miou_b2': res['IOU2'], 'miou_ave': res['IOU_ave'],
           'val_conflict_ratio': res['val_conflict_ratio']}
    out.update({k: res[k] for k in AGREE_KEYS})
    out['agree_wrong_class'] = res['agree_wrong_class']
    out['iou_b1'] = dict(zip(names, (100.0 * res['iou_class1']).tolist()))
    ok = True
    if final is not None:
        diffs = {k: abs(out[k] - float(final[k])) for k in ('miou_b1', 'miou_b2', 'miou_ave')}
        ok = all(v <= a.tol for v in diffs.values())
        out['check_vs_final_eval'] = {'final_eval': fe_path, 'tol': a.tol, 'abs_diff': diffs, 'ok': ok}
    else:
        out['check_vs_final_eval'] = None
    out_path = a.out or os.path.join(run_dir, 'eval_ckpt.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    base = os.path.splitext(out_path)[0]
    write_confusion(base + '_confusion_b1.csv', res['confusion_b1'], names)
    write_confusion(base + '_confusion_agree_wrong.csv', res['confusion_agree_wrong'], names)

    print('[eval_ckpt] mIoU b1 %.2f | b2 %.2f | ave %.2f' % (out['miou_b1'], out['miou_b2'], out['miou_ave']))
    print('[eval_ckpt] agree_wrong %.4f | agree_right %.4f | conflict %.4f | err_b1 %.4f' % tuple(
        out[k] for k in AGREE_KEYS))
    if final is not None:
        print('[eval_ckpt] vs final_eval.json: %s (max |diff| %.4f, tol %.2f)' % (
            'OK' if ok else 'MISMATCH', max(out['check_vs_final_eval']['abs_diff'].values()), a.tol))
    print('[eval_ckpt] -> %s' % out_path)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
