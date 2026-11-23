"""Behaviour fingerprint of a CCVC training script on CPU (spec 2.3 steps 2-3, 9.1 tests 2, 4, 5).

    python tools/golden_capture.py --script <repo>/CCVC_aug.py --work <tmp dir> --out <json>
                                   [--at iter|init] [--iters 5] [--extra "--use_dapg False"]

Runs the script in this process (runpy) on synthetic OEM data, CPU, crop 64, the ccvc_aug preset of
run_baseline.sh, every seed fixed, one thread. Nothing is written inside the repository: the working
directory is --work (fake data, fake pretrained weights, splits, save_path).

--at iter : records L_total/L_sup/L_con/L_dis of the first --iters iterations at full precision (patched
            RunContext.after_iter, in memory only) and a checksum of every optimised parameter after them.
--at init : records a checksum of the model state_dict when training starts (keys da1.* / da2.* excluded),
            i.e. the initialisation of the origin part of the model.
The modules are imported from the repository that holds --script (sys.path[0] = its directory), so the
same harness fingerprints the clean tag ccvc-origin (worktree) and this repository.
"""
import argparse
import hashlib
import json
import os
import random
import runpy
import shlex
import shutil
import subprocess
import sys

PRESET = ('--backbone resnet101 --crop_size 64 --base_lr 0.001 --lr_backbone 1.0 --lr_network 10.0 --mul_scheduler 0.9 '
          '--mode_mapping else --conf_threshold 0.9 --conf_prob True --w_confident 2.0 --dropout 0.5 --use_SPL False '
          '--w_CE 2.0 --w_con 2.0 --w_dis 1.0 --mode_criterion CE --cc_mask_mixed True --use_MLP True --use_norm True '
          '--use_dropout True --mode_confident vote_threshold')


class _Stop(Exception):
    pass


def tensor_digest(tensors):
    import torch
    h = hashlib.sha256()
    total = 0.0
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
        total += float(t.double().sum())
    return h.hexdigest(), total


def prepare(repo, work, py):
    """Synthetic data + deterministic fake weights + split 1_4 (origin tools of the target repo)."""
    inp = os.path.join(work, 'input')
    if not os.path.isdir(os.path.join(inp, 'oem')):
        subprocess.check_call([py, os.path.join(repo, 'tools', 'tests', 'make_fake_oem.py'), '--out', inp,
                               '--n_train', '16', '--n_val', '4', '--size', '96'], stdout=subprocess.DEVNULL)
    pth = os.path.join(work, 'pretrained', 'resnet101.pth')
    if not os.path.exists(pth):
        import torch
        from model.backbone import resnet
        torch.manual_seed(12345)
        m = resnet.ResNet(resnet.Bottleneck, [3, 4, 23, 3], replace_stride_with_dilation=[False, False, True])
        sd = dict(m.state_dict())
        sd['fc.weight'] = torch.zeros(1000, 2048)
        sd['fc.bias'] = torch.zeros(1000)
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        torch.save(sd, pth)
    part = os.path.join(work, 'partitions')
    out = subprocess.check_output([py, os.path.join(repo, 'tools', 'prepare_oem.py'), '--split', '1_4',
                                   '--search_root', inp, '--part_root', part], universal_newlines=True)
    part_dir = [l.split('=', 1)[1] for l in out.splitlines() if l.startswith('PART_DIR=')][-1]
    info = json.load(open(os.path.join(part_dir, 'split_info.json'), encoding='utf-8'))
    return info['data_root'], part_dir, os.path.join(part, 'val.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--script', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--at', default='iter', choices=['iter', 'init'])
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--extra', default='')
    a = ap.parse_args()

    script = os.path.abspath(a.script)
    repo = os.path.dirname(script)
    work = os.path.abspath(a.work)
    out_path = os.path.abspath(a.out)
    os.makedirs(work, exist_ok=True)
    sys.path.insert(0, repo)

    import numpy as np
    import torch
    torch.set_num_threads(1)
    data_root, part_dir, val = prepare(repo, work, sys.executable)

    save = os.path.join(work, 'save_%s_%s' % (os.path.splitext(os.path.basename(script))[0], a.at))
    shutil.rmtree(save, ignore_errors=True)
    argv = ['--config', os.path.join(repo, 'configs', 'oem.yaml'), '--data_root', data_root,
            '--labeled_id_path', os.path.join(part_dir, 'labeled.txt'),
            '--unlabeled_id_path', os.path.join(part_dir, 'unlabeled.txt'), '--val_id_path', val,
            '--save_path', save, '--epochs', '80', '--batch_size', '2', '--num_workers', '0',
            '--seed', str(a.seed), '--gpus', '1', '--device', 'cpu', '--eval_every', '5',
            '--trial_iters', str(a.iters + 1), '--log_every', '1000'] + shlex.split(PRESET) + shlex.split(a.extra)

    import util.run_utils as ru
    record = {'losses': []}
    orig_after, orig_start = ru.RunContext.after_iter, ru.RunContext.start

    def after_iter(self, epoch, i, iters, optimizer, loss, loss_CE, loss_con, loss_dis, *rest, **kw):
        record['losses'].append([repr(float(x)) for x in (loss, loss_CE, loss_con, loss_dis)])
        r = orig_after(self, epoch, i, iters, optimizer, loss, loss_CE, loss_con, loss_dis, *rest, **kw)
        if len(record['losses']) >= a.iters:
            params = [p for g in optimizer.param_groups for p in g['params']]
            record['weights_sha256'], record['weights_sum'] = tensor_digest(params)
            record['n_params'] = len(params)
            raise _Stop()
        return r

    def start(self, model, *args, **kw):
        if a.at == 'init':
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith(('da1.', 'da2.'))}
            record['init_sha256'], record['init_sum'] = tensor_digest([sd[k] for k in sorted(sd)])
            record['init_keys'] = len(sd)
            record['da_keys'] = sorted({k.split('.')[0] for k in model.state_dict() if k.startswith('da')})
            raise _Stop()
        return orig_start(self, model, *args, **kw)

    ru.RunContext.after_iter, ru.RunContext.start = after_iter, start
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    cwd = os.getcwd()
    os.chdir(work)                                  # pretrained/resnet101.pth is read relative to the cwd
    sys.argv = [script] + argv
    try:
        runpy.run_path(script, run_name='__main__')
        raise RuntimeError('the script finished before the capture point')
    except _Stop:
        pass
    finally:
        os.chdir(cwd)
        ru.RunContext.after_iter, ru.RunContext.start = orig_after, orig_start

    record.update({'script': os.path.basename(script), 'at': a.at, 'iters': a.iters if a.at == 'iter' else 0,
                   'seed': a.seed, 'extra': a.extra, 'torch': torch.__version__, 'numpy': np.__version__,
                   'python': sys.version.split()[0], 'device': 'cpu', 'threads': 1,
                   'note': 'losses are repr(float) of L_total, L_sup, L_con, L_dis per iteration; '
                           'weights = every optimised parameter after the last recorded iteration'})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, indent=1)
        f.write('\n')
    print('[golden] %s -> %s' % (os.path.basename(script), out_path), flush=True)
    os._exit(0)                                      # skip the script's sys.exit / atexit handlers


if __name__ == '__main__':
    main()
