"""Environment report printed by run_baseline.sh before a run (information only; hard checks come later).

    python tools/check_env.py [--search_root /kaggle/input] [--work_dir /kaggle/working]
    python tools/check_env.py --gpu_count        # prints only the number of visible GPUs
"""
import argparse
import os
import shutil
import sys


def find_dirs(root, test, max_depth):
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath[len(root):].count(os.sep) >= max_depth:
            dirnames[:] = []
        if test(dirpath):
            out.append(dirpath)
            dirnames[:] = []
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_root', default='/kaggle/input')
    parser.add_argument('--work_dir', default='/kaggle/working')
    parser.add_argument('--gpu_count', action='store_true')
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print('[env] PyTorch is not installed')
        sys.exit(1)
    if args.gpu_count:
        print(torch.cuda.device_count())
        return

    print('[env] python %s | torch %s | CUDA %s' % (sys.version.split()[0], torch.__version__, torch.version.cuda))
    n = torch.cuda.device_count()
    for k in range(n):
        p = torch.cuda.get_device_properties(k)
        print('[env] GPU%d: %s, %.1f GB' % (k, p.name, p.total_memory / 1024 ** 3))
    if n == 0:
        print('[env] no GPU visible (Kaggle: Settings -> Accelerator -> GPU T4 x2)')

    data = find_dirs(args.search_root, lambda d: all(os.path.isdir(os.path.join(d, a, b))
                                                     for a in ('images', 'label') for b in ('train', 'val')), 5)
    print('[env] OEM data (images/ + label/): %s' % (data or 'NOT FOUND - add the OEM dataset'))
    weights = []
    for dirpath, _, files in os.walk(args.search_root) if os.path.isdir(args.search_root) else []:
        if dirpath[len(args.search_root):].count(os.sep) > 5:
            continue
        weights += [os.path.join(dirpath, f) for f in files if f == 'resnet101.pth']
    print('[env] resnet101.pth: %s' % (weights or 'NOT FOUND - add the model_lab dataset'))
    prev = find_dirs(args.search_root, lambda d: os.path.isdir(os.path.join(d, 'exp', 'oem')), 4)
    print('[env] previous session output (exp/oem): %s' % (prev or 'none (first session)'))
    if os.path.isdir(args.work_dir):
        print('[env] free disk in %s: %.1f GB' % (args.work_dir, shutil.disk_usage(args.work_dir).free / 1024 ** 3))


if __name__ == '__main__':
    main()
