"""Restore exp/ and partitions/ from the output of a previous Kaggle session (plan 5.3).

    python tools/restore_output.py --restore_dir auto|none|<dir> [--search_root /kaggle/input]
                                   [--out_root /kaggle/working/exp/oem] [--part_root /kaggle/working/partitions/oem]

/kaggle/working is wiped between sessions and /kaggle/input is read-only: the previous session's output,
added as input data, is copied back. Rules:
- each run directory (one with config.json) is copied when missing locally, or when the source is further
  along (finished > more epochs done); a local run that is further along is never overwritten;
- trial/ and *_old_* directories are skipped;
- partitions: missing split directories are copied; an existing one with a different labeled/unlabeled list
  stops everything (exit 2), as does a different val.txt;
- free disk is checked first (needs copied size + 2 GB).
"""
import argparse
import filecmp
import json
import os
import shutil
import sys


def fail(msg):
    print('[restore] ERROR: ' + msg, flush=True)
    sys.exit(2)


def progress(run_dir):
    """(2, epochs) finished, (1, epochs_done) resumable, (0, 0) nothing usable."""
    if os.path.exists(os.path.join(run_dir, 'final_eval.json')):
        return (2, 10 ** 6)
    p = os.path.join(run_dir, 'progress.json')
    if os.path.exists(p) and os.path.exists(os.path.join(run_dir, 'last.pth')):
        with open(p, encoding='utf-8') as f:
            return (1, int(json.load(f).get('epochs_done', 0)))
    return (0, 0)


def run_dirs(exp_root):
    out = []
    for dirpath, dirnames, filenames in os.walk(exp_root):
        dirnames[:] = [d for d in dirnames if d != 'trial' and '_old_' not in d]
        if 'config.json' in filenames:
            out.append(os.path.relpath(dirpath, exp_root).replace(os.sep, '/'))
            dirnames[:] = []
    return out


def dir_size(path):
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)


def candidates(args):
    if args.restore_dir == 'none':
        return []
    if args.restore_dir != 'auto':
        if not os.path.isdir(args.restore_dir):
            fail('RESTORE_DIR=%s does not exist' % args.restore_dir)
        roots = [args.restore_dir]
    else:
        roots = []
        if os.path.isdir(args.search_root):
            for dirpath, dirnames, _ in os.walk(args.search_root):
                depth = dirpath[len(args.search_root):].count(os.sep)
                if os.path.isdir(os.path.join(dirpath, 'exp', 'oem')) or \
                        os.path.isdir(os.path.join(dirpath, 'partitions', 'oem')):
                    roots.append(dirpath)
                    dirnames[:] = []
                elif depth >= 4:
                    dirnames[:] = []
    work = os.path.abspath(os.path.dirname(os.path.dirname(args.out_root)))
    return [r for r in roots if os.path.abspath(r) != work]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_dir', default='auto')
    parser.add_argument('--search_root', default='/kaggle/input')
    parser.add_argument('--out_root', default='/kaggle/working/exp/oem')
    parser.add_argument('--part_root', default='/kaggle/working/partitions/oem')
    args = parser.parse_args()

    roots = candidates(args)
    if not roots:
        print('[restore] nothing to restore (RESTORE_DIR=%s)' % args.restore_dir)
        return
    print('[restore] previous outputs found: %s' % roots)

    # ---- partitions first: a mismatch must stop before anything is copied
    part_jobs = []
    for r in roots:
        src_part = os.path.join(r, 'partitions', 'oem')
        if not os.path.isdir(src_part):
            continue
        for name in sorted(os.listdir(src_part)):
            src, dst = os.path.join(src_part, name), os.path.join(args.part_root, name)
            if os.path.isdir(src):
                if os.path.isdir(dst):
                    for f in ('labeled.txt', 'unlabeled.txt'):
                        if os.path.exists(os.path.join(src, f)) and os.path.exists(os.path.join(dst, f)) and \
                                not filecmp.cmp(os.path.join(src, f), os.path.join(dst, f), shallow=False):
                            fail('partition %s/%s differs between %s and the local copy; splits must be identical '
                                 'to resume / stay nested' % (name, f, r))
                elif (src, dst) not in part_jobs:
                    part_jobs.append((src, dst))
            elif name == 'val.txt' and os.path.exists(dst) and not filecmp.cmp(src, dst, shallow=False):
                fail('val.txt differs between %s and the local copy' % r)
            elif not os.path.exists(dst) and (src, dst) not in part_jobs:
                part_jobs.append((src, dst))

    # ---- runs: keep the most advanced copy of each run directory
    best = {}
    for r in roots:
        src_exp = os.path.join(r, 'exp', 'oem')
        if not os.path.isdir(src_exp):
            continue
        for rel in run_dirs(src_exp):
            p = progress(os.path.join(src_exp, rel))
            if p[0] > 0 and (rel not in best or p > best[rel][0]):
                best[rel] = (p, os.path.join(src_exp, rel))
    run_jobs = []
    for rel, (p, src) in sorted(best.items()):
        dst = os.path.join(args.out_root, rel)
        local = progress(dst) if os.path.isdir(dst) else (0, 0)
        if p > local:
            run_jobs.append((rel, p, local, src, dst))
        else:
            print('[restore] keep local %s (local %s >= source %s)' % (rel, local, p))

    need = sum(dir_size(s) if os.path.isdir(s) else os.path.getsize(s) for s, _ in part_jobs) + \
        sum(dir_size(j[3]) for j in run_jobs)
    os.makedirs(args.out_root, exist_ok=True)
    free = shutil.disk_usage(args.out_root).free
    if need + 2 * 1024 ** 3 > free:
        fail('not enough disk: need %.1f GB (+2 GB margin), free %.1f GB' % (need / 1024 ** 3, free / 1024 ** 3))

    for src, dst in part_jobs:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print('[restore] partition %s <- %s' % (os.path.basename(dst), src))
    for rel, p, local, src, dst in run_jobs:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('trial', '*.tmp'))
        state = 'finished' if p[0] == 2 else '%d epochs done' % p[1]
        print('[restore] run %s <- %s (%s)' % (rel, src, state))
    print('[restore] done: %d partition item(s), %d run(s), %.2f GB copied' % (len(part_jobs), len(run_jobs),
                                                                              need / 1024 ** 3))


if __name__ == '__main__':
    main()
