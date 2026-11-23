"""Restore the CCVC-DAPN results of a previous Kaggle session (spec 6). tools/restore_output.py is not modified.

    python tools/restore_output_dapn.py --restore_dir auto|none|<dir> [--search_root /kaggle/input]
        [--out_root /kaggle/working/exp/oem_dapn] [--part_root /kaggle/working/partitions/oem_dapn]

Same rules as tools/restore_output.py (most advanced copy of each run kept, local runs never overwritten by
less advanced ones, trial/ and *_old_* skipped, partition mismatch -> exit 2, disk check), but only the trees
exp/oem_dapn and partitions/oem_dapn are read. exp/oem and partitions/oem (origin runs of run_baseline.sh)
are never copied: when a previous output holds them a warning is printed and they are skipped.
"""
import argparse
import filecmp
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restore_output import fail, progress, run_dirs, dir_size  # noqa: E402  (origin helpers, imported)

EXP, PART = os.path.join('exp', 'oem_dapn'), os.path.join('partitions', 'oem_dapn')
ORIGIN_TREES = (os.path.join('exp', 'oem'), os.path.join('partitions', 'oem'))


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
                has = [t for t in (EXP, PART) + ORIGIN_TREES if os.path.isdir(os.path.join(dirpath, t))]
                if has:
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
    parser.add_argument('--out_root', default='/kaggle/working/exp/oem_dapn')
    parser.add_argument('--part_root', default='/kaggle/working/partitions/oem_dapn')
    args = parser.parse_args()

    roots = candidates(args)
    for r in roots:
        for t in ORIGIN_TREES:
            if os.path.isdir(os.path.join(r, t)):
                print('[restore_dapn] WARNING: %s holds origin results (%s): skipped, only %s / %s are restored'
                      % (r, t.replace(os.sep, '/'), EXP.replace(os.sep, '/'), PART.replace(os.sep, '/')))
    roots = [r for r in roots if os.path.isdir(os.path.join(r, EXP)) or os.path.isdir(os.path.join(r, PART))]
    if not roots:
        print('[restore_dapn] nothing to restore (RESTORE_DIR=%s)' % args.restore_dir)
        return
    print('[restore_dapn] previous DAPN outputs found: %s' % roots)

    part_jobs = []
    for r in roots:
        src_part = os.path.join(r, PART)
        if not os.path.isdir(src_part):
            continue
        for name in sorted(os.listdir(src_part)):
            src, dst = os.path.join(src_part, name), os.path.join(args.part_root, name)
            if os.path.isdir(src):
                if os.path.isdir(dst):
                    for f in ('labeled.txt', 'unlabeled.txt'):
                        if os.path.exists(os.path.join(src, f)) and os.path.exists(os.path.join(dst, f)) and \
                                not filecmp.cmp(os.path.join(src, f), os.path.join(dst, f), shallow=False):
                            fail('partition %s/%s differs between %s and the local copy' % (name, f, r))
                elif (src, dst) not in part_jobs:
                    part_jobs.append((src, dst))
            elif name == 'val.txt' and os.path.exists(dst) and not filecmp.cmp(src, dst, shallow=False):
                fail('val.txt differs between %s and the local copy' % r)
            elif not os.path.exists(dst) and (src, dst) not in part_jobs:
                part_jobs.append((src, dst))

    best = {}
    for r in roots:
        src_exp = os.path.join(r, EXP)
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
            run_jobs.append((rel, p, src, dst))
        else:
            print('[restore_dapn] keep local %s (local %s >= source %s)' % (rel, local, p))

    need = sum(dir_size(s) if os.path.isdir(s) else os.path.getsize(s) for s, _ in part_jobs) + \
        sum(dir_size(j[2]) for j in run_jobs)
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
        print('[restore_dapn] partition %s <- %s' % (os.path.basename(dst), src))
    for rel, p, src, dst in run_jobs:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('trial', '*.tmp'))
        print('[restore_dapn] run %s <- %s (%s)' % (rel, src, 'finished' if p[0] == 2 else '%d epochs done' % p[1]))
    print('[restore_dapn] done: %d partition item(s), %d run(s), %.2f GB copied' % (len(part_jobs), len(run_jobs),
                                                                                  need / 1024 ** 3))


if __name__ == '__main__':
    main()
