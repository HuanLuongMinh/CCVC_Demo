"""Confirm that this repository still contains CCVC origin (spec 2.3).

    python tools/verify_origin.py                       # steps 1, 2, 3
    python tools/verify_origin.py --steps 1             # integrity only (run_dapn.sh calls this before every run)
    python tools/verify_origin.py --steps 4 --part-root <dir> --baseline-part-root <dir> [--split-dir 1_4_s0]

1. integrity : sha256 of every file of ORIGIN_MANIFEST.sha256 (the tree of CCVC_Demo b44dc57) -> modified /
               missing files are listed.
2. behaviour : CCVC_aug.py, 5 iterations on CPU, synthetic data, fixed seeds (tools/golden_capture.py) ->
               losses of every iteration + checksum of the weights must equal tests/golden/origin_cpu_5iter.json.
               The golden file was generated from the clean tag ccvc-origin with the torch version it records;
               with another torch/numpy/python version this step reports SKIP (regenerate the golden file from
               the tag, README_DAPN.md) and step 3 compares against the fresh origin run instead.
3. superset  : CCVC_aug_dapn.py --use_dapg False under the same conditions -> identical to step 2 (and to the
               golden file when the versions match).
4. partitions: labeled.txt / unlabeled.txt of the split directories and val.txt of --part-root equal (sha256)
               those of --baseline-part-root (the baseline run).
Prints a PASS/FAIL table; exit code 1 when any step FAILs.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, 'tests', 'golden', 'origin_cpu_5iter.json')
VERSION_KEYS = ('torch', 'numpy', 'python')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def step_integrity(root=ROOT):
    bad = []
    n = 0
    with open(os.path.join(root, 'ORIGIN_MANIFEST.sha256'), encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            digest, name = line.split(' ', 1)
            name = name.lstrip('*').lstrip(' ')
            n += 1
            p = os.path.join(root, name)
            if not os.path.exists(p):
                bad.append('missing  ' + name)
            elif sha256(p) != digest:
                bad.append('modified ' + name)
    detail = '%d origin files, all identical to b44dc57' % n if not bad else '%d/%d differ: %s' % (
        len(bad), n, '; '.join(bad[:10]) + (' ...' if len(bad) > 10 else ''))
    return ('PASS' if not bad else 'FAIL'), detail


def capture(script, work, out, extra=''):
    cmd = [sys.executable, os.path.join(ROOT, 'tools', 'golden_capture.py'), '--script', script, '--work', work,
           '--out', out]
    if extra:
        cmd += ['--extra', extra]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError('golden_capture failed for %s:\n%s' % (script, r.stdout[-3000:]))
    return json.load(open(out, encoding='utf-8'))


def same(a, b):
    return a['losses'] == b['losses'] and a['weights_sha256'] == b['weights_sha256']


def first_diff(a, b):
    for k, (x, y) in enumerate(zip(a['losses'], b['losses'])):
        if x != y:
            return 'iteration %d: %s vs %s' % (k, x, y)
    return 'weights %s vs %s' % (a['weights_sha256'][:12], b['weights_sha256'][:12])


def compare_partitions(part_root, base_root, split_dir):
    rows = []
    dirs = [split_dir] if split_dir else sorted(
        d for d in os.listdir(part_root) if os.path.isdir(os.path.join(part_root, d)) and
        os.path.isdir(os.path.join(base_root, d)))
    if not dirs:
        return 'FAIL', 'no split directory common to %s and %s' % (part_root, base_root)
    files = [os.path.join(d, f) for d in dirs for f in ('labeled.txt', 'unlabeled.txt')] + ['val.txt']
    for rel in files:
        a, b = os.path.join(part_root, rel), os.path.join(base_root, rel)
        if not (os.path.exists(a) and os.path.exists(b)):
            rows.append('%s missing (%s)' % (rel, a if not os.path.exists(a) else b))
        elif sha256(a) != sha256(b):
            rows.append('%s differs' % rel)
    detail = '%d file(s) identical (%s)' % (len(files), ', '.join(dirs)) if not rows else '; '.join(rows)
    return ('PASS' if not rows else 'FAIL'), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', default='')
    ap.add_argument('--work', default='', help='working directory of steps 2-3 (default: temporary, removed)')
    ap.add_argument('--part-root', '--part_root', default='')
    ap.add_argument('--baseline-part-root', '--baseline_part_root', default='')
    ap.add_argument('--split-dir', '--split_dir', default='')
    ap.add_argument('--json-out', '--json_out', default='')
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(',') if s] or ([1, 2, 3] + ([4] if a.baseline_part_root else []))

    results = []
    if 1 in steps:
        results.append(('1 integrity',) + step_integrity())
    if 2 in steps or 3 in steps:
        work = a.work or tempfile.mkdtemp(prefix='verify_origin_')
        try:
            golden = json.load(open(GOLDEN, encoding='utf-8'))
            origin = capture(os.path.join(ROOT, 'CCVC_aug.py'), work, os.path.join(work, 'origin.json'))
            versions_match = all(origin.get(k) == golden.get(k) for k in VERSION_KEYS)
            vtxt = ', '.join('%s %s' % (k, origin.get(k)) for k in VERSION_KEYS)
            if 2 in steps:
                if not versions_match:
                    results.append(('2 behaviour', 'SKIP', 'golden made with %s; here %s -> regenerate the golden '
                                    'file from tag ccvc-origin' % (', '.join('%s %s' % (k, golden.get(k))
                                                                             for k in VERSION_KEYS), vtxt)))
                elif same(origin, golden):
                    results.append(('2 behaviour', 'PASS', 'CCVC_aug.py: %d iterations, losses and weights '
                                    'bit-identical to the golden file (%s)' % (len(golden['losses']), vtxt)))
                else:
                    results.append(('2 behaviour', 'FAIL', 'CCVC_aug.py differs from golden: ' + first_diff(origin, golden)))
            if 3 in steps:
                off = capture(os.path.join(ROOT, 'CCVC_aug_dapn.py'), work, os.path.join(work, 'dapn_off.json'),
                              '--use_dapg False')
                ref, name = (golden, 'golden file') if versions_match else (origin, 'fresh CCVC_aug.py run')
                if same(off, ref) and same(off, origin):
                    results.append(('3 superset', 'PASS', 'CCVC_aug_dapn.py --use_dapg False == %s (losses and '
                                    'weights bit-identical)' % name))
                else:
                    results.append(('3 superset', 'FAIL', 'CCVC_aug_dapn.py --use_dapg False differs from %s: %s'
                                    % (name, first_diff(off, ref))))
        except Exception as e:  # noqa: BLE001
            results.append(('2/3 behaviour', 'FAIL', '%s: %s' % (type(e).__name__, e)))
        finally:
            if not a.work:
                shutil.rmtree(work, ignore_errors=True)
    if 4 in steps:
        if not (a.part_root and a.baseline_part_root):
            results.append(('4 partitions', 'FAIL', 'needs --part-root and --baseline-part-root'))
        else:
            results.append(('4 partitions',) + compare_partitions(a.part_root, a.baseline_part_root, a.split_dir))

    print('verify_origin (%s)' % ROOT)
    print('  %-14s %-5s %s' % ('step', 'result', 'detail'))
    for name, res, detail in results:
        print('  %-14s %-5s %s' % (name, res, detail))
    failed = [r for r in results if r[1] == 'FAIL']
    print('verify_origin: %s' % ('FAIL' if failed else 'PASS'))
    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as f:
            json.dump({'result': 'FAIL' if failed else 'PASS',
                       'steps': [{'step': n, 'result': r, 'detail': d} for n, r, d in results]}, f, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
