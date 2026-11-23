"""Helper of tools/run_diag_origin.sh (spec "chay chan doan origin", sections 3.1, 3.2, 3.4).

    python tools/summarize_diag_origin.py find      --search-root /kaggle/input [--run-1-4 D] [--run-1-8 D]
                                                    [--val-list F] [--data-root D] [--splits "1_4 1_8"] --env-out F
    python tools/summarize_diag_origin.py preflight --env F [--n-val 384] [--out-root D] [--min-free-gb 1]
    python tools/summarize_diag_origin.py summary   --out-root D [--splits "1_4 1_8"]

find      : locates the origin run directories (final.pth + config.json with method ccvc_aug, split, seed 0),
            val.txt and the OEM data root; writes KEY=value lines to --env-out and prints the chosen paths.
            0 or several candidates -> exit 2 with the list (copies of the same final.pth count once).
preflight : files of every run, val.txt line count + every image/label present, val.txt checksum (when one is
            recorded), free disk space. Any error -> exit 2.
summary   : reads the outputs of eval_ckpt.py / diagnose_entanglement.py under --out-root and writes summary.md
            and summary.csv (numbers only, no interpretation). Missing outputs -> "chưa có (xem run_log)".
Reads outputs only; computes nothing new except err_b1 - conflict_ratio (former lower bound) and the
agree_* sum check.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys

MISSING = 'chưa có (xem run_log)'
GROUPS = ('agree_right', 'agree_wrong', 'conflict')


def err(msg):
    print('[diag_origin] ERROR: ' + msg, flush=True)
    sys.exit(2)


def load(p):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def walk(root, max_depth):
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
        dirnames.sort()
        yield dirpath, dirnames, filenames


# ------------------------------------------------------------------------------------------------ find
def is_origin_run(d, split):
    c = load(os.path.join(d, 'config.json')) or {}
    return c.get('method') == 'ccvc_aug' and str(c.get('split')) == split and str(c.get('seed')) == '0'


def find_runs(search_root, split):
    hits = [d for d, _, files in walk(search_root, 12) if 'final.pth' in files and is_origin_run(d, split)]
    if len(hits) > 1:                       # the same run restored into several output versions counts once
        by_sha = {}
        for d in hits:
            by_sha.setdefault(sha256(os.path.join(d, 'final.pth')), []).append(d)
        if len(by_sha) == 1:
            print('[diag_origin] split %s: %d copies of the same final.pth, using %s' % (split, len(hits), hits[0]))
            return [hits[0]]
    return hits


def is_oem_val_list(p):
    try:
        with open(p, encoding='utf-8') as f:
            first = f.readline()
    except Exception:
        return False
    return first.startswith('images/val/')


def find_val(runs, search_root):
    cands = [c.get('val_list') for c in (load(os.path.join(r, 'config.json')) or {} for r in runs)]
    cands = [p for p in cands if p and os.path.isfile(p)]
    source = 'config.json val_list'
    if not cands:
        source = 'val.txt under %s' % search_root
        cands = [os.path.join(d, 'val.txt') for d, _, files in walk(search_root, 10) if 'val.txt' in files]
        cands = [p for p in cands if is_oem_val_list(p)]
    by_sha = {}
    for p in cands:
        by_sha.setdefault(sha256(p), []).append(p)
    if len(by_sha) != 1:
        err('need exactly one OEM val.txt (%s), found %d different file(s):\n  %s\npass --part-root <dir with '
            'val.txt>.' % (source, len(by_sha), '\n  '.join('%s  %s' % (s[:12], ', '.join(v)) for s, v in
                                                          by_sha.items()) or '-'))
    paths = list(by_sha.values())[0]
    return paths[0], source + ('' if len(paths) == 1 else ' (%d identical copies)' % len(paths))


def has_val_data(d):
    return all(os.path.isdir(os.path.join(d, a, 'val')) for a in ('images', 'label'))


def find_data_root(runs, search_root):
    """Same rule as tools/prepare_oem.py find_data_root (depth <= 5, one hit), on images/val + label/val."""
    for r in runs:
        p = (load(os.path.join(r, 'config.json')) or {}).get('data_root')
        if p and has_val_data(p):
            return p, 'config.json data_root'
    root = os.path.abspath(search_root)
    hits = []
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath[len(root):].count(os.sep) > 5:
            dirnames[:] = []
            continue
        if has_val_data(dirpath):
            hits.append(dirpath)
            dirnames[:] = []
    if len(hits) != 1:
        err('need exactly one directory with images/val and label/val under %s, found %d: %s\npass --data-root.'
            % (search_root, len(hits), hits))
    return hits[0], 'search under %s' % search_root


def cmd_find(a):
    splits = a.splits.split()
    given = {'1_4': a.run_1_4, '1_8': a.run_1_8}
    runs = {}
    for s in splits:
        if given.get(s):
            d = os.path.abspath(given[s])
            if not os.path.isfile(os.path.join(d, 'final.pth')):
                err('--run-%s %s: no final.pth there' % (s.replace('_', '-'), d))
            runs[s] = d
            continue
        if not os.path.isdir(a.search_root):
            err('search root %s does not exist; pass --run-%s' % (a.search_root, s.replace('_', '-')))
        hits = find_runs(a.search_root, s)
        if len(hits) != 1:
            err('split %s: need exactly one final.pth with config.json (method ccvc_aug, split %s, seed 0) under %s, '
                'found %d:\n  %s\npass --run-%s <run dir>.' % (s, s, a.search_root, len(hits),
                                                               '\n  '.join(hits) or '-', s.replace('_', '-')))
        runs[s] = hits[0]
    if a.val_list:
        val, vsrc = os.path.abspath(a.val_list), 'given'
    else:
        val, vsrc = find_val(list(runs.values()), a.search_root)
    if a.data_root:
        data, dsrc = os.path.abspath(a.data_root), 'given'
    else:
        data, dsrc = find_data_root(list(runs.values()), a.search_root)
    rows = [('run %s' % s.replace('_', '/'), runs[s], 'given' if given.get(s) else 'found') for s in splits]
    rows += [('val.txt', val, vsrc), ('data root', data, dsrc)]
    print('[diag_origin] paths:')
    for name, p, src in rows:
        print('  %-10s %s   (%s)' % (name, p, src))
    with open(a.env_out, 'w', encoding='utf-8', newline='\n') as f:
        for s in splits:
            f.write('RUN_%s=%s\n' % (s, runs[s]))
        f.write('VAL=%s\nDATA=%s\n' % (val, data))


# ------------------------------------------------------------------------------------------------ preflight
def read_env(p):
    env = {}
    with open(p, encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                k, v = line.rstrip('\n').split('=', 1)
                env[k] = v
    return env


def cmd_preflight(a):
    env = read_env(a.env)
    problems = []
    runs = {k[4:]: v for k, v in env.items() if k.startswith('RUN_')}
    for s, d in sorted(runs.items()):
        miss = [f for f in ('final.pth', 'config.json', 'final_eval.json') if not os.path.isfile(os.path.join(d, f))]
        if miss:
            problems.append('run %s (%s): missing %s' % (s, d, ', '.join(miss)))
        else:
            print('[preflight] run %s: final.pth, config.json, final_eval.json present' % s)
    val, data = env['VAL'], env['DATA']
    if not os.path.isfile(val):
        problems.append('val.txt not found: %s' % val)
    else:
        lines = [l for l in open(val, encoding='utf-8').read().splitlines() if l.strip()]
        print('[preflight] val.txt: %d lines (expected %d)' % (len(lines), a.n_val))
        if len(lines) != a.n_val:
            problems.append('val.txt has %d lines, expected %d: %s' % (len(lines), a.n_val, val))
        absent = [p for l in lines for p in l.split(' ')[:2] if not os.path.isfile(os.path.join(data, p))]
        if absent:
            problems.append('%d image/label file(s) of val.txt missing under %s, e.g. %s' % (
                len(absent), data, absent[:5]))
        else:
            print('[preflight] every image/label of val.txt exists under %s' % data)
        digest = sha256(val)
        recorded = []
        for s, d in sorted(runs.items()):
            for src in (load(os.path.join(d, 'config.json')) or {}, load(os.path.join(d, 'split_info.json')) or {}):
                for k in ('val_list_sha256', 'val_sha256'):
                    if src.get(k):
                        recorded.append((s, k, src[k]))
        if not recorded:
            print('[preflight] val.txt sha256 %s: không kiểm được (no checksum recorded in config.json / '
                  'split_info.json)' % digest[:12])
        for s, k, v in recorded:
            if v != digest:
                problems.append('val.txt sha256 %s != %s recorded in run %s (%s)' % (digest[:12], v[:12], s, k))
            else:
                print('[preflight] val.txt sha256 matches run %s (%s)' % (s, k))
    if a.out_root:
        os.makedirs(a.out_root, exist_ok=True)
        free = shutil.disk_usage(a.out_root).free / 1024 ** 3
        print('[preflight] free disk at %s: %.1f GB (need > %.1f GB)' % (a.out_root, free, a.min_free_gb))
        if free <= a.min_free_gb:
            problems.append('only %.2f GB free at %s' % (free, a.out_root))
    for p in problems:
        print('[preflight] FAIL: ' + p)
    if problems:
        sys.exit(2)
    print('[preflight] PASS')


# ------------------------------------------------------------------------------------------------ summary
def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'nan'
    return '%.*f' % (nd, x)


def ms(o, key, nd=4):
    v = o.get(key) if o else None
    if not v:
        return MISSING, None, None
    return '%s ± %s' % (fmt(v['mean'], nd), fmt(v['std'], nd)), v['mean'], v['std']


def cmd_summary(a):
    splits = a.splits.split()
    md, rows = ['# Chẩn đoán origin (0a/0b) — tổng hợp', '',
                'Chỉ số liệu, không diễn giải. Nguồn: `<split>/eval_ckpt/eval_ckpt.json`, '
                '`<split>/diagnose_b<n>/diagnose_b<n>.json`.', ''], []

    def add(split, branch, table, metric, mean, std=None):
        rows.append({'split': split, 'branch': branch, 'table': table, 'metric': metric,
                     'mean': '' if mean is None else mean, 'std': '' if std is None else std})

    ev = {s: load(os.path.join(a.out_root, s, 'eval_ckpt', 'eval_ckpt.json')) for s in splits}

    # ---- table A
    md += ['## Bảng A — eval_ckpt (mỗi split)', '', '| chỉ số | ' + ' | '.join(s.replace('_', '/') for s in splits) + ' |',
           '|---|' + '---|' * len(splits)]
    names = [('miou_b1', 2), ('miou_b2', 2), ('miou_ave', 2), ('check', 0), ('agree_right_ratio', 4),
             ('agree_wrong_ratio', 4), ('conflict_ratio', 4), ('err_b1', 4), ('err_b1 - conflict_ratio (cận dưới cũ)', 4),
             ('agree_wrong + agree_right + conflict', 6)]
    for name, nd in names:
        cells = []
        for s in splits:
            e = ev[s]
            if not e:
                cells.append(MISSING)
                continue
            if name == 'check':
                c = e.get('check_vs_final_eval')
                if not c:
                    cells.append('không có final_eval.json')
                    continue
                mx = max(c['abs_diff'].values())
                cells.append('%s (max \\|diff\\| %.4f, tol %.2f)' % ('OK' if c['ok'] else 'MISMATCH', mx, c['tol']))
                add(s, '', 'A', 'check_vs_final_eval', 'OK' if c['ok'] else 'MISMATCH')
                add(s, '', 'A', 'max_abs_diff', mx)
                continue
            if name.startswith('err_b1 -'):
                v = e['err_b1'] - e['conflict_ratio']
                key = 'err_b1_minus_conflict'
            elif name.startswith('agree_wrong +'):
                v = e['agree_wrong_ratio'] + e['agree_right_ratio'] + e['conflict_ratio']
                key = 'agree_sum'
                cells.append('%s (%s)' % (fmt(v, nd), 'OK' if abs(v - 1) <= 1e-6 else 'LỆCH'))
                add(s, '', 'A', key, v)
                continue
            else:
                v, key = e.get(name), name
            cells.append(fmt(v, nd))
            add(s, '', 'A', key, v)
        md.append('| %s | %s |' % (name, ' | '.join(cells)))
    md.append('')

    # ---- table B
    classes = []
    for s in splits:
        if ev[s] and ev[s].get('agree_wrong_class'):
            classes = list(ev[s]['agree_wrong_class'])
            break
    md += ['## Bảng B — agree_wrong theo lớp: #(GT = c, y1 = y2 ≠ c) / #(GT = c)', '',
           '| lớp | ' + ' | '.join(s.replace('_', '/') for s in splits) + ' |', '|---|' + '---|' * len(splits)]
    if not classes:
        md.append('| (mọi lớp) | %s |' % ' | '.join(MISSING for _ in splits))
    for c in classes:
        cells = []
        for s in splits:
            v = ev[s]['agree_wrong_class'].get(c) if ev[s] else None
            cells.append(fmt(v) if ev[s] else MISSING)
            if ev[s]:
                add(s, '', 'B', 'agree_wrong_%s' % c, v)
        md.append('| %s | %s |' % (c, ' | '.join(cells)))
    md.append('')

    # ---- table C
    md += ['## Bảng C — diagnose_entanglement (mean ± std trên các seed EM)', '']
    cols = [('share', 'tỉ lệ pixel'), ('mean_purity', 'độ thuần TB'), ('in_mixed_cluster', 'ở cụm lẫn'),
            ('near_boundary', 'ở dải biên'), ('in_pure_cluster_of_predicted_class', 'ở cụm thuần của lớp bị đoán')]
    for s in splits:
        for b in a.branches.split():
            d = load(os.path.join(a.out_root, s, 'diagnose_b%s' % b, 'diagnose_b%s.json' % b))
            md.append('### split %s · nhánh %s' % (s.replace('_', '/'), b))
            md.append('')
            if not d:
                md += [MISSING, '']
                continue
            o = d['overall']
            st = d.get('settings', {})
            md.append('K %s, mixed < %s, biên %s px, %d ảnh val, seed EM %s, %d pixel lấy mẫu' % (
                st.get('k'), st.get('mixed_threshold'), st.get('boundary_radius_px_full_res'),
                st.get('n_val_images', 0), st.get('em_seeds'), d.get('n_sampled', 0)))
            md.append('')
            md += ['| nhóm | ' + ' | '.join(c[1] for c in cols) + ' |', '|---|' + '---|' * len(cols)]
            for g in GROUPS:
                cells = []
                for k, _ in cols:
                    if k == 'in_pure_cluster_of_predicted_class' and g != 'agree_wrong':
                        cells.append('–')
                        continue
                    txt, m, sd = ms(o, '%s.%s' % (g, k))
                    cells.append(txt)
                    add(s, b, 'C', '%s.%s' % (g, k), m, sd)
                md.append('| %s | %s |' % (g, ' | '.join(cells)))
            md.append('')
            for k, label, nd in (('n_clusters_used', 'số cụm được dùng', 1),
                                 ('mean_purity_all_pixels', 'độ thuần toàn cục (mọi pixel)', 4)):
                txt, m, sd = ms(o, k, nd)
                md.append('- %s: %s' % (label, txt))
                add(s, b, 'C', k, m, sd)
            md.append('')

    with open(os.path.join(a.out_root, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(a.out_root, 'summary.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'branch', 'table', 'metric', 'mean', 'std'])
        w.writeheader()
        w.writerows(rows)
    print('[diag_origin] -> %s, %s' % (os.path.join(a.out_root, 'summary.md'), os.path.join(a.out_root, 'summary.csv')))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('find')
    f.add_argument('--search-root', default='/kaggle/input')
    f.add_argument('--run-1-4', default='')
    f.add_argument('--run-1-8', default='')
    f.add_argument('--val-list', default='')
    f.add_argument('--data-root', default='')
    f.add_argument('--splits', default='1_4 1_8')
    f.add_argument('--env-out', required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--env', required=True)
    p.add_argument('--n-val', type=int, default=384)
    p.add_argument('--out-root', default='')
    p.add_argument('--min-free-gb', type=float, default=1.0)
    s = sub.add_parser('summary')
    s.add_argument('--out-root', required=True)
    s.add_argument('--splits', default='1_4 1_8')
    s.add_argument('--branches', default='1 2')
    a = ap.parse_args()
    {'find': cmd_find, 'preflight': cmd_preflight, 'summary': cmd_summary}[a.cmd](a)


if __name__ == '__main__':
    main()
