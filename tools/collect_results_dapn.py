"""Gather origin and DAPN runs into one results_summary.csv (spec 8). tools/collect_results.py is not modified.

    python tools/collect_results_dapn.py [--roots /kaggle/working/exp/oem /kaggle/working/exp/oem_dapn <more>]
                                         [--out /kaggle/working/exp/oem_dapn/results_summary.csv]

- Every finished run (final_eval.json) under each root; trial/ and *_old_* skipped. A root may also be a previous
  output mounted under /kaggle/input (e.g. <input>/exp/oem of the origin notebook).
- code_version: from final_eval.json ('dapn') or 'origin'. Same run found in several roots -> kept once.
- agree_wrong_ratio, agree_right_ratio, conflict_ratio, err_b1: from final_eval.json (DAPN runs) or from an
  eval_ckpt.json of tools/eval_ckpt.py for that checkpoint (next to it, or under <root>/eval_ckpt/ of any root);
  empty for an origin run that has not been through eval_ckpt.py.
- diag_b<n>_*: purity of the agree-right / agree-wrong pixels from diagnose_b<n>.json (tools/diagnose_entanglement.py)
  when present (next to the checkpoint, or under <root>/diagnose/).
- Prints every run and mean +- std (ddof 1) over seeds per (split, code_version, method).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

CLASS_NAMES = ['Bareland', 'Rangeland', 'Developed space', 'Road', 'Tree', 'Water', 'Agriculture land', 'Building']
AGREE = ('agree_wrong_ratio', 'agree_right_ratio', 'conflict_ratio', 'err_b1')


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def side_files(roots, pattern_in_run, sub, name_glob):
    """{abs run dir: json} for files next to a checkpoint or under <root>/<sub>/ (matched by their 'ckpt')."""
    out = {}
    for r in roots:
        for p in glob.glob(os.path.join(r, '**', pattern_in_run), recursive=True) + \
                glob.glob(os.path.join(r, sub, '**', name_glob), recursive=True):
            try:
                j = load(p)
            except (OSError, ValueError):
                continue
            ck = j.get('ckpt')
            run = os.path.normcase(os.path.abspath(os.path.dirname(ck) if ck else os.path.dirname(p)))
            out.setdefault(run, []).append((p, j))
    return out


def run_tail(path):
    """<split dir>/<method>/seed<n>: identifies a run whatever the root it was copied to."""
    parts = os.path.normpath(path).split(os.sep)
    return '/'.join(parts[-3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', default=['/kaggle/working/exp/oem', '/kaggle/working/exp/oem_dapn'])
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    roots = [r for r in a.roots if os.path.isdir(r)]
    writable = [r for r in roots if os.access(r, os.W_OK)]
    pick = [r for r in writable if os.path.basename(os.path.normpath(r)) == 'oem_dapn'] + writable + ['.']
    out = a.out or os.path.join(pick[0], 'results_summary.csv')

    evals = side_files(roots, 'eval_ckpt.json', 'eval_ckpt', 'eval_ckpt_*.json')
    diags = {}
    for b in (1, 2):
        diags[b] = side_files(roots, 'diagnose_b%d.json' % b, 'diagnose', 'diagnose_b%d.json' % b)
    ev_tail = {run_tail(k): v for k, v in evals.items()}
    dg_tail = {b: {run_tail(k): v for k, v in diags[b].items()} for b in diags}

    rows, seen = [], set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != 'trial' and '_old_' not in d
                                 and d not in ('eval_ckpt', 'diagnose'))
            if 'final_eval.json' not in filenames:
                continue
            r = load(os.path.join(dirpath, 'final_eval.json'))
            cv = r.get('code_version', 'origin')
            key = (cv, run_tail(dirpath))
            if key in seen:
                continue
            seen.add(key)
            row = {'code_version': cv, 'split': r['split'], 'split_seed': r['split_seed'], 'method': r['method'],
                   'lambda_preset': r.get('lambda_preset'), 'seed': r['seed'],
                   'w_dapg': (r.get('dapg') or {}).get('w_dapg', ''),
                   'dapg_on': (r.get('dapg') or {}).get('dapg_on', ''),
                   'dapg_branches': (r.get('dapg') or {}).get('dapg_branches', ''),
                   'dapg_pos': (r.get('dapg') or {}).get('dapg_pos', ''),
                   'n_labeled': r['n_labeled'], 'n_unlabeled': r['n_unlabeled'], 'n_val': r['n_val'],
                   'epochs': r['epoch'], 'eval_mode': r.get('eval_mode'),
                   'miou_b1': round(r['miou_b1'], 2), 'miou_b2': round(r['miou_b2'], 2),
                   'miou_ave': round(r['miou_ave'], 2)}
            src = ''
            if all(k in r for k in AGREE):
                row.update({k: r[k] for k in AGREE})
                src = 'final_eval'
            else:
                e = evals.get(os.path.normcase(os.path.abspath(dirpath))) or ev_tail.get(run_tail(dirpath))
                if e:
                    row.update({k: e[0][1][k] for k in AGREE})
                    src = 'eval_ckpt'
                else:
                    row.update({k: '' for k in AGREE})
            row['agree_source'] = src
            for b in (1, 2):
                d = diags[b].get(os.path.normcase(os.path.abspath(dirpath))) or dg_tail[b].get(run_tail(dirpath))
                o = d[0][1]['overall'] if d else None
                row['diag_b%d_purity_agree_right' % b] = o['agree_right.mean_purity']['mean'] if o else ''
                row['diag_b%d_purity_agree_wrong' % b] = o['agree_wrong.mean_purity']['mean'] if o else ''
                row['diag_b%d_agree_wrong_in_mixed' % b] = o['agree_wrong.in_mixed_cluster']['mean'] if o else ''
            for c in CLASS_NAMES:
                row['iou_b1_%s' % c] = round(r['iou_b1'].get(c, float('nan')), 2)
            row['run_dir'] = dirpath.replace(os.sep, '/')
            rows.append(row)

    if not rows:
        print('no finished run (final_eval.json) under %s' % roots)
        return
    order = {'1_16': 0, '1_8': 1, '1_4': 2, '1_2': 3, 'full': 5}
    rows.sort(key=lambda r: (order.get(r['split'], 4), r['split'], r['code_version'] != 'origin', r['method'],
                             r['seed']))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('%d finished run(s) -> %s' % (len(rows), out))
    fmt = '%-6s %-7s %-14s %4s %8s %8s %8s %12s'
    print(fmt % ('split', 'code', 'method', 'seed', 'mIoU_b1', 'mIoU_b2', 'mIoU_ave', 'agree_wrong'))
    for r in rows:
        aw = '%.4f' % r['agree_wrong_ratio'] if r['agree_wrong_ratio'] != '' else '-'
        print(fmt % (r['split'], r['code_version'], r['method'], r['seed'], '%.2f' % r['miou_b1'],
                     '%.2f' % r['miou_b2'], '%.2f' % r['miou_ave'], aw))
    groups = {}
    for r in rows:
        groups.setdefault((r['split'], r['code_version'], r['method']), []).append(r)
    print('mean +- std over seeds (ddof 1):')
    for (s, cv, m), rs in groups.items():
        b1 = np.array([x['miou_b1'] for x in rs], dtype=float)
        aw = np.array([x['agree_wrong_ratio'] for x in rs if x['agree_wrong_ratio'] != ''], dtype=float)
        sd = lambda v: v.std(ddof=1) if v.size > 1 else float('nan')  # noqa: E731
        print('  %-6s %-7s %-14s n=%d  mIoU_b1 %.2f +- %.2f%s' % (
            s, cv, m, len(rs), b1.mean(), sd(b1),
            '  agree_wrong %.4f +- %.4f (n=%d)' % (aw.mean(), sd(aw), aw.size) if aw.size else ''))
    if len({(r['n_labeled'], r['n_val']) for r in rows if r['split'] == rows[0]['split']}) > 1:
        print('WARNING: runs of the same split use different labeled/val sizes: not directly comparable.')


if __name__ == '__main__':
    main()
