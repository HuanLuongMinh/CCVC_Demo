"""Dry run of a DAPN preset vs the baseline (spec 7). Called by run_dapn.sh after a --dryrun.

    python tools/trial_compare_dapn.py find --split_dir 1_4_s0 --seed 0 --out_root <exp/oem_dapn>
                                            [--search_root /kaggle/input] [--baseline_trial <file>]
        prints the baseline trial_report.json to use, or nothing. Order: --baseline_trial; origin dry run of this
        notebook (<out_root>/../oem/<split>/ccvc_aug/seed*/trial); run_dapn.sh --method ccvc_origin dry run;
        origin dry run in a previous output under --search_root; ccvc_dapn_off dry run.
    python tools/trial_compare_dapn.py compare --report <DAPN trial_report.json> --baseline <trial_report.json>
        adds 'dapn_vs_baseline' to the DAPN trial_report.json: s/iter, peak memory per GPU, projected hours and
        Kaggle sessions for the full run, with the increase in % over the baseline; prints a short table.
"""
import argparse
import glob
import json
import os
import sys


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def find(a):
    if a.baseline_trial:
        return a.baseline_trial if os.path.exists(a.baseline_trial) else ''
    work_exp = os.path.dirname(os.path.abspath(a.out_root))
    cands = sorted(glob.glob(os.path.join(work_exp, 'oem', a.split_dir, 'ccvc_aug', 'seed%s' % a.seed, 'trial',
                                          'trial_report.json')))
    cands += [os.path.join(a.out_root, a.split_dir, 'ccvc_origin', 'seed%s' % a.seed, 'trial', 'trial_report.json')]
    if a.search_root and os.path.isdir(a.search_root):
        for depth in range(0, 6):
            pat = os.path.join(a.search_root, *(['*'] * depth), 'exp', 'oem', a.split_dir, 'ccvc_aug', 'seed*',
                               'trial', 'trial_report.json')
            cands += sorted(glob.glob(pat))
    cands += [os.path.join(a.out_root, a.split_dir, 'ccvc_dapn_off', 'seed%s' % a.seed, 'trial', 'trial_report.json')]
    for c in cands:
        if os.path.exists(c):
            return c
    return ''


def peak(rep):
    m = [g.get('train_peak_allocated_gb') for g in rep.get('memory_per_gpu') or [] if g]
    m = [x for x in m if x is not None]
    return max(m) if m else None


def pct(new, old):
    if new is None or old in (None, 0):
        return None
    return 100.0 * (new - old) / old


def compare(a):
    rep, base = load(a.report), load(a.baseline)
    s_new, s_old = rep.get('s_per_iter_median'), base.get('s_per_iter_median')
    m_new, m_old = peak(rep), peak(base)
    h_new, h_old = (rep.get('projection') or {}).get('total_hours'), (base.get('projection') or {}).get('total_hours')
    same = {k: rep.get(k) == base.get(k) for k in ('world_size', 'batch_per_gpu_per_loader', 'amp', 'split')}
    out = {
        'baseline_report': os.path.abspath(a.baseline), 'baseline_method': base.get('method'),
        'comparable': all(same.values()), 'same_settings': same,
        's_per_iter_median': {'dapn': s_new, 'baseline': s_old, 'increase_pct': pct(s_new, s_old)},
        'train_peak_allocated_gb_max_gpu': {'dapn': m_new, 'baseline': m_old, 'increase_pct': pct(m_new, m_old)},
        'projected_hours_full_run': {'dapn': h_new, 'baseline': h_old, 'increase_pct': pct(h_new, h_old)},
        'kaggle_sessions': {'dapn': (rep.get('projection') or {}).get('kaggle_sessions'),
                            'baseline': (base.get('projection') or {}).get('kaggle_sessions')},
        'epochs': (rep.get('projection') or {}).get('epochs'),
    }
    rep['dapn_vs_baseline'] = out
    with open(a.report, 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=2)
    f2 = lambda v, u='': '-' if v is None else '%.3f%s' % (v, u)  # noqa: E731
    p2 = lambda v: '-' if v is None else '%+.1f%%' % v  # noqa: E731
    print('-' * 78)
    print('DAPN vs baseline dry run (%s, method %s)%s' % (a.baseline, base.get('method'),
                                                          '' if out['comparable'] else
                                                          '  WARNING: different settings %s' % same))
    print('  %-26s %12s %12s %10s' % ('', 'DAPN', 'baseline', 'increase'))
    for name, k, u in (('s/iter (median)', 's_per_iter_median', ' s'),
                       ('train peak / GPU', 'train_peak_allocated_gb_max_gpu', ' GB'),
                       ('full run (%s epochs)' % out['epochs'], 'projected_hours_full_run', ' h')):
        print('  %-26s %12s %12s %10s' % (name, f2(out[k]['dapn'], u), f2(out[k]['baseline'], u),
                                          p2(out[k]['increase_pct'])))
    print('  %-26s %12s %12s' % ('Kaggle sessions', out['kaggle_sessions']['dapn'], out['kaggle_sessions']['baseline']))
    print('-' * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['find', 'compare'])
    ap.add_argument('--split_dir', default='')
    ap.add_argument('--seed', default='0')
    ap.add_argument('--out_root', default='/kaggle/working/exp/oem_dapn')
    ap.add_argument('--search_root', default='')
    ap.add_argument('--baseline_trial', default='')
    ap.add_argument('--report', default='')
    ap.add_argument('--baseline', default='')
    a = ap.parse_args()
    if a.cmd == 'find':
        print(find(a))
    else:
        compare(a)


if __name__ == '__main__':
    sys.exit(main())
