"""Stage-2 measurements right after a finished run (called by run_dapn.sh; also usable alone).

    python tools/post_run_dapn.py --run_dir <out_root>/<split>/<method>/seed<n> --out_root <exp/oem_dapn>
        --val_list <partitions/oem_dapn/val.txt> --data_root <OEM root> [--search_root /kaggle/input]
        [--baseline_run <origin run dir>] [--no_baseline] [--session_start <unix time>] [--hard_limit_hours 12]
        [--margin_min 20] [--device cuda|cpu] [--num_workers 2] [--k 64 --iters 20 --sample 200000 --em_seeds 3]

Steps (each one skipped when its output already exists, so the same command can finish the work later):
 1. this run: eval_ckpt.py when final_eval.json has no agreement metrics (ccvc_origin runs);
    diagnose_entanglement.py on branch 1 and branch 2.
 2. the baseline of the same split and seed (origin CCVC final.pth): eval_ckpt.py (agree_wrong_ratio of the
    baseline; its three mIoU must match its final_eval.json) and diagnose_entanglement.py on branches 1 and 2.
    Search order: --baseline_run; <out_root>/../oem/<split>/ccvc_aug/seed<n> (run_baseline.sh in this
    notebook); <out_root>/<split>/ccvc_origin/seed<n>; <search_root>/**/exp/oem/<split>/ccvc_aug/seed<n>
    (previous origin output added as input). Skipped for a ccvc_origin run (it is the baseline).
 3. collect_results_dapn.py over <out_root>, <out_root>/../oem and the baseline's exp root ->
    <out_root>/results_summary.csv, then a short DAPN-vs-baseline table for this split and seed.
Outputs land next to each checkpoint when that directory is writable, otherwise under <out_root>/eval_ckpt/ and
<out_root>/diagnose/<split>_<method>_seed<n>/ (Kaggle input is read-only).
Time: each step is estimated from the evaluation time of the run (eval_log.csv); a step that would end past
hard_limit_hours - margin after session_start is not started (message: run the same command next session).
Exit code 0 unless something unexpected fails (1); a skipped step is not a failure.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def log(msg):
    print('[analysis] ' + msg, flush=True)


def load(p, default=None):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class Budget(object):
    def __init__(self, a, run_dir):
        self.start = a.session_start or time.time()
        self.limit = a.hard_limit_hours * 3600 - a.margin_min * 60 if a.hard_limit_hours > 0 else None
        ev = []
        for r in (csv.DictReader(open(os.path.join(run_dir, 'eval_log.csv'), encoding='utf-8'))
                  if os.path.exists(os.path.join(run_dir, 'eval_log.csv')) else []):
            try:
                ev.append(float(r['eval_time_s']))
            except (KeyError, ValueError):
                pass
        self.eval_s = sorted(ev)[len(ev) // 2] if ev else 600.0
        self.stopped = False

    def allow(self, what, factor):
        need = factor * self.eval_s
        if self.stopped:
            return False
        if self.limit is not None:
            elapsed = time.time() - self.start
            if elapsed + need > self.limit:
                left = max(self.limit - elapsed, 0) / 60
                log('NOT STARTED: %s needs ~%.0f min, %.0f min left before the session limit -> run the SAME '
                    'command in the next session to finish the analysis (the training is done)' % (what, need / 60, left))
                self.stopped = True
                return False
        log('%s (estimated %.0f min) ...' % (what, need / 60))
        return True


def run_tool(args):
    t0 = time.time()
    r = subprocess.run([PY] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith(('[eval_ckpt]', '[diagnose]'))
             and '->' not in l or 'MISMATCH' in l or 'ERROR' in l]
    for l in lines:
        print('    ' + l, flush=True)
    if r.returncode != 0:
        tail = '\n'.join(r.stdout.splitlines()[-15:])
        log('FAILED (exit %d, %.0f s):\n%s' % (r.returncode, time.time() - t0, tail))
    return r.returncode


def out_dirs(run_dir, out_root):
    """(eval_ckpt json path, diagnose dir) for a run: next to it when writable, else under out_root."""
    if os.access(run_dir, os.W_OK):
        return os.path.join(run_dir, 'eval_ckpt.json'), run_dir
    c = load(os.path.join(run_dir, 'config.json'), {})
    tag = '%s_%s_seed%s' % (c.get('split'), c.get('method'), c.get('seed'))
    return (os.path.join(out_root, 'eval_ckpt', 'eval_ckpt_%s.json' % tag),
            os.path.join(out_root, 'diagnose', tag))


def measure(run_dir, label, a, budget, need_eval):
    """eval_ckpt (if need_eval) + diagnose b1/b2 of one run. Returns the number of failures."""
    ckpt = os.path.join(run_dir, 'final.pth')
    ev_path, dg_dir = out_dirs(run_dir, a.out_root)
    common = ['--ckpt', ckpt, '--val-list', a.val_list, '--device', a.device, '--num-workers', str(a.num_workers)]
    if a.data_root:
        common += ['--data-root', a.data_root]
    fails = 0
    if need_eval:
        if os.path.exists(ev_path):
            log('%s: eval_ckpt already done (%s)' % (label, ev_path))
        elif budget.allow('%s: eval_ckpt.py (agreement metrics)' % label, 1.3):
            os.makedirs(os.path.dirname(ev_path), exist_ok=True)
            rc = run_tool([os.path.join(ROOT, 'tools', 'eval_ckpt.py')] + common + ['--out', ev_path])
            if rc != 0:
                fails += 1
                log('%s: eval_ckpt.py reported a problem (MISMATCH with final_eval.json = stop and ask, spec 11)' % label)
    for b in (1, 2):
        dj = os.path.join(dg_dir, 'diagnose_b%d.json' % b)
        if os.path.exists(dj):
            log('%s: diagnose branch %d already done (%s)' % (label, b, dj))
            continue
        if not budget.allow('%s: diagnose_entanglement.py branch %d' % (label, b), 2.5):
            continue
        rc = run_tool([os.path.join(ROOT, 'tools', 'diagnose_entanglement.py')] + common +
                      ['--branch', str(b), '--k', str(a.k), '--iters', str(a.iters), '--sample', str(a.sample),
                       '--n-seeds', str(a.em_seeds), '--out-dir', dg_dir])
        fails += rc != 0
    return fails


def find_baseline(a, split_dir, seed):
    if a.baseline_run:
        return a.baseline_run if os.path.exists(os.path.join(a.baseline_run, 'final.pth')) else None
    exp = os.path.dirname(os.path.abspath(a.out_root))
    cands = [os.path.join(exp, 'oem', split_dir, 'ccvc_aug', 'seed%s' % seed),
             os.path.join(a.out_root, split_dir, 'ccvc_origin', 'seed%s' % seed)]
    if a.search_root and os.path.isdir(a.search_root):
        for depth in range(0, 6):
            cands += sorted(glob.glob(os.path.join(a.search_root, *(['*'] * depth), 'exp', 'oem', split_dir,
                                                   'ccvc_aug', 'seed%s' % seed)))
    for c in cands:
        if os.path.exists(os.path.join(c, 'final.pth')) and os.path.exists(os.path.join(c, 'final_eval.json')):
            return c
    return None


def summary(csv_path, split, seed, run_dir, base_dir):
    if not os.path.exists(csv_path):
        return
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
    norm = lambda p: os.path.normcase(os.path.abspath(p)).replace('\\', '/') if p else None  # noqa: E731
    want = {norm(run_dir): 'this run', norm(base_dir): 'baseline'}
    pick = [(want[norm(r['run_dir'])], r) for r in rows if norm(r['run_dir']) in want]
    if not pick:
        return
    pick.sort(key=lambda x: x[0] != 'baseline')
    keys = [('mIoU b1', 'miou_b1'), ('mIoU b2', 'miou_b2'), ('mIoU ave', 'miou_ave'),
            ('agree_wrong_ratio', 'agree_wrong_ratio'), ('agree_right_ratio', 'agree_right_ratio'),
            ('conflict_ratio', 'conflict_ratio'), ('purity agree-right b1', 'diag_b1_purity_agree_right'),
            ('purity agree-wrong b1', 'diag_b1_purity_agree_wrong'), ('agree-wrong in mixed b1', 'diag_b1_agree_wrong_in_mixed'),
            ('purity agree-right b2', 'diag_b2_purity_agree_right'), ('purity agree-wrong b2', 'diag_b2_purity_agree_wrong')]
    print('=' * 78)
    print('ANALYSIS SUMMARY  split %s  seed %s   (%s)' % (split, seed, csv_path))
    print('  %-26s' % '' + ''.join('%22s' % ('%s: %s' % (lab, r['method']))[:22] for lab, r in pick))
    for name, k in keys:
        vals = []
        for _, r in pick:
            v = r.get(k, '')
            try:
                vals.append('%22.4f' % float(v) if 'ratio' in k or 'diag' in k else '%22.2f' % float(v))
            except ValueError:
                vals.append('%22s' % '-')
        print('  %-26s' % name + ''.join(vals))
    print('  (one seed = first signal only; compare over seeds with tools/collect_results_dapn.py)')
    print('=' * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--val_list', required=True)
    ap.add_argument('--data_root', default='')
    ap.add_argument('--search_root', default='')
    ap.add_argument('--baseline_run', default='')
    ap.add_argument('--no_baseline', action='store_true')
    ap.add_argument('--session_start', type=float, default=0.0)
    ap.add_argument('--hard_limit_hours', type=float, default=12.0)
    ap.add_argument('--margin_min', type=float, default=20.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--sample', type=int, default=200000)
    ap.add_argument('--em_seeds', type=int, default=3)
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    final = load(os.path.join(run_dir, 'final_eval.json'))
    conf = load(os.path.join(run_dir, 'config.json'), {})
    if final is None or not os.path.exists(os.path.join(run_dir, 'final.pth')):
        log('no final.pth / final_eval.json in %s: nothing to measure' % run_dir)
        return 0
    split_dir = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
    seed = conf.get('seed', final.get('seed'))
    budget = Budget(a, run_dir)
    log('=' * 70)
    log('stage-2 measurements for %s (eval time of this run ~%.0f s)' % (run_dir, budget.eval_s))
    fails = measure(run_dir, 'this run (%s)' % conf.get('method'), a, budget,
                    need_eval='agree_wrong_ratio' not in final)

    base = None
    if conf.get('code_version', 'origin') == 'origin' or conf.get('method') == 'ccvc_origin':
        log('this run is CCVC origin: it is its own baseline')
    elif a.no_baseline:
        log('baseline measurements skipped (--no-analysis-baseline)')
    else:
        base = find_baseline(a, split_dir, seed)
        if base is None:
            log('no baseline final.pth for %s seed %s (run_baseline.sh in this notebook, --method ccvc_origin, or the '
                'origin output added as input; or --baseline-run <dir>): baseline measurements skipped' % (split_dir, seed))
        else:
            log('baseline: %s' % base)
            fails += measure(base, 'baseline (%s)' % load(os.path.join(base, 'config.json'), {}).get('method'),
                             a, budget, need_eval=True)

    roots = [a.out_root, os.path.join(os.path.dirname(os.path.abspath(a.out_root)), 'oem')]
    if base:
        roots.append(os.path.dirname(os.path.dirname(os.path.dirname(base))))
    roots = [r for k, r in enumerate(roots) if os.path.isdir(r) and
             os.path.normcase(os.path.abspath(r)) not in [os.path.normcase(os.path.abspath(x)) for x in roots[:k]]]
    out_csv = os.path.join(a.out_root, 'results_summary.csv')
    rc = subprocess.run([PY, os.path.join(ROOT, 'tools', 'collect_results_dapn.py'), '--roots'] + roots +
                        ['--out', out_csv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if rc.returncode != 0:
        log('collect_results_dapn.py failed:\n' + rc.stdout[-2000:])
        fails += 1
    else:
        log('results_summary.csv updated: %s' % out_csv)
    summary(out_csv, split_dir, seed, run_dir, base)
    if budget.stopped:
        log('analysis INCOMPLETE (session time): run the same command again next session; finished steps are kept')
    else:
        log('analysis complete' + ('' if not fails else ' with %d problem(s), see above' % fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
