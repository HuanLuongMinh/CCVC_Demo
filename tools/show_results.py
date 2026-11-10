"""Print the results of one run (called at the end of run_baseline.sh) + the summary of every finished split.

    python tools/show_results.py --run_dir <run dir or its trial/> [--out_root /kaggle/working/exp/oem]
"""
import argparse
import glob
import json
import os
import subprocess
import sys


def load(path):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--out_root', default='/kaggle/working/exp/oem')
    args = parser.parse_args()
    d = args.run_dir
    print('=' * 78)
    print('RESULTS  %s' % d)

    final = load(os.path.join(d, 'final_eval.json'))
    trial = load(os.path.join(d, 'trial_report.json'))
    prog = load(os.path.join(d, 'progress.json'))
    if final:
        print('  FINISHED  %s epochs | main number = mIoU branch 1 at the last epoch' % final['epoch'])
        print('  mIoU  branch 1 = %.2f | branch 2 = %.2f | average = %.2f' % (final['miou_b1'], final['miou_b2'],
                                                                            final['miou_ave']))
        print('  IoU branch 1 per class: ' + ', '.join('%s %.1f' % kv for kv in final['iou_b1'].items()))
        if final.get('best_on_val_reference_only'):
            b = final['best_on_val_reference_only']
            print('  (reference only) best branch 1 on val: %.2f at epoch %d' % (b['miou_b1'], b['epoch']))
    elif trial:
        p = trial['projection']
        print('  TRIAL  mIoU b1 %.2f (sanity check only) | eval mode %s' % (trial['miou_b1'], trial['eval_mode']))
        if trial.get('s_per_iter_median'):
            print('  speed %.3f s/iter -> full run %d iters + %d evals = %.1f h = %s Kaggle session(s)' % (
                trial['s_per_iter_median'], p['total_iters'], p['n_evals'], p['total_hours'], p['kaggle_sessions']))
        for k, m in enumerate(trial.get('memory_per_gpu', [])):
            if m:
                print('  GPU%d peak: train %.2f GB%s' % (k, m['train_peak_allocated_gb'],
                                                      ' | eval %.2f GB' % m['eval_peak_allocated_gb']
                                                      if 'eval_peak_allocated_gb' in m else ''))
    elif prog:
        print('  NOT FINISHED: %d/%d epochs done -> run the same command in the next session'
              % (prog['epochs_done'], prog['epochs']))
    else:
        print('  no result yet')

    plots = sorted(glob.glob(os.path.join(d, 'plots', '*.png')))
    vis = sorted(glob.glob(os.path.join(d, 'vis', '*.png')))
    if plots:
        print('  plots: %s/plots/{%s}' % (d, ','.join(os.path.basename(p) for p in plots)))
    if vis:
        print('  latest example panel: %s' % vis[-1])
    print('=' * 78, flush=True)
    if os.path.isdir(args.out_root):
        subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'collect_results.py'),
                         '--out_root', args.out_root])


if __name__ == '__main__':
    main()
