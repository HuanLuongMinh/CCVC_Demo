"""Gather every finished run (final_eval.json) into results_summary.csv (spec 6.2).

    python tools/collect_results.py [--out_root /kaggle/working/exp/oem] [--out results_summary.csv]
Main number = mIoU of branch 1 at the last epoch; best-on-val is for reference only.
"""
import argparse
import csv
import json
import os

CLASS_NAMES = ['Bareland', 'Rangeland', 'Developed space', 'Road', 'Tree', 'Water', 'Agriculture land', 'Building']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root', default='/kaggle/working/exp/oem')
    parser.add_argument('--out', default='')
    args = parser.parse_args()
    out = args.out or os.path.join(args.out_root, 'results_summary.csv')

    rows = []
    for dirpath, dirnames, filenames in os.walk(args.out_root):
        dirnames[:] = sorted(d for d in dirnames if d != 'trial' and '_old_' not in d)
        if 'final_eval.json' not in filenames:
            continue
        with open(os.path.join(dirpath, 'final_eval.json'), encoding='utf-8') as f:
            r = json.load(f)
        row = {'split': r['split'], 'full_unlabeled': r.get('full_unlabeled') or '', 'split_seed': r['split_seed'],
               'method': r['method'], 'lambda_preset': r.get('lambda_preset'), 'seed': r['seed'],
               'n_labeled': r['n_labeled'], 'n_unlabeled': r['n_unlabeled'], 'n_train': r.get('n_train'),
               'n_val': r['n_val'], 'epochs': r['epoch'], 'eval_mode': r.get('eval_mode'),
               'miou_b1': round(r['miou_b1'], 2), 'miou_b2': round(r['miou_b2'], 2),
               'miou_ave': round(r['miou_ave'], 2),
               'best_b1_reference': round(r['best_on_val_reference_only']['miou_b1'], 2)
               if r.get('best_on_val_reference_only') else '',
               'run_dir': os.path.relpath(dirpath, args.out_root).replace(os.sep, '/')}
        for c in CLASS_NAMES:
            row['iou_b1_%s' % c] = round(r['iou_b1'].get(c, float('nan')), 2)
        rows.append(row)

    order = {'1_16': 0, '1_8': 1, '1_4': 2, '1_2': 3, 'full': 5}
    rows.sort(key=lambda r: (order.get(r['split'], 4), r['split'], r['full_unlabeled'], r['method'], r['seed']))
    if not rows:
        print('no finished run (final_eval.json) under %s' % args.out_root)
        return
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('%d finished run(s) -> %s' % (len(rows), out))
    print('%-14s %-9s %-14s %4s %8s %8s %8s %8s' % ('split', 'method', 'lambda', 'seed', 'labeled', 'mIoU_b1',
                                                 'mIoU_b2', 'mIoU_ave'))
    for r in rows:
        split = r['split'] + ('/' + r['full_unlabeled'] if r['full_unlabeled'] else '')
        print('%-14s %-9s %-14s %4s %8s %8.2f %8.2f %8.2f' % (split, r['method'], r['lambda_preset'], r['seed'],
                                                            r['n_labeled'], r['miou_b1'], r['miou_b2'],
                                                            r['miou_ave']))
    sizes = {(r['n_train'], r['n_val']) for r in rows}
    if len(sizes) > 1:
        print('WARNING: runs use different train/val sizes %s - the Kaggle dataset changed between runs; '
              'these rows are not directly comparable.' % sorted(sizes, key=str))


if __name__ == '__main__':
    main()
