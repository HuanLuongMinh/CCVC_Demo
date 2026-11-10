"""Plots of one run, rebuilt from its CSV/JSON logs (no training needed).

    python tools/plot_run.py <run_dir>          # writes <run_dir>/plots/*.png

miou_curve.png, loss_curve.png, per_class_iou.png, conflict_ratio.png,
pseudo_label_class_dist.png, confidence_reliability.png
"""
import csv
import json
import os
import sys


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _line_color(rgb):
    # white/very light palette entries (Road) would vanish on a white background
    return (0.25, 0.25, 0.25) if sum(rgb) > 600 else tuple(c / 255.0 for c in rgb)


def plot_run(run_dir, quiet=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = {}
    if os.path.exists(os.path.join(run_dir, 'config.json')):
        with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
            cfg = json.load(f)
    ipe = cfg.get('iters_per_epoch') or 0
    title = '%s · %s · seed %s' % (cfg.get('split', '?'), cfg.get('method', '?'), cfg.get('seed', '?'))
    out_dir = os.path.join(run_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    train = _rows(os.path.join(run_dir, 'train_log.csv'))
    evals = _rows(os.path.join(run_dir, 'eval_log.csv'))
    pseudo = _rows(os.path.join(run_dir, 'pseudo_label_dist.csv'))
    conf = _rows(os.path.join(run_dir, 'confidence_reliability.csv'))
    written = []

    def epoch_axis(ax):
        if ipe:
            sec = ax.secondary_xaxis('top', functions=(lambda x: x / ipe, lambda e: e * ipe))
            sec.set_xlabel('epoch')

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name), dpi=90)
        plt.close(fig)
        written.append(name)

    # 1. mIoU
    if evals:
        x = [_f(r['iteration']) for r in evals]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, [_f(r['miou_b1']) for r in evals], 'o-', lw=2.8, color='C0', label='nhánh 1 (dự đoán chính)')
        ax.plot(x, [_f(r['miou_b2']) for r in evals], 's--', lw=1.3, color='C1', label='nhánh 2')
        ax.plot(x, [_f(r['miou_ave']) for r in evals], '^:', lw=1.3, color='C2', label='trung bình 2 nhánh')
        ax.set_xlabel('iteration')
        ax.set_ylabel('mIoU val (%)')
        ax.set_title('mIoU — ' + title)
        ax.grid(alpha=0.3)
        ax.legend()
        epoch_axis(ax)
        save(fig, 'miou_curve.png')

    # 2. losses, one subplot each
    if train:
        x = [_f(r['iteration']) for r in train]
        keys = [('L_sup', 'L_sup (×w_CE)'), ('L_con', 'L_con (×w_con)'), ('L_dis', 'L_dis (×w_dis)'),
                ('L_total', 'L_total')]
        fig, axes = plt.subplots(len(keys), 1, figsize=(8, 2.2 * len(keys)), sharex=True)
        for ax, (k, lab) in zip(axes, keys):
            ax.plot(x, [_f(r[k]) for r in train], lw=1.2)
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
        axes[0].set_title('Loss (trung bình mỗi %s iter) — %s' % (cfg.get('log_every', 50), title))
        axes[-1].set_xlabel('iteration')
        save(fig, 'loss_curve.png')

    # 3. per-class IoU of branch 1
    if evals:
        cls = [k[len('iou_b1_'):] for k in evals[0] if k.startswith('iou_b1_')]
        palette = cfg.get('palette') or []
        x = [_f(r['iteration']) for r in evals]
        fig, ax = plt.subplots(figsize=(9, 5))
        for k, c in enumerate(cls):
            color = _line_color(palette[k]) if k < len(palette) else None
            ax.plot(x, [_f(r['iou_b1_' + c]) for r in evals], 'o-', lw=1.5, color=color, label=c.replace('_', ' '))
        ax.set_xlabel('iteration')
        ax.set_ylabel('IoU val nhánh 1 (%)')
        ax.set_title('IoU từng lớp (nhánh 1) — ' + title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        epoch_axis(ax)
        save(fig, 'per_class_iou.png')

    # 4. conflict / CC ratios
    tr = [r for r in train if r.get('conflict_ratio') not in (None, '')]
    if tr or evals:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if tr:
            x = [_f(r['iteration']) for r in tr]
            ax.plot(x, [100 * _f(r['conflict_ratio']) for r in tr], lw=1.2, color='C3',
                    label='xung đột ŷ₁≠ŷ₂ (ảnh không nhãn, train)')
            ax.plot(x, [100 * _f(r['cc_ratio']) for r in tr], lw=1.2, color='C4',
                    label='CC: xung đột & tự tin > γ (TB 2 nhánh)')
        if evals:
            ax.plot([_f(r['iteration']) for r in evals], [100 * _f(r['val_conflict_ratio']) for r in evals],
                    'ko', ms=5, label='xung đột trên val (mỗi lần eval)')
        ax.set_xlabel('iteration')
        ax.set_ylabel('% pixel hợp lệ')
        ax.set_title('Tỉ lệ xung đột — ' + title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        epoch_axis(ax)
        save(fig, 'conflict_ratio.png')

    # 5. pseudo-label class distribution vs labeled ground truth
    if pseudo:
        import numpy as np
        cls = [k[len('pct_'):] for k in pseudo[0] if k.startswith('pct_')]
        palette = cfg.get('palette') or [[int(255 * k / max(len(cls) - 1, 1))] * 3 for k in range(len(cls))]
        base = [r for r in pseudo if r['branch'] == 'labeled_gt']
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, b in zip(axes, ('1', '2')):
            rows = ([('GT có nhãn', base[0])] if base else []) + \
                   [('ep %d' % (int(r['epoch']) + 1), r) for r in pseudo if r['branch'] == b]
            if not rows:
                continue
            bottom = np.zeros(len(rows))
            for k, c in enumerate(cls):
                vals = np.array([_f(r['pct_' + c]) for _, r in rows])
                ax.bar(range(len(rows)), vals, bottom=bottom, color=np.array(palette[k]) / 255.0,
                       edgecolor='0.4', lw=0.4, label=c.replace('_', ' '))
                bottom += vals
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels([n for n, _ in rows], rotation=60, fontsize=8)
            ax.set_title('Nhãn giả nhánh %s' % b)
        axes[0].set_ylabel('% pixel')
        axes[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
        fig.suptitle('Phân bố lớp nhãn giả (giữa hai lần eval) vs GT tập có nhãn — ' + title)
        save(fig, 'pseudo_label_class_dist.png')

    # 6. confidence / reliability of branch 1 on val
    if conf:
        x = [_f(r['iteration']) for r in conf]
        gamma = conf[0].get('gamma')
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, [100 * _f(r['conf_ratio']) for r in conf], 'o-', lw=2, color='C0',
                label='%% pixel có max-softmax > γ=%s' % gamma)
        ax.plot(x, [100 * _f(r['acc_conf']) for r in conf], 's-', color='C2', label='độ chính xác nhóm > γ')
        ax.plot(x, [100 * _f(r['acc_nonconf']) for r in conf], 'v--', color='C3', label='độ chính xác nhóm ≤ γ')
        ax.set_xlabel('iteration')
        ax.set_ylabel('%')
        ax.set_title('Độ tin cậy nhánh 1 trên val — ' + title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        epoch_axis(ax)
        save(fig, 'confidence_reliability.png')

    if not quiet:
        print('plots written to %s: %s' % (out_dir, ', '.join(written) or 'nothing (no logs yet)'))
    return written


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python tools/plot_run.py <run_dir>')
    plot_run(sys.argv[1])
