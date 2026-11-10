"""Example panels saved at every evaluation: Image | Pred branch 1 | Pred branch 2 | Conflict map | Ground truth."""
import numpy as np
from PIL import Image

IGNORE_GT_COLOR = (0, 0, 0)              # ground truth 'unknown' (255)
CONFLICT_COLOR = (255, 0, 255)           # y1 != y2
AGREE_COLOR = (235, 235, 235)
IGNORE_CONFLICT_COLOR = (110, 110, 110)  # ignore pixels in the conflict map
MAX_SIDE = 512                           # display resolution per cell


def colorize(label, palette):
    out = np.zeros(label.shape + (3,), dtype=np.uint8)
    out[:] = IGNORE_GT_COLOR
    for c, color in enumerate(palette):
        out[label == c] = color
    return out


def conflict_map(pred1, pred2, gt):
    valid = gt != 255
    conflict = (pred1 != pred2) & valid
    out = np.zeros(gt.shape + (3,), dtype=np.uint8)
    out[:] = AGREE_COLOR
    out[conflict] = CONFLICT_COLOR
    out[~valid] = IGNORE_CONFLICT_COLOR
    ratio = conflict.sum() / max(valid.sum(), 1)
    return out, ratio


def _shrink(arr, nearest):
    h, w = arr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale >= 1:
        return arr
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.array(Image.fromarray(arr).resize(size, Image.NEAREST if nearest else Image.BILINEAR))


def save_panel(samples, cfg, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not samples:
        return
    palette = cfg.get('palette') or [[int(255 * k / max(cfg['nclass'] - 1, 1))] * 3 for k in range(cfg['nclass'])]
    names = cfg.get('class_names', ['class %d' % c for c in range(cfg['nclass'])])
    cols = ['Image', 'Pred nhánh 1 (chính)', 'Pred nhánh 2', 'Conflict map (ŷ₁ ≠ ŷ₂)', 'Ground truth']
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(17, 3.5 * n + 1.6), squeeze=False)
    for r, s in enumerate(samples):
        cmap, ratio = conflict_map(s['pred1'], s['pred2'], s['gt'])
        cells = [_shrink(s['image'], False), _shrink(colorize(s['pred1'], palette), True),
                 _shrink(colorize(s['pred2'], palette), True), _shrink(cmap, True),
                 _shrink(colorize(s['gt'], palette), True)]
        for c, cell in enumerate(cells):
            ax = axes[r][c]
            ax.imshow(cell)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        axes[r][0].set_ylabel('%s\nmIoU nhánh 1 = %.2f' % (s['name'], s['miou1']), fontsize=9)
        axes[r][3].set_xlabel('xung đột: %.1f%%' % (100.0 * ratio), fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    handles = [Patch(facecolor=np.array(palette[c]) / 255.0, edgecolor='0.5', label=names[c])
               for c in range(len(names))]
    handles += [Patch(facecolor=np.array(IGNORE_GT_COLOR) / 255.0, edgecolor='0.5', label='Ignore (unknown) – GT'),
                Patch(facecolor=np.array(CONFLICT_COLOR) / 255.0, edgecolor='0.5', label='Xung đột'),
                Patch(facecolor=np.array(IGNORE_CONFLICT_COLOR) / 255.0, edgecolor='0.5', label='Ignore – conflict map')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 1.3 / (3.5 * n + 1.6), 1, 1 - 0.5 / (3.5 * n + 1.6)), h_pad=2.0)
    fig.savefig(path, dpi=80)
    plt.close(fig)
