"""Check the OEM loader on real tiles (labels 0..8, mapping, unlabeled GT never opened, padding).

    python tools/tests/check_loader_real.py --data_root /kaggle/input/<oem>          # images/{split}, label/{split}
    python tools/tests/check_loader_real.py --label_root D:/x/label --image_root D:/x/OpenEarthMap_wo_xBD
        (image_root searched recursively by file name, e.g. the original <city>/images layout)
"""
import argparse
import glob
import os
import random
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from dataset.aug_semi2 import SemiDataset  # noqa: E402
from dataset.tif_io import read_label  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='')
    parser.add_argument('--label_root', default='')
    parser.add_argument('--image_root', default='')
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()

    if args.data_root:
        label_root, find_img = os.path.join(args.data_root, 'label'), \
            lambda split, n: os.path.join(args.data_root, 'images', split, n)
    else:
        index = {os.path.basename(p): p for p in glob.glob(os.path.join(args.image_root, '**', '*.tif'), recursive=True)
                 if os.sep + 'label' not in p}
        label_root, find_img = args.label_root, lambda split, n: index.get(n)

    ok = True
    lines = {}
    for split in ('train', 'val'):
        names = sorted(os.listdir(os.path.join(label_root, split)))
        missing = [n for n in names if not find_img(split, n) or not os.path.exists(find_img(split, n))]
        print('%s: %d labels, %d without image %s' % (split, len(names), len(missing), missing[:5]))
        ok &= not missing
        random.seed(0)
        pick = random.sample(names, min(args.n, len(names)))
        lines[split] = ['%s %s' % (find_img(split, n), os.path.join(label_root, split, n)) for n in pick]
        vals = set()
        for n in pick:
            vals |= set(np.unique(read_label(os.path.join(label_root, split, n))).tolist())
        print('  raw label values in %d samples: %s' % (len(pick), sorted(vals)))
        ok &= vals <= set(range(9))

    tmp = tempfile.mkdtemp()
    for split in ('train', 'val'):
        with open(os.path.join(tmp, split + '.txt'), 'w') as f:
            f.write('\n'.join(lines[split]))
    opened = []
    orig = Image.open
    Image.open = lambda fp, *a, **k: (opened.append(str(fp)), orig(fp, *a, **k))[1]

    tl = SemiDataset('oem', '', 'train_l', 512, os.path.join(tmp, 'train.txt'))
    for k in range(len(tl)):
        x, y = tl[k]
        u = set(torch.unique(y).tolist())
        ok &= tuple(x.shape) == (3, 512, 512) and u <= set(range(8)) | {255}
    print('train_l: images (3,512,512), mask values within {0..7, 255}')

    tu = SemiDataset('oem', '', 'train_u', 512, os.path.join(tmp, 'train.txt'))
    opened.clear()
    for k in range(len(tu)):
        tu[k]
    lab_opened = [p for p in opened if os.path.normpath(label_root) in os.path.normpath(p)]
    print('train_u: label files opened = %d (must be 0)' % len(lab_opened))
    ok &= not lab_opened

    va = SemiDataset('oem', '', 'val', val_id_path=os.path.join(tmp, 'val.txt'))
    img, mask, _ = va[0]
    raw = read_label(lines['val'][0].split(' ')[1])
    m = mask.numpy()
    mapping_ok = np.all(m[raw == 0] == 255) and all(np.all(m[raw == c] == c - 1) for c in range(1, 9))
    print('val: full size %s, mapping 0->255, c->c-1 exact: %s' % (tuple(img.shape), mapping_ok))
    ok &= bool(mapping_ok)
    print('LOADER CHECK: %s' % ('OK' if ok else 'FAILED'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
