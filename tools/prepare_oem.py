"""Data check + nested labeled/unlabeled split for ONE run (spec section 2). Called by tools/run_baseline.sh.

    python tools/prepare_oem.py --split 1_4 [--split_seed 0] [--full_unlabeled same|none]
                                [--import_labeled FILE] [--data_root DIR] [--part_root DIR]

1. Finds the OEM root (a directory holding images/{train,val} and label/{train,val}).
2. data_report.json (first call; reused while the image counts do not change): pairing, sample reads,
   label values of every train/val label (must be 0..8).
3. Writes <part_root>/<split dir>/{labeled,unlabeled}.txt, split_info.json, labeled_class_dist.json and
   <part_root>/val.txt. A fixed permutation from --split_seed takes the first k images as labeled, so
   1/16 c 1/8 c 1/4 c 1/2 across separate runs. Existing files with a different content -> error.
The last stdout line is PART_DIR=<path> (read by run_baseline.sh).
Exit codes: 0 ok, 2 data/split problem (message printed).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset.tif_io import read_label, read_rgb  # noqa: E402

CLASS_NAMES = ['Bareland', 'Rangeland', 'Developed space', 'Road', 'Tree', 'Water', 'Agriculture land', 'Building']
DENOMS = {'1_16': 16, '1_8': 8, '1_4': 4, '1_2': 2}


def fail(msg):
    print('[prepare_oem] ERROR: ' + msg, flush=True)
    sys.exit(2)


def find_data_root(search_root):
    hits = []
    for dirpath, dirnames, _ in os.walk(search_root):
        if dirpath[len(search_root):].count(os.sep) > 5:
            dirnames[:] = []
            continue
        if all(os.path.isdir(os.path.join(dirpath, a, b)) for a in ('images', 'label') for b in ('train', 'val')):
            hits.append(dirpath)
            dirnames[:] = []
    if len(hits) != 1:
        fail('need exactly one directory with images/{train,val} and label/{train,val} under %s, found %d: %s\n'
             'set DATA_ROOT explicitly.' % (search_root, len(hits), hits))
    return hits[0]


def list_pairs(root, split):
    imgs = sorted(f for f in os.listdir(os.path.join(root, 'images', split)) if f.lower().endswith(('.tif', '.tiff')))
    labs = sorted(f for f in os.listdir(os.path.join(root, 'label', split)) if f.lower().endswith(('.tif', '.tiff')))
    only_img, only_lab = sorted(set(imgs) - set(labs)), sorted(set(labs) - set(imgs))
    if only_img or only_lab:
        fail('%s: images and labels are not paired.\n  images without label (%d): %s\n  labels without image (%d): %s'
             % (split, len(only_img), only_img[:50], len(only_lab), only_lab[:50]))
    return imgs


def data_report(root, part_root, train, val):
    path = os.path.join(part_root, 'data_report.json')
    old = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None
    if old and old['n_train'] == len(train) and old['n_val'] == len(val) and old.get('data_root') == root:
        print('[prepare_oem] data_report.json reused: %d train / %d val' % (len(train), len(val)))
        return old
    if old:
        print('[prepare_oem] WARNING: image counts changed (report: %d train / %d val, now: %d / %d). The Kaggle '
              'dataset version probably changed: splits run before are NOT directly comparable with new ones.'
              % (old['n_train'], old['n_val'], len(train), len(val)), flush=True)
        os.replace(path, path.replace('.json', '_%s.json' % time.strftime('%Y%m%d_%H%M%S')))

    samples = []
    for name in (train[:10] + val[:10]):
        split = 'train' if name in train[:10] else 'val'
        img = np.array(read_rgb(os.path.join(root, 'images', split, name)))
        lab = read_label(os.path.join(root, 'label', split, name))
        samples.append({'name': name, 'split': split, 'image_shape': list(img.shape), 'image_dtype': str(img.dtype),
                        'label_shape': list(lab.shape), 'label_dtype': str(lab.dtype)})
        if img.shape[:2] != lab.shape:
            fail('%s/%s: image %s and label %s sizes differ' % (split, name, img.shape, lab.shape))
    stats = {}
    for split, names in (('train', train), ('val', val)):
        counts = np.zeros(256, dtype=np.int64)
        t0 = time.time()
        for k, name in enumerate(names):
            counts += np.bincount(read_label(os.path.join(root, 'label', split, name)).ravel(), minlength=256)
            if (k + 1) % 500 == 0:
                print('[prepare_oem] label stats %s: %d/%d (%.0f s)' % (split, k + 1, len(names), time.time() - t0),
                      flush=True)
        values = {int(v): int(counts[v]) for v in np.nonzero(counts)[0]}
        bad = [v for v in values if v > 8]
        if bad:
            fail('%s labels contain values outside 0..8: %s' % (split, {v: values[v] for v in bad}))
        stats[split] = values
    rep = {'data_root': root, 'n_train': len(train), 'n_val': len(val), 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
           'samples': samples, 'label_values': stats,
           'class_mapping': '0 unknown -> 255 (ignored); 1..8 -> 0..7 = ' + ', '.join(CLASS_NAMES)}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=2)
    print('[prepare_oem] data_report.json written: %d train / %d val, label values train %s'
          % (len(train), len(val), sorted(stats['train'])))
    return rep


def write_list(path, lines):
    text = ''.join(l + '\n' for l in lines)
    if os.path.exists(path):
        if open(path, encoding='utf-8').read() != text:
            fail('%s already exists with a different content; not overwriting.\n'
                 'This split was generated from other data or settings. Delete it only if you are sure.' % path)
        return False
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return True


def sha1(lines):
    return hashlib.sha1(''.join(l + '\n' for l in lines).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--full_unlabeled', default='same', choices=['same', 'none'])
    parser.add_argument('--import_labeled', default='')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--search_root', default='/kaggle/input')
    parser.add_argument('--part_root', default='/kaggle/working/partitions/oem')
    args = parser.parse_args()

    split = args.split
    if not (split in DENOMS or split == 'full' or (split.startswith('n') and split[1:].isdigit())):
        fail('invalid --split %r (1_16 | 1_8 | 1_4 | 1_2 | full | n<number>)' % split)
    root = os.path.abspath(args.data_root) if args.data_root else find_data_root(args.search_root)
    os.makedirs(args.part_root, exist_ok=True)
    train, val = list_pairs(root, 'train'), list_pairs(root, 'val')
    rep = data_report(root, args.part_root, train, val)
    N = len(train)
    line = lambda s, n: 'images/%s/%s label/%s/%s' % (s, n, s, n)  # noqa: E731

    write_list(os.path.join(args.part_root, 'val.txt'), [line('val', n) for n in val])

    if args.import_labeled:
        wanted = [os.path.basename(l.split()[0]) for l in open(args.import_labeled, encoding='utf-8') if l.strip()]
        missing = sorted(set(wanted) - set(train))
        if missing:
            fail('--import_labeled: %d names are not train images: %s' % (len(missing), missing[:50]))
        labeled = sorted(set(wanted))
        tag = '%s_imp%s_s%d' % (split, sha1(labeled)[:8], args.split_seed)
    else:
        perm = np.random.RandomState(args.split_seed).permutation(N)
        if split == 'full':
            k = N
        elif split in DENOMS:
            k = N // DENOMS[split]
        else:
            k = int(split[1:])
            if not 1 <= k <= N:
                fail('%s: need 1 <= k <= N=%d' % (split, N))
        labeled = sorted(train[j] for j in perm[:k])
        tag = ('full_%s_s%d' % (args.full_unlabeled, args.split_seed)) if split == 'full' else \
            '%s_s%d' % (split, args.split_seed)
    if split == 'full':
        unlabeled = list(train) if args.full_unlabeled == 'same' else []
    else:
        unlabeled = sorted(set(train) - set(labeled))
    if split != 'full' and set(labeled) & set(unlabeled):
        fail('labeled and unlabeled overlap')

    part_dir = os.path.join(args.part_root, tag)
    os.makedirs(part_dir, exist_ok=True)
    write_list(os.path.join(part_dir, 'labeled.txt'), [line('train', n) for n in labeled])
    write_list(os.path.join(part_dir, 'unlabeled.txt'), [line('train', n) for n in unlabeled])

    dist_path = os.path.join(part_dir, 'labeled_class_dist.json')
    if not os.path.exists(dist_path):
        counts = np.zeros(256, dtype=np.int64)
        for n in labeled:
            counts += np.bincount(read_label(os.path.join(root, 'label', 'train', n)).ravel(), minlength=256)
        c = counts[1:9]                                  # 0 = unknown (ignored); 1..8 -> classes 0..7
        with open(dist_path, 'w', encoding='utf-8') as f:
            json.dump({'source': 'ground truth of labeled.txt only (ignore excluded)', 'n_pixels': int(c.sum()),
                       'pct': {name: 100.0 * float(v) / max(int(c.sum()), 1) for name, v in zip(CLASS_NAMES, c)}},
                      f, indent=2)

    info = {'split': split, 'split_seed': args.split_seed, 'full_unlabeled': args.full_unlabeled if split == 'full'
            else None, 'import_labeled': args.import_labeled or None, 'tag': tag, 'data_root': root,
            'n_train': N, 'n_val': len(val), 'n_labeled': len(labeled), 'n_unlabeled': len(unlabeled),
            'labeled_sha1': sha1([line('train', n) for n in labeled]),
            'unlabeled_sha1': sha1([line('train', n) for n in unlabeled])}
    info_path = os.path.join(part_dir, 'split_info.json')
    if not os.path.exists(info_path):
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

    print('[prepare_oem] data root %s' % root)
    print('[prepare_oem] split %s (split_seed %d): N=%d train, k=%d labeled, %d unlabeled, %d val -> %s'
          % (split, args.split_seed, N, len(labeled), len(unlabeled), len(val), part_dir))
    if split == 'full':
        print('[prepare_oem] WARNING: SPLIT=full differs from the paper\'s "Full" row (VOC keeps 9,118 SBD images as '
              'unlabeled data). Here unlabeled = %s.' % ('the same train images without labels'
                                                         if args.full_unlabeled == 'same' else 'none (supervised only)'))
    print('PART_DIR=%s' % part_dir.replace(os.sep, '/'))


if __name__ == '__main__':
    main()
