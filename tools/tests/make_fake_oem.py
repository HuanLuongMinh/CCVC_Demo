"""Synthetic OpenEarthMap-like data + fake pretrained weights for CPU tests (spec 8.1.1).

    python tools/tests/make_fake_oem.py --out <dir> [--n_train 32 --n_val 8 --size 256]

<dir>/oem/images/{train,val}/*.tif  RGB uint8
<dir>/oem/label/{train,val}/*.tif   uint8 labels 0..8 (0 = unknown)
<dir>/model_lab/resnet101.pth       deep-stem ResNet-101 keys (+ fc.*) -> must be accepted
<dir>/bad_weights/*.pth             torchvision / missing / wrong-shape / extra-key variants -> must be rejected
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from model.backbone import resnet  # noqa: E402


def blob_label(rs, h, w):
    """Piecewise-constant label map with a few rectangles of classes 1..8 and some 0 (unknown)."""
    lab = np.full((h, w), rs.randint(1, 9), dtype=np.uint8)
    for _ in range(rs.randint(4, 10)):
        y, x = rs.randint(0, h), rs.randint(0, w)
        dy, dx = rs.randint(h // 10, h // 2), rs.randint(w // 10, w // 2)
        lab[y:y + dy, x:x + dx] = rs.randint(0, 9) if rs.rand() < 0.15 else rs.randint(1, 9)
    return lab


def image_from_label(rs, lab):
    colors = rs.randint(0, 256, size=(9, 3))
    img = colors[lab].astype(np.float32) + rs.normal(0, 20, size=lab.shape + (3,))
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--n_train', type=int, default=32)
    parser.add_argument('--n_val', type=int, default=8)
    parser.add_argument('--size', type=int, default=256, help='tile side; a few tiles use 3/4 of it (OEM mixes sizes)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rs = np.random.RandomState(args.seed)

    root = os.path.join(args.out, 'oem')
    for split, n in (('train', args.n_train), ('val', args.n_val)):
        os.makedirs(os.path.join(root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(root, 'label', split), exist_ok=True)
        for k in range(n):
            side = args.size if k % 4 else (args.size * 3) // 4
            lab = blob_label(rs, side, side)
            name = 'city%d_%d.tif' % (k % 5, k)
            Image.fromarray(image_from_label(rs, lab), mode='RGB').save(os.path.join(root, 'images', split, name))
            Image.fromarray(lab, mode='L').save(os.path.join(root, 'label', split, name))

    model = resnet.ResNet(resnet.Bottleneck, [3, 4, 23, 3], replace_stride_with_dilation=[False, False, True])
    good = dict(model.state_dict())
    good['fc.weight'] = torch.zeros(1000, 2048)
    good['fc.bias'] = torch.zeros(1000)
    os.makedirs(os.path.join(args.out, 'model_lab'), exist_ok=True)
    torch.save(good, os.path.join(args.out, 'model_lab', 'resnet101.pth'))

    bad_dir = os.path.join(args.out, 'bad_weights')
    os.makedirs(bad_dir, exist_ok=True)
    import torchvision
    torch.save(torchvision.models.resnet101().state_dict(), os.path.join(bad_dir, 'torchvision.pth'))
    missing = dict(good)
    del missing['layer3.5.conv2.weight']
    torch.save(missing, os.path.join(bad_dir, 'missing_key.pth'))
    shape = dict(good)
    shape['layer2.0.bn1.weight'] = torch.zeros(7)
    torch.save(shape, os.path.join(bad_dir, 'wrong_shape.pth'))
    nostem = {k: v for k, v in good.items() if not k.startswith('conv1.6')}
    torch.save(nostem, os.path.join(bad_dir, 'no_stem.pth'))
    extra = dict(good)
    extra['foo.bar'] = torch.zeros(1)
    torch.save(extra, os.path.join(bad_dir, 'extra_key.pth'))
    print('fake OEM: %d train / %d val tiles in %s; weights in %s' % (args.n_train, args.n_val, root,
                                                                       os.path.join(args.out, 'model_lab')))


if __name__ == '__main__':
    main()
