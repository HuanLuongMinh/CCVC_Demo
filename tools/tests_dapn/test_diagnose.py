"""Spec 9.1 test 13: tools/diagnose_entanglement.py on constructed features (no model).

    python tools/tests_dapn/test_diagnose.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import torch  # noqa: E402

from diagnose_entanglement import boundary_map, sample_features, fit_em, accumulate, summarise  # noqa: E402

C, D, K = 8, 304, 16
FAILS = []


def check(name, cond, detail=''):
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, (' (%s)' % detail) if detail else ''), flush=True)
    if not cond:
        FAILS.append(name)


def make_images(mode, n_img=6, n_px=3000, seed=0):
    """List of (f, gt, y1, y2, boundary). mode: 'separable' (f follows GT), 'entangled' (f random),
    'wrong_like_pred' (f follows GT, but agree-wrong pixels carry the feature of the predicted class)."""
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(C, D, generator=g) * 3
    out = []
    for _ in range(n_img):
        gt = torch.randint(0, C, (n_px,), generator=g)
        y1, y2 = gt.clone(), gt.clone()
        r = torch.rand(n_px, generator=g)
        aw = r < 0.1                                           # agreed but wrong
        wrong = (gt + 1 + torch.randint(0, C - 1, (n_px,), generator=g)) % C
        y1[aw], y2[aw] = wrong[aw], wrong[aw]
        cf = (r >= 0.1) & (r < 0.2)                            # conflict
        y2[cf] = (gt[cf] + 1) % C
        if mode == 'entangled':
            f = torch.randn(n_px, D, generator=g) * 3
        else:
            src = gt.clone()
            if mode == 'wrong_like_pred':
                src[aw] = y1[aw]
            f = emb[src] + 0.3 * torch.randn(n_px, D, generator=g)
        b = torch.rand(n_px, generator=g) < 0.25
        out.append((f, gt, y1, y2, b))
    return out


def run(images, seeds=(0, 1, 2)):
    X = sample_features(iter(images), len(images), 6000, 0)
    centres = [fit_em(X, K, 20, 0.1, s, torch.device('cpu')) for s in seeds]
    counts = accumulate(iter(images), centres, C)
    return X, counts, summarise(counts, 0.6, ['c%d' % c for c in range(C)])


def main():
    imgs = make_images('separable')
    X, counts, (per_seed, o, per_class) = run(imgs)
    check('sampling: 6000 pixels (quota per image, then subsample)', X.shape == (6000, D), str(tuple(X.shape)))
    n_valid = sum(im[0].shape[0] for im in imgs)
    check('every pixel assigned once per EM seed', all(int(T.sum()) == n_valid for T in counts))
    shares = sum(o[g + '.share']['mean'] for g in ('agree_right', 'agree_wrong', 'conflict'))
    check('group shares sum to 1', abs(shares - 1) < 1e-9)
    check('separable features -> purity ~ 1 (all pixels, every group)',
          o['mean_purity_all_pixels']['mean'] > 0.99 and all(o[g + '.mean_purity']['mean'] > 0.99
                                                              for g in ('agree_right', 'agree_wrong', 'conflict')),
          '%.4f' % o['mean_purity_all_pixels']['mean'])
    check('separable -> no pixel in a mixed cluster', o['agree_wrong.in_mixed_cluster']['mean'] < 0.01)
    check('separable -> agree_wrong not in a pure cluster of the predicted class',
          o['agree_wrong.in_pure_cluster_of_predicted_class']['mean'] < 0.05)
    check('3 EM seeds -> mean and std reported', len(per_seed) == 3 and 'std' in o['agree_wrong.mean_purity'])
    check('per class: 8 classes, purity ~ 1', len(per_class) == C and
          all(per_class['c%d' % c]['agree_right.mean_purity']['mean'] > 0.99 for c in range(C)))
    check('near_boundary = share of boundary pixels (~0.25)', abs(o['agree_right.near_boundary']['mean'] - 0.25) < 0.03)

    _, _, (_, oe, _) = run(make_images('entangled'))
    check('entangled (random) features -> low purity, mostly mixed clusters',
          oe['mean_purity_all_pixels']['mean'] < 0.5 and oe['agree_right.in_mixed_cluster']['mean'] > 0.8,
          '%.3f' % oe['mean_purity_all_pixels']['mean'])

    _, _, (_, ow, _) = run(make_images('wrong_like_pred'))
    check('agree_wrong pixels carrying the predicted class feature -> in pure cluster of the predicted class',
          ow['agree_wrong.in_pure_cluster_of_predicted_class']['mean'] > 0.95,
          '%.3f' % ow['agree_wrong.in_pure_cluster_of_predicted_class']['mean'])

    m = torch.full((1, 20, 20), 255, dtype=torch.long)
    m[0, :, :10], m[0, :, 10:18] = 1, 2
    b = boundary_map(m, 3, C)
    check('boundary band: radius 3 around the class edge, padding is not an edge',
          bool(b[0, 5, 7]) and bool(b[0, 5, 12]) and not bool(b[0, 5, 6]) and not bool(b[0, 5, 13])
          and not bool(b[0, 5, 17]) and not bool(b[0, 5, 19]))

    print('test_diagnose: %d failed' % len(FAILS))
    sys.exit(len(FAILS))


if __name__ == '__main__':
    main()
