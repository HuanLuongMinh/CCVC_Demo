"""Check the CC mask of the CutMix (strong) pass, from tensors dumped by CCVC_aug.py --debug_dump_cc.

    python tools/tests/check_cc_mask.py <dump_iterA.pth> [<dump_iterB.pth> ...] --expect mixed|upstream [--require_differ]

CutMix pastes image flip(0)[b] = image B-1-b into image b inside the box. With --cc_mask_mixed True
(expect=mixed) the CC mask must, inside the box, equal the conflicting-and-confident mask of the PASTED
image's predictions (and of the image itself outside), i.e. the same source as the CutMix-ed pseudo-labels.
With --cc_mask_mixed False (expect=upstream) it equals the mask of the un-mixed image everywhere.
Every dump must be correct. Whether a batch can tell the two behaviours apart depends on the random batch
(the pasted and the own masks may coincide inside the box), so --require_differ asks for at least one
discriminating batch among the dumps.
"""
import argparse
import sys

import torch


def cc_mask(pred1, pred2, gamma, use_prob):
    p1 = pred1.softmax(dim=1) if use_prob else pred1
    p2 = pred2.softmax(dim=1) if use_prob else pred2
    c1, l1 = p1.max(dim=1)
    c2, l2 = p2.max(dim=1)
    conflict = l1 != l2
    return conflict & (c1 > gamma), conflict & (c2 > gamma)


def check(path, expect):
    """Returns (ok, differ) for one dump."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    B = d['unlabeled_pred1'].shape[0]
    box = d['cutmix_box'].bool()                                 # (B, H, W)
    flip = torch.arange(B - 1, -1, -1)
    ok = True

    def report(name, cond):
        nonlocal ok
        print('  %-4s %s' % ('OK' if cond else 'FAIL', name))
        ok &= bool(cond)

    print(path)
    report('batch has a non-empty CutMix box (%d px inside)' % int(box.sum()), box.any())
    exp_mix = d['unlabeled_pred1'].clone()
    exp_mix[box.unsqueeze(1).expand_as(exp_mix)] = d['unlabeled_pred1'][flip][box.unsqueeze(1).expand_as(exp_mix)]
    report('mixed predictions = pasted image inside the box, own image outside', torch.equal(exp_mix, d['mixed_pred1']))
    report('CutMix-ed pseudo-label 1 = argmax of the mixed predictions of branch 1',
           torch.equal(d['mixed_pred1'].argmax(dim=1), d['pseudo1_mixed']))

    own1, own2 = cc_mask(d['unlabeled_pred1'], d['unlabeled_pred2'], d['gamma'], d['conf_prob'])
    pasted1, pasted2 = own1[flip], own2[flip]
    differ = int(((pasted1 != own1) & box).sum() + ((pasted2 != own2) & box).sum())
    print('       pixels inside the box where the pasted and the own image CC masks differ: %d' % differ)

    if expect == 'mixed':
        report('CC mask (branch 1) inside the box = mask of the pasted image', torch.equal(d['cc1'][box], pasted1[box]))
        report('CC mask (branch 2) inside the box = mask of the pasted image', torch.equal(d['cc2'][box], pasted2[box]))
        report('CC masks outside the box = mask of the own image', torch.equal(d['cc1'][~box], own1[~box]) and
               torch.equal(d['cc2'][~box], own2[~box]))
        report('CC masks recomputed from the mixed predictions (softmax, gamma=%s) match' % d['gamma'],
               torch.equal(d['cc1'], torch.where(box, pasted1, own1)) and
               torch.equal(d['cc2'], torch.where(box, pasted2, own2)))
    else:
        report('upstream: CC masks = mask of the un-mixed image everywhere',
               torch.equal(d['cc1'], own1) and torch.equal(d['cc2'], own2))
        if differ:
            report('upstream: inside the box they do NOT follow the pasted image',
                   not torch.equal(d['cc1'][box], pasted1[box]) or not torch.equal(d['cc2'][box], pasted2[box]))
    return ok, differ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dumps', nargs='+')
    parser.add_argument('--expect', required=True, choices=['mixed', 'upstream'])
    parser.add_argument('--require_differ', action='store_true',
                        help='fail unless at least one dump can tell the two behaviours apart (use with real weights)')
    args = parser.parse_args()

    results = [check(p, args.expect) for p in args.dumps]
    all_ok = all(ok for ok, _ in results)
    discriminating = sum(1 for _, differ in results if differ > 0)
    print('%d dump(s): correct in %d, discriminating in %d' % (len(results), sum(ok for ok, _ in results),
                                                               discriminating))
    if discriminating == 0:
        msg = 'no dump can tell the two behaviours apart (masks coincide inside every box)'
        if args.require_differ:
            print('FAIL ' + msg)
            all_ok = False
        else:
            print('WARN ' + msg + '; run with real pretrained weights and --require_differ')
    print('CC MASK CHECK (%s): %s' % (args.expect, 'OK' if all_ok else 'FAILED'))
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
