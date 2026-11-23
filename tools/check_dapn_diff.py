"""Spec 3.2: every difference between CCVC_aug.py (origin) and CCVC_aug_dapn.py is marked [DAPN].

    python tools/check_dapn_diff.py [--origin CCVC_aug.py] [--dapn CCVC_aug_dapn.py]

Allowed: inserted lines inside '# [DAPN] begin ...' / '# [DAPN] end' blocks, and origin lines replaced by
lines that end with '# [DAPN]' (or sit inside a block). Anything else (an unmarked added/changed line, an
origin line deleted without a marked replacement, an unbalanced block) -> FAIL, exit code 1.
"""
import argparse
import difflib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEGIN = re.compile(r'^\s*# \[DAPN\] begin\b')
END = re.compile(r'^\s*# \[DAPN\] end\s*$')
LINE = re.compile(r'# \[DAPN\]\s*$')


def marked_lines(lines):
    """Indices of the DAPN file lines that are marked (inside a block, the block markers, or '# [DAPN]' lines)."""
    marked, errors, inside = set(), [], None
    for k, l in enumerate(lines):
        if BEGIN.match(l):
            if inside is not None:
                errors.append('line %d: nested [DAPN] begin (open since line %d)' % (k + 1, inside + 1))
            inside = k
            marked.add(k)
        elif END.match(l):
            if inside is None:
                errors.append('line %d: [DAPN] end without begin' % (k + 1))
            inside = None
            marked.add(k)
        elif inside is not None or LINE.search(l):
            marked.add(k)
    if inside is not None:
        errors.append('line %d: [DAPN] begin never closed' % (inside + 1))
    return marked, errors


def check(origin_path, dapn_path):
    a = open(origin_path, encoding='utf-8').read().split('\n')
    b = open(dapn_path, encoding='utf-8').read().split('\n')
    marked, errors = marked_lines(b)
    n_added = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        if tag == 'delete':
            errors.append('origin lines %d-%d deleted without a [DAPN] replacement: %r' % (i1 + 1, i2, a[i1]))
            continue
        for j in range(j1, j2):
            n_added += 1
            if j not in marked:
                errors.append('line %d not marked [DAPN]: %r' % (j + 1, b[j]))
    return errors, n_added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--origin', default=os.path.join(ROOT, 'CCVC_aug.py'))
    ap.add_argument('--dapn', default=os.path.join(ROOT, 'CCVC_aug_dapn.py'))
    a = ap.parse_args()
    errors, n = check(a.origin, a.dapn)
    for e in errors:
        print('  ' + e)
    print('%s  check_dapn_diff: %d changed/added line(s), %d unmarked' % ('PASS' if not errors else 'FAIL', n,
                                                                          len(errors)))
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
