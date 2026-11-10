"""Helpers for run_baseline.sh (keeps Python out of the shell script).

    python tools/run_meta.py data_root <split_info.json>             # prints the OEM root of a split
    python tools/run_meta.py write <split_info.json> <run_meta.json> key=value ...
"""
import json
import sys


def main():
    cmd = sys.argv[1]
    info = json.load(open(sys.argv[2], encoding='utf-8'))
    if cmd == 'data_root':
        print(info['data_root'])
    elif cmd == 'write':
        for kv in sys.argv[4:]:
            k, v = kv.split('=', 1)
            info[k] = {'true': True, 'false': False}.get(v, v)
        with open(sys.argv[3], 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
    else:
        sys.exit('unknown command %s' % cmd)


if __name__ == '__main__':
    main()
