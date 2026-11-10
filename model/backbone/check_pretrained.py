"""Check a ResNet checkpoint before training (called by tools/run_baseline.sh).

    python -m model.backbone.check_pretrained --path pretrained/resnet101.pth --arch resnet101
Exit code 0 = accepted, 1 = rejected (the offending keys are listed).
"""
import argparse
import sys

from model.backbone import resnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--arch', default='resnet101', choices=['resnet50', 'resnet101'])
    args = parser.parse_args()

    layers = {'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3]}[args.arch]
    model = resnet.ResNet(resnet.Bottleneck, layers, replace_stride_with_dilation=[False, False, True])
    report, errors = resnet.check_pretrained(model, resnet.load_pretrained_file(args.path))
    print('[pretrained] %s: matched %d/%d backbone keys (%.1f%%; spec threshold 95%%), ignored %s'
          % (args.path, report['n_matched'], report['n_backbone_keys'], 100 * report['match_ratio'],
             report['ignored_extra'] or 'nothing'))
    if errors:
        print('[pretrained] REJECTED:\n  ' + '\n  '.join(errors))
        sys.exit(1)
    print('[pretrained] OK (deep-stem ResNet, every backbone key present with the right shape)')


if __name__ == '__main__':
    main()
