import torch
import torch.nn as nn


__all__ = ['ResNet', 'resnet50', 'resnet101']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, multi_grid=False, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], multi_grid=multi_grid)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, multi_grid=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        grids = [1] * blocks
        if multi_grid:
            grids = [2, 2, 4]

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation * grids[0], norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation * grids[i],
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def base_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4


# deep-stem ResNet of UniMatch/CCVC (NOT torchvision): the stem must be present with these shapes
DEEP_STEM_SHAPES = {
    'conv1.0.weight': (64, 3, 3, 3),
    'conv1.3.weight': (64, 64, 3, 3),
    'conv1.6.weight': (128, 64, 3, 3),
    'bn1.weight': (128,), 'bn1.bias': (128,), 'bn1.running_mean': (128,), 'bn1.running_var': (128,),
}
ALLOWED_UNEXPECTED = {'fc.weight', 'fc.bias'}


def load_pretrained_file(path):
    try:
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:  # noqa: BLE001 - older pickles that need full unpickling
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('state_dict', 'model'):
        if isinstance(state_dict, dict) and key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
    return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}


def check_pretrained(model, state_dict):
    """Compare a backbone with a checkpoint. Returns (report, errors); errors name every offending key.

    Blocks on: a backbone key missing from the file or with a different shape (num_batches_tracked
    ignored), a missing/wrong deep-stem key, or a key in the file that is neither in the backbone nor
    fc.weight / fc.bias.
    """
    def skip(k):
        return k.endswith('num_batches_tracked')

    expected = {k: tuple(v.shape) for k, v in model.state_dict().items() if not skip(k)}
    given = {k: tuple(v.shape) for k, v in state_dict.items() if not skip(k)}
    missing = sorted(k for k in expected if k not in given)
    wrong_shape = sorted((k, expected[k], given[k]) for k in expected if k in given and given[k] != expected[k])
    stem = sorted((k, shape, given.get(k)) for k, shape in DEEP_STEM_SHAPES.items() if given.get(k) != shape)
    unexpected = sorted(k for k in given if k not in expected and k not in ALLOWED_UNEXPECTED)
    matched = len(expected) - len(missing) - len(wrong_shape)
    report = {'n_backbone_keys': len(expected), 'n_matched': matched,
              'match_ratio': matched / max(len(expected), 1),
              'ignored_extra': sorted(k for k in given if k in ALLOWED_UNEXPECTED)}
    sep = '\n    '
    errors = []
    if missing:
        errors.append('missing backbone keys (%d):' % len(missing) + sep + sep.join(missing))
    if wrong_shape:
        errors.append('wrong shape (%d):' % len(wrong_shape) + sep +
                      sep.join('%s: expected %s, file has %s' % t for t in wrong_shape))
    if stem:
        errors.append('deep-stem keys missing or wrong (%d) - is this a torchvision ResNet?' % len(stem) + sep +
                      sep.join('%s: expected %s, file has %s' % t for t in stem))
    if unexpected:
        errors.append('unexpected keys (%d):' % len(unexpected) + sep + sep.join(unexpected))
    return report, errors


def _main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


_reported = set()


def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        pretrained_path = "pretrained/%s.pth" % arch
        state_dict = load_pretrained_file(pretrained_path)
        report, errors = check_pretrained(model, state_dict)
        if errors:
            raise RuntimeError('pretrained weights %s rejected:\n  ' % pretrained_path + '\n  '.join(errors))
        if _main_process() and pretrained_path not in _reported:
            _reported.add(pretrained_path)
            print('[pretrained] %s: matched %d/%d backbone keys (%.1f%%), ignored %s' % (
                pretrained_path, report['n_matched'], report['n_backbone_keys'], 100 * report['match_ratio'],
                report['ignored_extra'] or 'nothing'), flush=True)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)
