"""CCVC two-branch DeepLabV3+ with one Dynamic Anchor module per branch (spec 3.1, 4.3).

DiscrepancyDeepLabV3PlusDAPN = Discrepancy_DeepLabV3Plus (origin, untouched) + da1 (branch 1) + da2 (branch 2).

- The origin part is built first by super().__init__ exactly as in the origin, then da1/da2 are created inside
  torch.random.fork_rng() with their own seed (training seed + DA_SEED_OFFSET): the global RNG stream is not
  consumed, so with the same seed the origin part is initialised bit-identically to CCVC_aug.py, with or
  without DAPN.
- use_dapg False: nothing is added; the module is the origin model.
- dapg_branches 'both' (default): da1 AND da2, one prototype set per sub-network. 'b1': da1 only (ablation).
- dapg_pos 'feat' (default): forward() is the origin forward; DAPG uses logits['feature1'/'feature2']
  (304 channels: branch 1 concat[reduce(c1), ASPP(c4)], branch 2 after mapping + dropout).
  dapg_pos 'c4': in training mode forward() also returns 'c4_1'/'c4_2' (layer4, 2048 channels); the same
  operations as the origin forward are used, so pred/feature are unchanged. Evaluation returns exactly the
  origin outputs.
"""
import torch
import torch.nn.functional as F

from model.semseg.deeplabv3plus import Discrepancy_DeepLabV3Plus
from model.dapn import DynamicAnchorModule

DA_SEED_OFFSET = 1000
FEATURE_DIM = {'feat': 2048 // 8 + 48, 'c4': 2048}


def da_kwargs(args):
    """DynamicAnchorModule arguments (spec 4.2) from the command line."""
    return dict(max_groups=args.da_max_groups, min_quality=args.da_min_quality, num_iters=args.da_num_iters,
                temperature=args.da_temperature, init_method=args.da_init, use_quality_gate=True,
                use_mask_predictor=False, ema_decay=args.da_ema_decay)


class DiscrepancyDeepLabV3PlusDAPN(Discrepancy_DeepLabV3Plus):
    def __init__(self, args, cfg):
        super(DiscrepancyDeepLabV3PlusDAPN, self).__init__(args, cfg)
        self.use_dapg = bool(getattr(args, 'use_dapg', False))
        self.dapg_pos = getattr(args, 'dapg_pos', 'feat')
        self.dapg_branches = getattr(args, 'dapg_branches', 'both')
        if not self.use_dapg:
            return
        dim = FEATURE_DIM[self.dapg_pos]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + DA_SEED_OFFSET)
            self.da1 = DynamicAnchorModule(dim, **da_kwargs(args))
            if self.dapg_branches == 'both':
                self.da2 = DynamicAnchorModule(dim, **da_kwargs(args))

    def da_modules(self):
        """{1: da1, 2: da2} for the branches that carry a DA module."""
        out = {}
        if self.use_dapg:
            out[1] = self.da1
            if self.dapg_branches == 'both':
                out[2] = self.da2
        return out

    def forward(self, x):
        if not (self.use_dapg and self.dapg_pos == 'c4' and self.training):
            return super(DiscrepancyDeepLabV3PlusDAPN, self).forward(x)
        logits = {}
        h, w = x.shape[-2:]
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp and x.is_cuda):
            out = []
            for br in (self.branch1, self.branch2):          # = DeepLabV3Plus.forward, keeping c4
                feats = br.backbone.base_forward(x)
                c1, c4 = feats[0], feats[-1]
                feature, pred = br._decode(c1, c4)
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)
                out.append((feature, pred, c4))
        logits['pred1'] = out[0][1].float()
        logits['feature1'] = out[0][0].float()
        logits['pred2'] = out[1][1].float()
        logits['feature2'] = out[1][0].float()
        logits['c4_1'] = out[0][2].float()
        logits['c4_2'] = out[1][2].float()
        return logits
