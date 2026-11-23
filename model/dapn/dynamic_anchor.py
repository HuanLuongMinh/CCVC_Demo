# [DAPN copy] Source: https://github.com/HuanLuongMinh/DAPN-for-VHR-Remote-Sensing-Segmentation
#   branch main_final, commit 33ad6387068f78240e8b04f1dbddbc8dd6dc5d6c
#   file mmseg/models/utils/dynamic_anchor.py (sha256 c3ed8f88ddf30589fad536eae086fa7c6a613395fbf4ca3239d1a59822e3265b)
# Computation copied verbatim. Only changes:
#   - mmcv.runner.BaseModule -> torch.nn.Module; init_cfg argument removed
#   - imports of mmcv / mmseg.models.builder and @MODELS.register_module removed
# ---------------------------------------------------------------
# DynamicAnchorModule: Dataset-Level Learnable Prototype Discovery
# Reference: docs/DAPCN.md
#
# Revision history:
#   v1 — Per-batch transient prototypes (FPS init each forward pass)
#   v2 — Persistent learnable prototypes (nn.Parameter) with
#         per-batch EM refinement.
#   v3 — EMA memory bank as EM initialisation.  Refined prototypes
#         are saved to the bank after each forward pass; the next
#         forward starts from the bank instead of from scratch.
#         Cold start falls back to the nn.Parameter seed.
# ---------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F_func



class DynamicAnchorModule(nn.Module):
    """Dataset-level learnable prototype discovery with EM refinement.

    Maintains persistent prototypes that accumulate dataset-wide
    knowledge across the entire training trajectory.  When
    ``ema_decay > 0``, an EMA memory bank stores cross-batch
    prototype history and serves as the starting point for each
    forward pass's EM refinement, giving the model a warm start
    rather than re-discovering structure from scratch.

    Forward flow::

        memory bank (or seed)
              │
              ▼
        EM refinement (num_iters steps)
              │
              ▼
        refined prototypes ──► DAPGLoss
              │
              ▼
        EMA update ──► memory bank  (saved for next forward)

    Args:
        feature_dim (int): Dimension of input features.
        max_groups (int): Number of prototypes (K). Default: 64.
        min_quality (float): Quality threshold for filtering.
            Default: 0.1.
        num_iters (int): EM refinement iterations per forward pass.
            Default: 3.
        temperature (float): Temperature for soft assignment.
            Default: 0.1.
        init_method (str): One-time seed initialisation
            ('xavier', 'kaiming', 'normal'). Default: 'xavier'.
        use_quality_gate (bool): Enable quality-net filtering.
            Default: True.
        use_mask_predictor (bool): Enable per-pixel attention mask.
            Default: False.
        ema_decay (float): EMA momentum for the memory bank.
            0 disables the bank (EM starts from nn.Parameter seed).
            Default: 0.0.
        init_cfg (dict, optional): MMEngine init config. Default: None.
    """

    EPS = 1e-6
    MAX_PROTO_NORM = 10.0
    MIN_PROTO_NORM = 0.1

    def __init__(self,
                 feature_dim,
                 max_groups=64,
                 min_quality=0.1,
                 num_iters=3,
                 temperature=0.1,
                 init_method='xavier',
                 use_quality_gate=True,
                 use_mask_predictor=False,
                 ema_decay=0.0):
        super(DynamicAnchorModule, self).__init__()
        self.feature_dim = feature_dim
        self.max_groups = max_groups
        self.min_quality = min_quality
        self.num_iters = num_iters
        self.temperature = temperature
        self.init_method = init_method
        self.use_quality_gate = use_quality_gate
        self.use_mask_predictor = use_mask_predictor
        self.ema_decay = ema_decay

        # ---- Seed prototypes (cold-start only) ----
        self.prototypes = nn.Parameter(
            torch.empty(max_groups, feature_dim))
        self._init_prototypes()

        # ---- Quality estimator (optional) ----
        if use_quality_gate:
            self.quality_net = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1),
                nn.Sigmoid()
            )

        # ---- Mask predictor (optional) ----
        if use_mask_predictor:
            self.mask_predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1),
                nn.Sigmoid()
            )

        # ---- EMA memory bank ----
        if ema_decay > 0:
            self.register_buffer(
                '_ema_prototypes',
                torch.zeros(max_groups, feature_dim))
            self.register_buffer(
                '_ema_initialised', torch.tensor(False))

    def _init_prototypes(self):
        """One-time initialisation of the seed prototype tensor."""
        if self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.prototypes)
        elif self.init_method == 'kaiming':
            nn.init.kaiming_normal_(
                self.prototypes, mode='fan_out', nonlinearity='relu')
        elif self.init_method == 'normal':
            std = 1.0 / math.sqrt(self.feature_dim)
            nn.init.normal_(self.prototypes, mean=0.0, std=std)
        else:
            raise ValueError(
                f"Unknown init_method: '{self.init_method}'. "
                f"Choose from 'xavier', 'kaiming', 'normal'.")

    # ------------------------------------------------------------------
    # EM initialisation
    # ------------------------------------------------------------------
    def _get_initial_prototypes(self):
        """Return normalised starting prototypes for EM refinement.

        Uses the EMA memory bank when available (warm start),
        otherwise falls back to the learnable seed (cold start).
        """
        if self.ema_decay > 0 and self._ema_initialised:
            source = self._ema_prototypes
        else:
            source = self.prototypes

        proto = F_func.normalize(source, dim=1)
        proto = torch.where(
            torch.isnan(proto).any(dim=1, keepdim=True),
            torch.randn_like(proto) * 0.01,
            proto
        )
        return proto

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, features):
        """Forward pass with EM refinement.

        Args:
            features (torch.Tensor): (B, C, H, W) encoder/decoder features.

        Returns:
            tuple:
                - assign_valid (Tensor): Soft assignments (N, K').
                - proto_valid (Tensor): Refined prototypes (K', C).
                - quality_valid (Tensor): Quality scores (K',).
        """
        B, C, H, W = features.shape
        N = B * H * W
        feats = features.permute(0, 2, 3, 1).reshape(N, C)

        feats_norm = F_func.normalize(feats, dim=1)
        feats_norm = torch.where(
            torch.isnan(feats_norm).any(dim=1, keepdim=True),
            torch.zeros_like(feats_norm),
            feats_norm
        )

        # ---- EM initialisation: memory bank or seed ----
        proto = self._get_initial_prototypes()

        # ---- Differentiable EM refinement ----
        assign = None
        for _ in range(self.num_iters):
            # E-step: soft assignment
            sim = torch.mm(feats_norm, proto.t()) / self.temperature
            sim = torch.clamp(sim, min=-50, max=50)
            assign = torch.softmax(sim, dim=1)

            # M-step: recompute prototypes
            group_sizes = assign.sum(dim=0).clamp(min=self.EPS * 100)
            proto_new = torch.mm(assign.t(), feats) / group_sizes.unsqueeze(1)

            proto_norms = torch.norm(proto_new, dim=1, keepdim=True)
            proto_norms = torch.clamp(
                proto_norms, min=self.MIN_PROTO_NORM, max=self.MAX_PROTO_NORM)
            proto = F_func.normalize(proto_new / proto_norms, dim=1)

        # Edge case: num_iters == 0
        if assign is None:
            sim = torch.mm(feats_norm, proto.t()) / self.temperature
            assign = torch.softmax(sim, dim=1)

        # ---- Save refined prototypes to memory bank ----
        if self.ema_decay > 0 and self.training:
            self._update_memory(proto.detach())

        # ---- Quality-guided filtering ----
        if self.use_quality_gate:
            group_quality = self.quality_net(proto).squeeze(-1)
            valid_mask = group_quality > self.min_quality

            if valid_mask.sum() == 0:
                valid_mask[group_quality.argmax()] = True

            assign_valid = assign[:, valid_mask]
            proto_valid = proto[valid_mask]
            quality_valid = group_quality[valid_mask]
        else:
            assign_valid = assign
            proto_valid = proto
            quality_valid = torch.ones(
                proto.shape[0], device=proto.device)

        # ---- Optional per-pixel attention mask ----
        if self.use_mask_predictor:
            mask = self.mask_predictor(feats).squeeze(-1)
            assign_valid = assign_valid * mask.unsqueeze(1)
            assign_valid = assign_valid / (
                assign_valid.sum(dim=1, keepdim=True) + self.EPS)

        return assign_valid, proto_valid, quality_valid

    # ------------------------------------------------------------------
    # Memory bank
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_memory(self, proto_refined):
        """Save refined prototypes to the EMA memory bank."""
        if not self._ema_initialised:
            self._ema_prototypes.copy_(proto_refined)
            self._ema_initialised.fill_(True)
        else:
            self._ema_prototypes.mul_(self.ema_decay).add_(
                proto_refined, alpha=1.0 - self.ema_decay)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def extra_repr(self):
        return (
            f'feature_dim={self.feature_dim}, '
            f'max_groups={self.max_groups}, '
            f'num_iters={self.num_iters}, '
            f'temperature={self.temperature}, '
            f'init_method={self.init_method}, '
            f'quality_gate={self.use_quality_gate}, '
            f'mask_predictor={self.use_mask_predictor}, '
            f'ema_decay={self.ema_decay}'
        )
