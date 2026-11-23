# [DAPN copy] Source: https://github.com/HuanLuongMinh/DAPN-for-VHR-Remote-Sensing-Segmentation
#   branch main_final, commit 33ad6387068f78240e8b04f1dbddbc8dd6dc5d6c
#   file mmseg/models/losses/dapg_loss.py (sha256 939e7917b56c428516d7d6860a1332d52e92d47f8e632a09b87837fc789eb8f3)
# Computation copied verbatim. Only changes:
#   - relative import of LOSSES and @LOSSES.register_module removed
#   - __main__ demo block removed
# ---------------------------------------------------------------
# DAPCN: Dynamic Attention-based Prototype Clustering Network
# Reference: docs/DAPCN.md
# --------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F



class DAPGLoss(nn.Module):
    """Dynamic Attention-based Prototype Grouping Loss.

    Components:
    1. Intra-group compactness (L_intra): Features close to their assigned
       group prototype
    2. Inter-group separation (L_inter): Different prototypes should be
       distinct beyond margin
    3. Quality regularization (L_quality): Encourage well-defined groups

    Args:
        margin (float): Margin for inter-group separation loss. Default: 0.3.
        lambda_inter (float): Weight for L_inter. Default: 0.5.
        lambda_quality (float): Weight for L_quality. Default: 0.1.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    EPS = 1e-6

    def __init__(self, margin=0.3, lambda_inter=0.5, lambda_quality=0.1,
                 loss_weight=1.0):
        super().__init__()
        self.margin = margin
        self.lambda_inter = lambda_inter
        self.lambda_quality = lambda_quality
        self.loss_weight = loss_weight

    def forward(self, features, group_assignments, group_prototypes,
                group_quality):
        """Compute DAPG loss.

        Args:
            features (torch.Tensor): Raw features of shape (N, C)
            group_assignments (torch.Tensor): Soft assignment weights of
                shape (N, K')
            group_prototypes (torch.Tensor): Group prototype vectors of
                shape (K', C)
            group_quality (torch.Tensor): Quality scores of shape (K',)

        Returns:
            tuple: (loss, loss_dict) where loss is scalar tensor and
                loss_dict is dict with individual components
        """
        # Get dimensions
        n_groups = group_prototypes.shape[0]

        if torch.isnan(group_assignments).any():
            raise ValueError("DAPGLoss: group_assignments contains NaN")
        if torch.isnan(group_prototypes).any():
            raise ValueError("DAPGLoss: group_prototypes contains NaN")
        if torch.isnan(group_quality).any():
            raise ValueError("DAPGLoss: group_quality contains NaN")

        # Normalize features and prototypes for cosine similarity
        f_norm = F.normalize(features, p=2, dim=1)
        f_norm = torch.nan_to_num(f_norm, nan=0.0, posinf=0.0, neginf=0.0)
        q_norm = F.normalize(group_prototypes, p=2, dim=1)
        q_norm = torch.nan_to_num(q_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # L_intra: Intra-group compactness
        # Formula: (1/N) * sum_i sum_k A_ik * (1 - cos(F_i, Q_k))
        sim_intra = torch.mm(f_norm, q_norm.t())  # (N, K')
        sim_intra = torch.clamp(sim_intra, -1.0, 1.0)
        weighted_sim = (sim_intra * group_assignments).sum(dim=1)
        loss_intra = (1.0 - weighted_sim).mean()

        if torch.isnan(loss_intra):
            raise ValueError(
                f"DAPGLoss: loss_intra is NaN "
                f"(sim_intra range=[{sim_intra.min():.4f}, {sim_intra.max():.4f}], "
                f"assignments range=[{group_assignments.min():.4f}, {group_assignments.max():.4f}])"
            )

        # L_inter: Inter-group separation with margin
        # Formula: (1/K(K-1)) * sum_{g!=h} max(0, cos(Q_g, Q_h) - margin)
        if n_groups > 1:
            sim_inter = torch.mm(q_norm, q_norm.t())  # (K', K')
            mask = 1.0 - torch.eye(n_groups, device=sim_inter.device)
            margin_violations = F.relu(sim_inter - self.margin) * mask
            loss_inter = margin_violations.sum() / (n_groups *
                                                    (n_groups - 1))
        else:
            loss_inter = torch.tensor(0.0, device=features.device)

        # L_quality: Quality regularization
        # Formula: -mean(log(quality + epsilon))
        quality_safe = torch.nan_to_num(group_quality, nan=0.5, posinf=1.0, neginf=0.0)
        quality_clamped = torch.clamp(quality_safe, min=self.EPS, max=1.0)
        loss_quality = -torch.log(quality_clamped).mean()

        # Total loss
        loss_proto = self.loss_weight * (loss_intra +
                                          self.lambda_inter * loss_inter +
                                          self.lambda_quality * loss_quality)

        if torch.isnan(loss_proto):
            raise ValueError(
                f"DAPGLoss: NaN detected in loss_proto "
                f"(intra={loss_intra.item():.4f}, "
                f"inter={loss_inter.item():.4f}, "
                f"quality={loss_quality.item():.4f})"
            )


        loss_dict = {
            'loss_intra': loss_intra,
            'loss_inter': loss_inter,
            'loss_quality': loss_quality,
        }

        return loss_proto, loss_dict
