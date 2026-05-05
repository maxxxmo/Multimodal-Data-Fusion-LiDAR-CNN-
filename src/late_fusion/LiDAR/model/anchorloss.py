import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.9, gamma=2.0):
    """
    logits  : tensor brut (avant sigmoid)
    targets : {0, 1}
    """

    targets = targets.float()

    # BCE de base
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    # Probabilités
    probs = torch.sigmoid(logits)

    # p_t
    p_t = probs * targets + (1 - probs) * (1 - targets)

    # Pondération alpha
    alpha_factor = (
        targets * alpha +
        (1 - targets) * (1 - alpha)
    )

    # Focal weighting
    focal_weight = alpha_factor * ((1 - p_t) ** gamma)

    loss = focal_weight * bce

    return loss.mean()


class AnchorDetectionLoss(nn.Module):

    def __init__(
        self,
        num_anchors,
        cls_weight=1.0,
        reg_weight=2.0,
        neg_pos_ratio=15
    ):
        super().__init__()
        self.reg_loss = 0.0
        self.cls_loss = 0.0
        self.num_anchors = num_anchors
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.neg_pos_ratio = neg_pos_ratio

        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def forward(
        self,
        preds_cls,
        preds_reg,
        targets_cls,
        targets_reg,
        pos_mask
    ):

        device = preds_cls.device

        # =====================================================
        # VALID MASK
        # =====================================================

        valid_mask = (targets_cls != -1)

        pos_mask_cls = (targets_cls == 1)

        neg_mask_cls = (targets_cls == 0)

        num_pos = pos_mask_cls.sum()

        if num_pos == 0:
            return torch.tensor(
                0.0,
                device=device,
                requires_grad=True
            )

        # =====================================================
        # HARD NEGATIVE SAMPLING
        # =====================================================

        neg_logits = preds_cls[neg_mask_cls]

        neg_losses = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits),
            reduction='none'
        )

        max_neg = min(
            neg_losses.numel(),
            num_pos.item() * self.neg_pos_ratio
        )

        hard_neg_idx = torch.topk(
            neg_losses,
            k=max_neg
        ).indices

        sampled_neg_mask = torch.zeros_like(
            neg_mask_cls,
            dtype=torch.bool
        )

        neg_indices = neg_mask_cls.nonzero(as_tuple=False)

        selected_neg_indices = neg_indices[
            hard_neg_idx
        ]

        sampled_neg_mask[
            selected_neg_indices[:, 0],
            selected_neg_indices[:, 1],
            selected_neg_indices[:, 2],
            selected_neg_indices[:, 3]
        ] = True

        # =====================================================
        # FINAL MASK
        # =====================================================

        final_cls_mask = (
            pos_mask_cls |
            sampled_neg_mask
        ) & valid_mask

        cls_logits = preds_cls[
            final_cls_mask
        ]

        cls_targets = targets_cls[
            final_cls_mask
        ]

        # =====================================================
        # CLS LOSS
        # =====================================================

        cls_loss = focal_loss(
            cls_logits,
            cls_targets,
            gamma=2.0
        )
        self.cls_loss = cls_loss
        # =====================================================
        # REG LOSS
        # =====================================================

        reg_mask = (
            pos_mask
            .unsqueeze(2)
            .repeat(1, 1, 8, 1, 1)
            .view(preds_reg.shape)
        )

        preds_reg_pos = (
            preds_reg[reg_mask]
            .view(-1, 8)
        )

        targets_reg_pos = (
            targets_reg[reg_mask]
            .view(-1, 8)
        )

        reg_loss = self.smooth_l1(
            preds_reg_pos,
            targets_reg_pos
        )
        self.reg_loss = reg_loss
        
        # =====================================================
        # DEBUG
        # =====================================================

        # print(
        #     f"Pos={num_pos.item()} | "
        #     f"Neg={max_neg} | "
        #     f"Cls={cls_loss.item():.4f} | "
        #     f"Reg={reg_loss.item():.4f}"
        # )

        # =====================================================
        # TOTAL
        # =====================================================

        total_loss = (
            self.cls_weight * cls_loss +
            self.reg_weight * reg_loss
        )

        return total_loss
    
    def get_losses(self):
        return (self.cls_weight * self.cls_loss, self.reg_weight * self.reg_loss)