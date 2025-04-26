import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_sigmoid=False, reduction='mean', loss_weight=1.0, num_classes=None):
        """
        Args:
            gamma (float): Focusing parameter for modulating factor (1 - p_t)^gamma.
            alpha (float or list or torch.Tensor): Weighting factor for the class imbalance.
                For multi-class (use_sigmoid=False), if a scalar is provided, it will be converted to a tensor of shape [num_classes].
            use_sigmoid (bool): If True, applies sigmoid on logits and uses binary cross-entropy loss.
                Otherwise, uses softmax on logits for multi-class classification.
            reduction (str): 'mean', 'sum', or 'none' for per-sample loss.
            loss_weight (float): A scalar multiplier for the final loss.
            num_classes (int): Number of classes. Required if use_sigmoid is False and alpha is a scalar.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if not self.use_sigmoid:
            # For multi-class classification using softmax, we require a per-class alpha vector.
            if isinstance(alpha, (float, int)):
                if num_classes is None:
                    raise ValueError("num_classes must be specified when use_sigmoid is False and alpha is a scalar")
                self.alpha = torch.full((num_classes,), alpha).to(self.device)
            elif isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha).to(self.device)
            else:
                self.alpha = alpha  # Assuming tensor
        else:
            # For binary/multi-label classification, alpha can be a scalar (float) or tensor.
            self.alpha = alpha

    def forward(self, logits, targets):
        if self.use_sigmoid:
            # --- Binary / multi-label focal loss ---
            # logits: [N, *] and targets should have the same shape (with values 0 or 1)
            prob = torch.sigmoid(logits)
            # binary cross entropy per element, no reduction
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss = bce_loss * ((1 - p_t) ** self.gamma)
            if self.alpha is not None:
                # If alpha is provided, apply it (it can be a scalar or of the same shape as targets)
                loss = self.alpha * loss

        else:
            # --- Multi-class focal loss ---
            # logits: [N, num_classes], targets: [N] with class indices.
            log_probs = F.log_softmax(logits, dim=1)  # [N, num_classes]
            probs = torch.exp(log_probs)               # [N, num_classes]
            # Gather the log-probabilities of the true labels
            targets = targets.view(-1, 1)  # [N, 1]
            logp_t = log_probs.gather(1, targets)  # [N, 1]
            p_t = probs.gather(1, targets)         # [N, 1]

            ce_loss = -logp_t  # standard cross entropy per sample
            loss = ce_loss * ((1 - p_t) ** self.gamma)  # focal modulation

            if self.alpha is not None:
                # Expecting self.alpha to be a tensor of shape [num_classes]
                alpha_tensor = self.alpha.to(logits.device)
                # For each sample, get the weight of the target class:
                alpha_factor = alpha_tensor.gather(0, targets.squeeze())
                loss = alpha_factor.view(-1, 1) * loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # Else, if reduction is 'none', keep the per-sample loss

        return self.loss_weight * loss
