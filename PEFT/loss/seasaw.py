import torch
import torch.nn as nn
import torch.nn.functional as F

class SeesawLoss(nn.Module):
    def __init__(self, num_classes, p=0.8, q=2.0, eps=1e-12, reduction='mean'):
        """
        Args:
            num_classes (int): Number of classes in the classification task.
            p (float): Exponent controlling the mitigation factor.
            q (float): Exponent controlling the compensation factor.
            eps (float): A small number to avoid division by zero.
            reduction (str): 'mean', 'sum', or 'none' (for per-sample losses).
        """
        super(SeesawLoss, self).__init__()
        # Set device for all tensor operations in the loss
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        self.reduction = reduction
        
        # Register a buffer for class counts on the correct device.
        # Initialize with ones to avoid division-by-zero.
        self.register_buffer('class_counts', torch.ones(num_classes, device=self.device))
    
    def update_class_counts(self, targets):
        """
        Updates the running count of samples per class.
        Args:
            targets (torch.Tensor): Tensor of class indices with shape [B].
        """
        with torch.no_grad():
            # Ensure targets are on CPU for bincount (or cast to correct device)
            bincount = torch.bincount(targets.cpu(), minlength=self.num_classes).float().to(self.device)
            self.class_counts += bincount

    def forward(self, logits, targets):
        """
        Computes the Seesaw Loss.
        Args:
            logits (torch.Tensor): Logits of shape [B, C].
            targets (torch.Tensor): Ground-truth class indices of shape [B].
        Returns:
            torch.Tensor: The computed loss (scalar if reduction is 'mean' or 'sum').
        """
        # Ensure targets are on the same device as logits
        targets = targets.to(self.device)
        logits = logits.to(self.device)
        
        # Update class counts with the current batch targets.
        self.update_class_counts(targets)
        
        # Compute log softmax and probabilities.
        log_probs = F.log_softmax(logits, dim=1)  # [B, C]
        probs = torch.exp(log_probs)               # [B, C]
        
        # Create one-hot mask for the ground-truth classes.
        class_mask = F.one_hot(targets, num_classes=self.num_classes).bool().to(self.device)
        
        # Extract target logits for compensation factor calculation.
        batch_size = targets.size(0)
        arange = torch.arange(batch_size, device=self.device)
        target_logits = logits[arange, targets].view(-1, 1)  # [B, 1]
        
        # ---- Compute Mitigation Factor ----
        counts = self.class_counts.clone()  # [C] already on self.device
        count_i = counts[targets].view(-1, 1)  # [B, 1]
        count_j = counts.view(1, -1)           # [1, C]
        mitigation_factor = (count_i / (count_j + self.eps)) ** self.p  # [B, C]
        mitigation_factor = torch.clamp(mitigation_factor, max=1.0)
        # Force target positions to have factor 1.
        mitigation_factor[class_mask] = 1.0
        
        # ---- Compute Compensation Factor ----
        compensation_factor = torch.ones_like(logits, device=self.device)
        compensation_mask = logits > target_logits  # [B, C]
        compensation_factor[compensation_mask] = (probs[compensation_mask]) ** self.q
        
        # ---- Combine Factors to form Seesaw Weights ----
        seesaw_weights = mitigation_factor * compensation_factor
        # Ensure target class weights remain 1.
        seesaw_weights[class_mask] = 1.0
        
        # ---- Adjust the logits using the seesaw weights ----
        adjusted_logits = logits - torch.log(seesaw_weights + self.eps)
        
        # ---- Compute the final cross-entropy loss ----
        loss = F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)
        return loss
