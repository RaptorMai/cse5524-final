import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLIP2Loss(nn.Module):
    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        """
        SigLIP2 loss: a sigmoid-based contrastive loss for vision-language models.
        
        Args:
            temperature (float): Scaling factor to control sharpness of logits.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super(SigLIP2Loss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute the SigLIP2 loss between image and text embeddings.

        Args:
            image_features (Tensor): [B, D] image embeddings.
            text_features (Tensor): [B, D] text embeddings.

        Returns:
            Tensor: Scalar loss if reduced, else [B, B] matrix of losses.
        """
        # Normalize embeddings
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarity matrix and scale by temperature
        logits = torch.matmul(image_features, text_features.T) / self.temperature  # [B, B]

        # Create binary labels: positive on diagonal, negative elsewhere
        batch_size = logits.size(0)
        targets = torch.eye(batch_size, device=logits.device)

        # Compute binary cross-entropy with logits
        loss_i2t = F.binary_cross_entropy_with_logits(logits, targets, reduction=self.reduction)
        loss_t2i = F.binary_cross_entropy_with_logits(logits.T,  targets, reduction=self.reduction)

        # Return the average of both directions
        loss = 0.5 * (loss_i2t + loss_t2i)
        return loss
