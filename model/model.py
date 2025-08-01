import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from typing import List, Tuple, Optional

class Mamba2GestureRecognizer(nn.Module):
    """
    Simplified Mamba2-based gesture recognition model for single classification.
    """
    
    def __init__(self, 
                 input_dim: int = 63,  # 21 landmarks * 3 coordinates
                 d_model: int = 256,   # Model dimension
                 num_classes: int = 14,  # All 14 gesture classes
                 num_layers: int = 4,   # Number of Mamba2 layers
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection: transform landmark features to model dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Stack of Mamba2 layers for sequence modeling
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=d_model) for _ in range(num_layers)
        ])
        
        # Layer normalization between Mamba blocks
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Final dropout before output
        self.final_dropout = nn.Dropout(dropout)
        
        # Output projection: map to gesture classes
        self.output_projection = nn.Linear(d_model, num_classes)
        
    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Actual sequence lengths (before padding)
            
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input landmarks to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Pass through Mamba2 layers with residual connections
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)  # Mamba2 processes the sequence
            x = layer_norm(x + residual)  # Add residual connection and normalize
        
        # Apply final dropout
        x = self.final_dropout(x)
        
        # Global average pooling to get fixed-size representation
        # Mask out padded positions if lengths are provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            x = x * mask
            # Average only over actual sequence length
            x = x.sum(dim=1) / lengths.unsqueeze(1).float()  # (batch_size, d_model)
        else:
            x = x.mean(dim=1)  # Simple average pooling
        
        # Project to output classes
        logits = self.output_projection(x)  # (batch_size, num_classes)
        
        return logits