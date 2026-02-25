####################################################################################[start]
####################################################################################[start]

"""

Audio feature encoder for MDM audio conditioning.

Takes frame-level librosa features (T_audio × audio_feat_dim) and projects them
to the transformer's hidden dimension (T_audio × latent_dim) using a small 1D CNN.

The CNN has a receptive field of ~0.5s at 20fps (kernel_size=5, 3 layers → 13 frames),
giving each embedding local temporal context around beats and onsets.

"""

import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    
    """
    
    1D CNN that encodes per-frame audio features into the transformer's latent space.

    Input:  (batch_size, T_audio, audio_feat_dim) — e.g., (B, T, 145)
    Output: (batch_size, T_audio, latent_dim) — e.g., (B, T, 512)
    
    """

    def __init__(self, audio_feat_dim=145, latent_dim=512, dropout=0.1):
        
        super().__init__()
        
        self.audio_feat_dim = audio_feat_dim
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(

            # -- layer 1: audio_feat_dim → 256 --
            
            nn.Conv1d(audio_feat_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            # -- layer 2: 256 → 512 --
            
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),

            # -- layer 3: 512 → latent_dim --
            
            nn.Conv1d(512, latent_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            
        )

    def forward(self, audio_features):
        
        """
        
        Args:
            audio_features: (batch_size, T_audio, audio_feat_dim)

        Returns:
            audio_proj: (batch_size, T_audio, latent_dim)
        
        """
        
        # -- conv1d expects (B, C, T), so transpose --
        
        x = audio_features.transpose(1, 2)      # (B, audio_feat_dim, T)
        x = self.conv_layers(x)                 # (B, latent_dim, T)
        x = x.transpose(1, 2)                   # (B, T, latent_dim)
        
        return x
    
####################################################################################[end]
####################################################################################[end]
