import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PillarBackbone(nn.Module):
    """This class defines a simple backbone for processing the pseudo-image generated from the LiDAR point cloud."""
    def __init__(self, in_channels=2, out_channels=7):
        """This backbone takes the pseudo-image as input and outputs a feature map that can be used for detection.
        in_channels: Number of channels in the input pseudo-image (e.g., 2 for height and density)  
        out_channels: Number of channels in the output feature map (e.g., 7 for (x,y,z,l,w,h,yaw) per cell)"""
        super(PillarBackbone, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        

        self.features = nn.Sequential(self.block1, self.block2)
        self.conv_out = nn.Conv2d(128, out_channels, kernel_size=1)
        self.detection_head = nn.Conv2d(128, out_channels, kernel_size=1)
    def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            out = self.detection_head(x)
            
            # 1. On sépare les canaux
            xyz = out[:, :3, :, :]
            lwh_raw = out[:, 3:6, :, :]
            yaw_raw = out[:, 6:7, :, :]
            
            # 2. On applique les transformations sur des variables distinctes
            # On utilise Softplus ou Exp, mais sur une nouvelle variable
            lwh = torch.exp(torch.clamp(lwh_raw, max=5.0))
            yaw = torch.tanh(yaw_raw) * 3.14159 # Utilise pi directement
            
            # 3. On concatène le tout pour créer le tenseur final
            # Cela crée un nouveau tenseur dans le graphe de calcul, c'est "safe"
            final_out = torch.cat([xyz, lwh, yaw], dim=1)
            
            return final_out