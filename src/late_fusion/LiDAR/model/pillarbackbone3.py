import math
import torch
import torch.nn as nn

class FastBlock(nn.Module):
    """Bloc de base optimisé (Depthwise Separable Convolution)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.05,inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class PillarBackbone(nn.Module):
    def __init__(self, in_channels=64, num_anchors=2):
        """
        in_channels: Nombre de canaux venant du PillarFeatureNet (souvent 64)
        num_anchors: (len(anchor_sizes) * len(anchor_rotations))
        """
        super().__init__()
        self.num_anchors = num_anchors
        
        # Encoder
        self.enc1 = FastBlock(in_channels, 32)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        
        self.enc2 = FastBlock(64, 64)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        
        # Bridge
        self.bridge = FastBlock(128, 128)
        
        # Decoder avec Skip Connections
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = FastBlock(128, 64) # Concat(64+64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = FastBlock(64, 32) # Concat(32+32)
        
        # Têtes : On projette sur num_anchors
        # CLS : 1 canal par ancre
        self.cls_head = nn.Conv2d(32, self.num_anchors * 1, 1)
        
        # Initialisation spécifique pour le classifieur (focal loss friendly)
        pi = 0.01
        bias_value = -math.log((1 - pi) / pi)
        nn.init.constant_(self.cls_head.bias, bias_value)
        
        # REG : 7 paramètres par ancre [x, y, z, w, l, h, theta]
        self.reg_head = nn.Conv2d(32, self.num_anchors * 7, 1)
        self.scale_reg = nn.Parameter(torch.ones(7))
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        d1 = self.down1(x1)
        x2 = self.enc2(d1)
        d2 = self.down2(x2)
        
        # Bridge
        b = self.bridge(d2)
        
        # Decoder
        u1 = self.up1(b)
        x_dec1 = self.dec1(torch.cat([u1, x2], dim=1))
        
        u2 = self.up2(x_dec1)
        x_dec2 = self.dec2(torch.cat([u2, x1], dim=1))
        
        cls_out = self.cls_head(x_dec2)
        reg_out = self.reg_head(x_dec2)
        N, C, H, W = reg_out.shape
        reg_out = reg_out.view(N, self.num_anchors, 7, H, W)
        reg_out = reg_out * self.scale_reg.view(1, 1, 7, 1, 1)
        reg_out = reg_out.view(N, C, H, W)
        # Sortie
        return cls_out, reg_out