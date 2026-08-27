import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class CNNBackbone(nn.Module):
    """Estrazione feature 2D scalabile dagli spettrogrammi Mel"""
    def __init__(self, out_channels=512):
        super().__init__()
        c1, c2, c3 = out_channels // 8, out_channels // 4, out_channels // 2
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResBlock(c1, c2, stride=2)
        self.layer2 = ResBlock(c2, c3, stride=2)
        self.layer3 = ResBlock(c3, out_channels, stride=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)  # Shape: [B, out_channels, F', T']

class AmpNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = CNNBackbone(out_channels=512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.global_pool(feat).flatten(1)
        return self.classifier(feat)

class DriveNet(nn.Module):
    def __init__(self, num_types=3, num_models=10):
        super().__init__()
        self.backbone = CNNBackbone(out_channels=512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        self.head_type = nn.Linear(256, num_types)
        self.head_model = nn.Linear(256, num_models)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.global_pool(feat).flatten(1)
        feat = self.shared_fc(feat)
        return self.head_type(feat), self.head_model(feat)

class CabinetNet(nn.Module):
    def __init__(self, num_cabs=9):
        super().__init__()
        self.backbone = CNNBackbone(out_channels=256)
        self.time_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.freq_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_cabs)
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.time_pool(feat)
        feat = self.freq_pool(feat).flatten(1)
        return self.classifier(feat)

class ChorusNet(nn.Module):
    def __init__(self, num_params=5):
        super().__init__()
        self.backbone = CNNBackbone(out_channels=512)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.head_active = nn.Linear(256, 1)
        self.head_params = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_params),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.freq_pool(feat).squeeze(2)
        feat = feat.permute(0, 2, 1)

        gru_out, _ = self.gru(feat)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(gru_out * attn_weights, dim=1)

        dense = torch.relu(context[:, :256] + context[:, 256:])
        
        is_active = self.head_active(dense)
        params = self.head_params(dense)
        
        return is_active, params

class HighpassNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone(out_channels=128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head_active = nn.Linear(128, 1)
        self.head_cutoff = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.global_pool(self.backbone(x)).flatten(1)
        active_logits = self.head_active(feat)
        cutoff_val = self.head_cutoff(feat)
        return active_logits, cutoff_val