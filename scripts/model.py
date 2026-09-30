import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Residual Block 2D con due convoluzioni 3x3 e skip connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class DualPoolingBackbone(nn.Module):
    """ Estrattore di feature per spettrogrammi Mel.
    Combina Global Average Pooling e Global Max Pooling per catturare sia 
    l'energia spettrale media sia i transienti ad alta dinamica.
    """
    def __init__(self, out_channels: int = 512):
        super().__init__()
        c1, c2, c3 = out_channels // 8, out_channels // 4, out_channels // 2
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResBlock(c1, c2, stride=2)
        self.layer2 = ResBlock(c2, c3, stride=2)
        self.drop = nn.Dropout2d(p=0.1)
        self.layer3 = ResBlock(c3, out_channels, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.drop(x)
        x = self.layer3(x)  # [B, C, F, T]

        # Concatenazione di AvgPool e MaxPool -> Feature vector di dimensione C * 2
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        return torch.cat([avg_feat, max_feat], dim=1)  # [B, out_channels * 2]


class DSPBlockHead(nn.Module):
    """ Head di stima multi-task per un singolo blocco DSP.
    Utilizza LayerNorm per garantire stabilità anche con batch size = 1.
    """
    def __init__(self, in_features: int, num_models: int, max_params: int):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        
        # 1. Stato del blocco (Attivo / Bypass)
        self.head_active = nn.Linear(256, 1)
        
        # 2. Modello DSP selezionato (Classificazione)
        self.head_model = nn.Linear(256, num_models)
        
        # 3. Parametri fisici del blocco (Regressione in [0, 1])
        self.head_params = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, max_params),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.shared(x)
        return {
            "active_logits": self.head_active(feat).squeeze(-1),
            "model_logits": self.head_model(feat),
            "params": self.head_params(feat)
        }


class FullSignalChainEstimator(nn.Module):
    """ Architettura unificata per la stima dell'intera catena di segnale.
    """
    def __init__(self, block_configs: list[dict[str, any]], feature_dim: int = 512):
        super().__init__()
        self.backbone = DualPoolingBackbone(out_channels=feature_dim)
        
        # Feature input per le head = feature_dim * 2 per via del Dual Pooling
        head_in_features = feature_dim * 2
        
        self.block_heads = nn.ModuleDict()
        self.block_ids = []
        
        for cfg in block_configs:
            b_id = cfg["block_id"]
            self.block_ids.append(b_id)
            self.block_heads[b_id] = DSPBlockHead(
                in_features=head_in_features,
                num_models=cfg["num_models"],
                max_params=cfg["max_params"]
            )

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        feat = self.backbone(x)  # [B, feature_dim * 2]
        
        outputs = {}
        for b_id in self.block_ids:
            outputs[b_id] = self.block_heads[b_id](feat)
            
        return outputs