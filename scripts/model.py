import torch
import torch.nn as nn
from typing import Dict, List, Any

class ResBlock(nn.Module):
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


class CNNBackbone(nn.Module):
    """Estrattore di feature 2D dagli spettrogrammi Mel."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.drop(x)
        return self.layer3(x)  # [B, out_channels, F_feat, T_feat]


class DSPBlockHead(nn.Module):
    """ Testa di stima per un singolo blocco della catena di segnale (es. dsp0_block1).
    Predice:
    1. Active / Enabled (Logit)
    2. Modello di effetto/amp/cab (Logits di classificazione)
    3. Valori dei parametri (Vettore continuo in [0, 1])
    """
    def __init__(self, in_features: int, num_models: int, max_params: int):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        
        # 1. Stato del blocco (Attivo / Bypass)
        self.head_active = nn.Linear(256, 1)
        
        # 2. Modello DSP selezionato
        self.head_model = nn.Linear(256, num_models)
        
        # 3. Parametri fisici del blocco
        self.head_params = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, max_params),
            nn.Sigmoid()  # Mantiene i parametri nello spazio normalizzato [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.shared(x)
        return {
            "active_logits": self.head_active(feat).squeeze(-1),
            "model_logits": self.head_model(feat),
            "params": self.head_params(feat)
        }


class FullSignalChainEstimator(nn.Module):
    """ Architettura unificata per la stima dell'intera catena di segnale Helix.
    """
    def __init__(self, block_configs: List[Dict[str, Any]], feature_dim: int = 512):
        """
        Args:
            block_configs: Lista di dizionari definente la struttura della catena. Esempio:
                [
                    {"block_id": "dsp0_block0", "num_models": 15, "max_params": 8},
                    {"block_id": "dsp0_block1", "num_models": 45, "max_params": 15},
                    ...
                ]
        """
        super().__init__()
        self.backbone = CNNBackbone(out_channels=feature_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.block_heads = nn.ModuleDict()
        self.block_ids = []
        
        for cfg in block_configs:
            b_id = cfg["block_id"]
            self.block_ids.append(b_id)
            self.block_heads[b_id] = DSPBlockHead(
                in_features=feature_dim,
                num_models=cfg["num_models"],
                max_params=cfg["max_params"]
            )

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        feat = self.backbone(x)
        feat = self.global_pool(feat).flatten(1)
        
        outputs = {}
        for b_id in self.block_ids:
            outputs[b_id] = self.block_heads[b_id](feat)
            
        return outputs