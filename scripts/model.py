import torch
import torch.nn as nn


class ResBlock(nn.Module):

  def __init__(self, in_channels: int, out_channels: int, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(
        out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_channels,
              out_channels,
              kernel_size=1,
              stride=stride,
              bias=False,
          ),
          nn.BatchNorm2d(out_channels),
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = self.shortcut(x)
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += residual
    return self.relu(out)


class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, max_len: int = 1000):
    super().__init__()
    self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    seq_len = x.size(1)
    return x + self.pos_embedding[:, :seq_len, :]


class CNNTransformerBackbone(nn.Module):

  def __init__(
      self,
      out_channels: int = 512,
      num_transformer_layers: int = 4,
      nhead: int = 8,
  ):
    super().__init__()

    c1, c2, c3 = out_channels // 8, out_channels // 4, out_channels // 2

    self.stem = nn.Sequential(
        nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c1),
        nn.ReLU(),
    )
    self.layer1 = ResBlock(c1, c2, stride=2)
    self.layer2 = ResBlock(c2, c3, stride=2)
    self.layer3 = ResBlock(c3, out_channels, stride=2)

    self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
    self.pos_encoder = PositionalEncoding(d_model=out_channels)

    # Pre-LayerNormalization attivata (norm_first=True)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=out_channels,
        nhead=nhead,
        dim_feedforward=out_channels * 4,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=True,  # Stabilizza l'addestramento da zero
    )
    self.transformer = nn.TransformerEncoder(
        encoder_layer, num_layers=num_transformer_layers
    )
    self.out_norm = nn.LayerNorm(out_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.freq_pool(x).squeeze(2).permute(0, 2, 1)
    x = self.pos_encoder(x)
    x = self.transformer(x)

    out = self.out_norm(x.mean(dim=1))
    return out


class DSPBlockHead(nn.Module):

  def __init__(self, in_features: int, num_models: int, max_params: int):
    super().__init__()

    self.shared = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
    )

    self.head_active = nn.Linear(256, 1)
    self.head_model = nn.Linear(256, num_models)
    self.head_params = nn.Sequential(
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, max_params),
        nn.Sigmoid(),
    )

  def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    feat = self.shared(x)
    return {
        "active_logits": self.head_active(feat).squeeze(-1),
        "model_logits": self.head_model(feat),
        "params": self.head_params(feat),
    }


class FullSignalChainEstimator(nn.Module):

  def __init__(
      self, block_configs: list[dict[str, any]], feature_dim: int = 512
  ):
    super().__init__()
    self.backbone = CNNTransformerBackbone(out_channels=feature_dim)

    self.block_heads = nn.ModuleDict()
    self.block_ids = []

    for cfg in block_configs:
      b_id = cfg["block_id"]
      self.block_ids.append(b_id)
      self.block_heads[b_id] = DSPBlockHead(
          in_features=feature_dim,
          num_models=cfg["num_models"],
          max_params=cfg["max_params"],
      )

  def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    feat = self.backbone(x)
    return {b_id: self.block_heads[b_id](feat) for b_id in self.block_ids}