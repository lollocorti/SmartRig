import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T


def normalize_param_value(key: str, val: float) -> float:
    """Normalizza qualsiasi valore di parametro Helix nell'intervallo preciso [0.0, 1.0]."""
    clean_val = float(val)
    k_lower = key.lower()

    # 1. Frequenze (Hz): scala logaritmica da 20Hz a 20000Hz
    if any(x in k_lower for x in ["freq", "hz", "cut", "hpf", "lpf"]):
        clean_val = max(20.0, min(20000.0, clean_val))
        log_val = np.log10(clean_val)
        norm_val = (log_val - np.log10(20.0)) / (np.log10(20000.0) - np.log10(20.0))
        return float(np.clip(norm_val, 0.0, 1.0))

    # 2. Tempi e Delay (ms): scala lineare fino a 5000ms
    if any(x in k_lower for x in ["time", "delay", "ms"]):
        return float(np.clip(clean_val / 5000.0, 0.0, 1.0))

    # 3. Valori già compresi tra 0 e 1
    if 0.0 <= clean_val <= 1.0:
        return float(clean_val)

    # 4. Fallback per guadagni o percentuali estese (> 1.0)
    if clean_val > 1.0:
        return float(np.clip(clean_val / 100.0, 0.0, 1.0))

    # 5. Valori negativi (es. dB fino a -80dB)
    if clean_val < 0.0:
        return float(np.clip((clean_val + 80.0) / 100.0, 0.0, 1.0))

    return float(np.clip(clean_val, 0.0, 1.0))


class GuitarDataset(Dataset):
    def __init__(self, dataset_dir: str, is_train: bool = True):
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        json_path = os.path.join(self.dataset_dir, "dataset_labels.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File di metadati non trovato: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.time_masking = T.TimeMasking(time_mask_param=35)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=30)

        self.chain_config, self.model2id = self._build_chain_vocab()

    def _build_chain_vocab(self):
        blocks_info = {}

        for item in self.metadata:
            chain = item.get("signal_chain", {})
            if "config" in chain and "blocks" in chain["config"]:
                blocks = chain["config"]["blocks"]
                for b_id, b_data in blocks.items():
                    if b_id not in blocks_info:
                        blocks_info[b_id] = {"models": set(), "max_params": 0}

                    model_name = b_data.get("model", "Unknown")
                    blocks_info[b_id]["models"].add(model_name)

                    params_dict = b_data.get("parameters", {})
                    num_params = sum(1 for v in params_dict.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
                    
                    if num_params > blocks_info[b_id]["max_params"]:
                        blocks_info[b_id]["max_params"] = num_params

        chain_config = []
        model2id = {}

        for b_id, info in sorted(blocks_info.items()):
            models_list = sorted(list(info["models"]))
            model2id[b_id] = {m_name: idx for idx, m_name in enumerate(models_list)}

            chain_config.append({
                "block_id": b_id,
                "num_models": max(len(models_list), 1),
                "max_params": max(info["max_params"], 1)
            })

        return chain_config, model2id

    def get_chain_config(self):
        return self.chain_config

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        rel_path = item.get("mel_file", "").replace("\\", "/")
        if rel_path.startswith("mel/"):
            rel_path = rel_path[4:]
        
        spec_path = os.path.join(self.dataset_dir, "mel", os.path.basename(rel_path))

        mel_norm = np.load(spec_path)
        x_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

        if self.is_train:
            x_tensor = self.freq_masking(x_tensor)
            x_tensor = self.time_masking(x_tensor)

        chain = item.get("signal_chain", {})
        targets = {}

        if "config" in chain and "blocks" in chain["config"]:
            sample_blocks = chain["config"]["blocks"]

            for cfg in self.chain_config:
                b_id = cfg["block_id"]
                max_params = cfg["max_params"]

                if b_id in sample_blocks:
                    b_data = sample_blocks[b_id]
                    is_active = 1.0 if b_data.get("enabled", False) else 0.0

                    m_name = b_data.get("model", "Unknown")
                    model_idx = self.model2id[b_id].get(m_name, 0)

                    params_dict = b_data.get("parameters", {})
                    sorted_keys = sorted(params_dict.keys())

                    num_params = []
                    for k in sorted_keys:
                        val = params_dict[k]
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            # Applicazione della normalizzazione specifica per tipo
                            clean_val = normalize_param_value(k, val)
                            num_params.append(clean_val)
                    
                    padded_params = num_params[:max_params] + [0.0] * max(0, max_params - len(num_params))
                else:
                    is_active = 0.0
                    model_idx = 0
                    padded_params = [0.0] * max_params

                targets[b_id] = {
                    "active": torch.tensor(is_active, dtype=torch.float32),
                    "model": torch.tensor(model_idx, dtype=torch.long),
                    "params": torch.tensor(padded_params, dtype=torch.float32)
                }
        else:
            for cfg in self.chain_config:
                b_id = cfg["block_id"]
                targets[b_id] = {
                    "active": torch.tensor(0.0, dtype=torch.float32),
                    "model": torch.tensor(0, dtype=torch.long),
                    "params": torch.tensor([0.0] * cfg["max_params"], dtype=torch.float32)
                }

        return x_tensor, targets