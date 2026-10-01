import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T

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

        # Costruzione dei vocabolari dei modelli e della configurazione globale
        self.chain_config, self.model2id = self._build_chain_vocab()

    def _build_chain_vocab(self):
        """Analizza il dataset per mappare tutti i blocchi, modelli e max parametri."""
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

                    # Conta solo i parametri numerici (ignorando i booleani)
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
        
        # FIX 1: Lettura sicura del percorso cross-platform
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

                    # FIX 2: Estrazione ordinata e scalata dei parametri
                    params_dict = b_data.get("parameters", {})
                    sorted_keys = sorted(params_dict.keys())
                    
                    num_params = []
                    for k in sorted_keys:
                        val = params_dict[k]
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            # Se il valore eccede 1.0, applichiamo una normalizzazione naive dividendo per un valore alto
                            # (Questo aiuta la stabilità della rete neurale in attesa di una normalizzazione formale)
                            clean_val = float(val)
                            if clean_val > 1.0:
                                clean_val = clean_val / 1000.0  
                            num_params.append(clean_val)
                    
                    # Padding fino a max_params
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