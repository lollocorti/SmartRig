import os
import json
import torch
import numpy as np
import librosa
import soundfile as sf
from torch.utils.data import Dataset
import torchaudio.transforms as T


def compute_mel_feature(audio: np.ndarray, sr: int = 48000, n_mels: int = 256) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels
    )
    
    mel_db = librosa.power_to_db(mel_spec, ref=1.0)
    mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)
    return mel_norm.astype(np.float32)


def generate_mel_spectrograms(dataset_dir: str, sample_rate: int = 48000, n_mels: int = 256):
    json_path = os.path.join(dataset_dir, "dataset_labels.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File di metadati non trovato: {json_path}")

    mel_dir = os.path.join(dataset_dir, "mel")
    os.makedirs(mel_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[MEL] Calcolo spettrogrammi ({n_mels} mels, N={len(metadata)})...")
    for idx, item in enumerate(metadata):
        audio_rel_path = item.get("audio_file")
        if not audio_rel_path:
            continue
        
        audio_abs_path = os.path.join(dataset_dir, audio_rel_path)
        if not os.path.exists(audio_abs_path):
            continue

        base_filename = os.path.splitext(os.path.basename(audio_rel_path))[0] + ".npy"
        mel_abs_path = os.path.join(mel_dir, base_filename)

        audio, sr = sf.read(audio_abs_path)
        mel_norm = compute_mel_feature(audio, sr=sr, n_mels=n_mels)
        np.save(mel_abs_path, mel_norm)

        item["mel_file"] = os.path.join("mel", base_filename)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(metadata):
            print(f"[MEL] Processati {idx + 1}/{len(metadata)}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


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

        # 1. Costruzione dei vocabolari dei modelli e della configurazione globale
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

                    num_params = len(b_data.get("parameters", {}))
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
        # Nel metodo __getitem__ di mel.py
        rel_path = self.samples[idx]["mel_file"]

        # Sostituisce le barre rovesciate di Windows con quelle standard di Linux/POSIX
        rel_path_clean = rel_path.replace("\\", "/")

        # Costruisce il percorso in modo portabile su qualsiasi sistema operativo
        spec_path = os.path.normpath(os.path.join(self.dataset_dir, rel_path_clean))

        mel_norm = np.load(spec_path)
        x_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

        if self.is_train:
            x_tensor = self.freq_masking(x_tensor)
            x_tensor = self.time_masking(x_tensor)

        chain = item.get("signal_chain", {})
        targets = {}

        # 2. Estrazione omogenea dei target per ciascun blocco della catena globale
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

                    # Estrazione e padding dei parametri continui/numerici
                    raw_params = list(b_data.get("parameters", {}).values())
                    num_params = [p for p in raw_params if isinstance(p, (int, float)) and not isinstance(p, bool)]
                    
                    # Padding fino a max_params
                    padded_params = num_params[:max_params] + [0.0] * max(0, max_params - len(num_params))
                else:
                    # Blocco non presente nel singolo preset -> target neutri/disattivati
                    is_active = 0.0
                    model_idx = 0
                    padded_params = [0.0] * max_params

                targets[b_id] = {
                    "active": torch.tensor(is_active, dtype=torch.float32),
                    "model": torch.tensor(model_idx, dtype=torch.long),
                    "params": torch.tensor(padded_params, dtype=torch.float32)
                }
        else:
            # Fallback se le label sono in formato differente
            for cfg in self.chain_config:
                b_id = cfg["block_id"]
                targets[b_id] = {
                    "active": torch.tensor(0.0, dtype=torch.float32),
                    "model": torch.tensor(0, dtype=torch.long),
                    "params": torch.tensor([0.0] * cfg["max_params"], dtype=torch.float32)
                }

        return x_tensor, targets