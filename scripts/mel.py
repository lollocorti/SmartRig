import os
import json
import torch
import numpy as np
import librosa
import soundfile as sf
from torch.utils.data import Dataset
import torchaudio.transforms as T

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

PARAM_RANGES = {
    "highpass_cutoff": (80.0, 350.0),
    "distortion_drive": (6.0, 30.0),
    "chorus_rate": (0.5, 2.2),
    "chorus_depth": (0.15, 0.50),
    "chorus_delay": (7.0, 20.0),
    "chorus_feedback": (0.10, 0.40),
    "chorus_mix": (0.20, 0.50),
    "delay_time": (0.18, 0.55),
    "delay_feedback": (0.15, 0.45),
    "delay_mix": (0.15, 0.45),
    "reverb_room": (0.2, 0.8),
    "reverb_damping": (0.2, 0.7),
    "reverb_wet": (0.15, 0.50),
    "reverb_dry": (0.8, 1.0),
    "reverb_width": (0.5, 1.0)
}

def normalize_val(val, min_v, max_v):
    """Normalizza un valore continuo nel range [0, 1]. Gestisce None e divisioni per zero."""
    if val is None:
        return 0.0
    if max_v == min_v:
        return 0.0
    return float(np.clip((val - min_v) / (max_v - min_v), 0.0, 1.0))

def compute_mel_feature(audio, sr=48000, n_mels=128, n_fft=2048, hop_length=512):
    """Funzione di estrazione Mel unica per training e inferenza."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # Usa ref=1.0 per preservare l'ampiezza e la dinamica reale introdotta da pedali ed amp
    mel_db = librosa.power_to_db(mel_spec, ref=1.0)
    
    # Normalizzazione lineare in range [0, 1] considerando dinamica audio tra -80dB e 0dB
    mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)
    return mel_norm.astype(np.float32)

def generate_mel_spectrograms(dataset_dir=DATASET_DIR):
    json_path = os.path.join(dataset_dir, "dataset_labels.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File di metadati non trovato: {json_path}")

    mel_dir = os.path.join(dataset_dir, "mel")
    os.makedirs(mel_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"--- Calcolo Spettrogrammi Mel ({len(metadata)} campioni) ---")
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
        mel_norm = compute_mel_feature(audio, sr=sr)
        np.save(mel_abs_path, mel_norm)

        # Salvataggio del percorso relativo verso la cartella mel/
        item["mel_file"] = os.path.join("mel", base_filename)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(metadata):
            print(f"Progresso Mel: {idx + 1}/{len(metadata)} spettrogrammi calcolati.")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print("Spettrogrammi generati e metadati aggiornati con successo!\n")


class GuitarDataset(Dataset):
    def __init__(self, dataset_dir=DATASET_DIR, is_train=True):
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        json_path = os.path.join(self.dataset_dir, "dataset_labels.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File di metadati non trovato: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        amps = set()
        for item in self.metadata:
            amp = item.get("amplifier")
            amp_name = f"{amp['brand']}_{amp['model']}" if amp else "UNKNOWN"
            amps.add(amp_name)
        
        self.amp_to_id = {amp_name: idx for idx, amp_name in enumerate(sorted(amps))}
        self.id_to_amp = {idx: amp_name for amp_name, idx in self.amp_to_id.items()}

        # SpecAugment applicato esclusivamente durante la fase di training
        self.time_masking = T.TimeMasking(time_mask_param=35)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # Converte 'mel\sample.npy' in 'mel/sample.npy' su Linux/Colab
        rel_mel_path = os.path.normpath(item["mel_file"])
        spec_path = os.path.join(self.dataset_dir, rel_mel_path)

        # Fallback di sicurezza nel caso i file siano estratti sfusi nella root
        if not os.path.exists(spec_path):
            spec_path = os.path.join(self.dataset_dir, os.path.basename(rel_mel_path))

        mel_norm = np.load(spec_path)
        x_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

        if self.is_train:
            x_tensor = self.freq_masking(x_tensor)
            x_tensor = self.time_masking(x_tensor)

        amp = item.get("amplifier")
        amp_name = f"{amp['brand']}_{amp['model']}" if amp else "UNKNOWN"
        amp_target = torch.tensor(self.amp_to_id[amp_name], dtype=torch.long)

        fx = item.get("signal_chain", {})
        
        hp = fx.get("highpass", {})
        dist = fx.get("drive", {})
        ch = fx.get("chorus", {})
        dl = fx.get("delay", {})
        rv = fx.get("reverb", {})

        onoff_targets = [
            float(hp.get("enabled", False)),
            float(dist.get("enabled", False)),
            float(ch.get("enabled", False)),
            float(dl.get("enabled", False)),
            float(rv.get("enabled", False))
        ]

        params_normalized = [
            normalize_val(hp.get("cutoff_frequency_hz"), *PARAM_RANGES["highpass_cutoff"]),
            normalize_val(dist.get("drive_db"), *PARAM_RANGES["distortion_drive"]),
            normalize_val(ch.get("rate_hz"), *PARAM_RANGES["chorus_rate"]),
            normalize_val(ch.get("depth"), *PARAM_RANGES["chorus_depth"]),
            normalize_val(ch.get("centre_delay_ms"), *PARAM_RANGES["chorus_delay"]),
            normalize_val(ch.get("feedback"), *PARAM_RANGES["chorus_feedback"]),
            normalize_val(ch.get("mix"), *PARAM_RANGES["chorus_mix"]),
            normalize_val(dl.get("delay_seconds"), *PARAM_RANGES["delay_time"]),
            normalize_val(dl.get("feedback"), *PARAM_RANGES["delay_feedback"]),
            normalize_val(dl.get("mix"), *PARAM_RANGES["delay_mix"]),
            normalize_val(rv.get("room_size"), *PARAM_RANGES["reverb_room"]),
            normalize_val(rv.get("damping"), *PARAM_RANGES["reverb_damping"]),
            normalize_val(rv.get("wet_level"), *PARAM_RANGES["reverb_wet"]),
            normalize_val(rv.get("dry_level"), *PARAM_RANGES["reverb_dry"]),
            normalize_val(rv.get("width"), *PARAM_RANGES["reverb_width"])
        ]

        return (
            x_tensor, 
            amp_target, 
            torch.tensor(onoff_targets, dtype=torch.float32), 
            torch.tensor(params_normalized, dtype=torch.float32)
        )