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

