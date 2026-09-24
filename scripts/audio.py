import glob
import os
import warnings
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
warnings.filterwarnings("ignore", category=UserWarning, module="pyloudnorm")

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False

def prepare_48k_cache(idmt_dir: str, cache_dir: str, target_sr: int) -> list:
    os.makedirs(cache_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(idmt_dir, "**", "*.wav"), recursive=True)
    clean_files = [
        f for f in all_files 
        if "acoustic_mic" not in f.lower() and "acoustic_pickup" not in f.lower()
    ]

    if not clean_files:
        raise FileNotFoundError(f"Percorso sorgente vuoto o non valido: {idmt_dir}")

    cached_file_paths = []
    
    for idx, file_path in enumerate(clean_files):
        rel_path = os.path.relpath(file_path, idmt_dir)
        cache_filename = rel_path.replace(os.sep, "_")
        cached_path = os.path.join(cache_dir, cache_filename)
        cached_file_paths.append(cached_path)

        if os.path.exists(cached_path):
            continue

        audio, orig_sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if orig_sr != target_sr:
            gcd = np.gcd(orig_sr, target_sr)
            audio = resample_poly(audio, target_sr // gcd, orig_sr // gcd)

        sf.write(cached_path, audio, target_sr)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(clean_files):
            print(f"[CACHE] Processati {idx + 1}/{len(clean_files)} file")

    return cached_file_paths


def get_audio_chunk(cached_files: list, chunk_samples: int) -> tuple:
    audio_clean = None
    selected_cache_file = None

    while audio_clean is None:
        selected_cache_file = np.random.choice(cached_files)
        try:
            audio_clean, _ = sf.read(selected_cache_file)
        except Exception:
            pass

    if len(audio_clean) < chunk_samples:
        repeats = int(np.ceil(chunk_samples / len(audio_clean)))
        audio_clean = np.tile(audio_clean, repeats)[:chunk_samples]
    else:
        max_start = len(audio_clean) - chunk_samples
        start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        audio_clean = audio_clean[start_idx : start_idx + chunk_samples]

    return audio_clean, selected_cache_file


def normalize_lufs_or_rms(audio: np.ndarray, sample_rate: int, target_lufs: float = -18.0) -> np.ndarray:
    if np.all(audio == 0):
        return audio

    if HAS_PYLOUDNORM:
        try:
            meter = pyln.Meter(sample_rate)
            loudness = meter.integrated_loudness(audio)
            if not np.isinf(loudness):
                audio = pyln.normalize.loudness(audio, loudness, target_lufs)
        except Exception:
            pass
    else:
        rms = np.sqrt(np.mean(audio**2) + 1e-9)
        target_rms = 10 ** (target_lufs / 20.0)
        audio = audio * (target_rms / rms)

    max_peak = np.max(np.abs(audio))
    if max_peak > 0.99:
        audio = audio / max_peak * 0.99

    return audio