import os
import json
import random
import shutil
import warnings
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from pedalboard import VST3Plugin

import audio
import helix
from mel import generate_mel_spectrograms

# Soppressione dei warning di normalizzazione nella console
warnings.filterwarnings("ignore", category=UserWarning, module="pyloudnorm")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

IDMT_DATASET_DIR = os.path.join(PROJECT_ROOT, "IDMT-SMT-GUITAR_V2")
CACHE_48K_DIR = os.path.join(PROJECT_ROOT, "idmt_48k_cache")
OUTPUT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
PRESETS_DIR = os.path.join(PROJECT_ROOT, "presets")

HELIX_VST3_PATH = r"C:\Program Files\Common Files\VST3\Line 6\Helix Native (x64).vst3"

SAMPLE_RATE = 48000
CHUNK_DURATION = 3.0
TARGET_LUFS = -18.0
NUM_SAMPLES = 5000
N_MELS = 256
NUM_WORKERS = max(1, os.cpu_count() - 2)  # Riserva 2 core per il sistema


def _process_sample_batch(batch_indices: list, cached_files: list, loaded_presets: list, audio_dir: str) -> list:
    """Funzione eseguita in modo isolato da ciascun processo figlio."""
    # Prova a caricare il VST3; se fallisce la DLL nativa, interrompe in modo pulito solo questo worker
    plugin = None
    try:
        plugin = VST3Plugin(HELIX_VST3_PATH)
    except Exception as e:
        print(f"[ERROR Worker] Impossibile caricare VST3 nel worker: {e}")
        return []

    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    batch_metadata = []

    for sample_idx in batch_indices:
        try:
            audio_clean, selected_cache_file = audio.get_audio_chunk(cached_files, chunk_samples)

            if loaded_presets:
                selected_preset = random.choice(loaded_presets)
                preset_config = helix.parse_and_randomize_preset_blocks(selected_preset["preset_data"])
                
                # Applica i parametri al VST3
                signal_chain_label = {
                    "source_preset": selected_preset["filename"],
                    "config": preset_config
                }
            else:
                signal_chain_label = {}

            # Processamento audio
            audio_stereo_in = np.stack([audio_clean, audio_clean], axis=0)
            
            # Chiamata protetta al VST3
            audio_processing = plugin(audio_stereo_in, SAMPLE_RATE)

            if audio_processing.ndim > 1:
                audio_processing = np.mean(audio_processing, axis=0)

            audio_final = np.nan_to_num(audio_processing, nan=0.0, posinf=0.9, neginf=-0.9)
            audio_final = audio.normalize_lufs_or_rms(audio_final, sample_rate=SAMPLE_RATE, target_lufs=TARGET_LUFS)

            filename = f"sample_{sample_idx:05d}.wav"
            abs_audio_path = os.path.join(audio_dir, filename)
            sf.write(abs_audio_path, audio_final, SAMPLE_RATE)

            batch_metadata.append({
                "audio_file": os.path.join("audio", filename),
                "source_file": os.path.basename(selected_cache_file),
                "signal_chain": signal_chain_label
            })
        except Exception as e:
            # Cattura errori su singoli campioni
            continue

    # Pulisci il riferimento al plugin alla fine del batch
    del plugin
    return batch_metadata


def generate_audio_dataset_sequential(num_samples: int = NUM_SAMPLES, dataset_dir: str = OUTPUT_DATASET_DIR):
    audio_dir = os.path.join(dataset_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(PRESETS_DIR, exist_ok=True)

    if not os.path.exists(HELIX_VST3_PATH):
        raise FileNotFoundError(f"Binary VST3 non trovato: {HELIX_VST3_PATH}")

    print("[PREPROC] Inizializzazione cache audio 48 kHz...")
    cached_files = audio.prepare_48k_cache(IDMT_DATASET_DIR, CACHE_48K_DIR, SAMPLE_RATE)

    # Caricamento e controllo dei preset ordinati
    loaded_presets = helix.load_helix_presets(PRESETS_DIR)
    if not loaded_presets:
        raise FileNotFoundError(f"Nessun file .hlx trovato nella cartella: {PRESETS_DIR}")
    
    print(f"[PRESET] Caricati correttamente {len(loaded_presets)} preset (da {loaded_presets[0]['filename']} a {loaded_presets[-1]['filename']})")

    # Inizializza un'UNICA istanza VST3 persistente nel processo principale
    print("[VST3] Caricamento istanza Helix Native...")
    plugin = VST3Plugin(HELIX_VST3_PATH)

    metadata = []
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    print(f"[SEQUENTIAL] Avvio generazione di {num_samples} campioni...")

    for sample_idx in range(num_samples):
        try:
            audio_clean, selected_cache_file = audio.get_audio_chunk(cached_files, chunk_samples)

            if loaded_presets:
                selected_preset = random.choice(loaded_presets)
                preset_config = helix.parse_and_randomize_preset_blocks(selected_preset["preset_data"])

                signal_chain_label = {
                    "preset_file": selected_preset["filename"],         # es. "preset12.hlx"
                    "preset_original_name": preset_config["preset_name"], # es. "Archetype Lead"
                    "config": preset_config
                }
            else:
                signal_chain_label = {}

            # Processamento audio VST3
            audio_stereo_in = np.stack([audio_clean, audio_clean], axis=0)
            audio_processing = plugin(audio_stereo_in, SAMPLE_RATE)

            if audio_processing.ndim > 1:
                audio_processing = np.mean(audio_processing, axis=0)

            audio_final = np.nan_to_num(audio_processing, nan=0.0, posinf=0.9, neginf=-0.9)
            audio_final = audio.normalize_lufs_or_rms(audio_final, sample_rate=SAMPLE_RATE, target_lufs=TARGET_LUFS)

            filename = f"sample_{sample_idx:05d}.wav"
            abs_audio_path = os.path.join(audio_dir, filename)
            sf.write(abs_audio_path, audio_final, SAMPLE_RATE)

            metadata.append({
                "audio_file": os.path.join("audio", filename),
                "source_file": os.path.basename(selected_cache_file),
                "signal_chain": signal_chain_label
            })

            if (sample_idx + 1) % 50 == 0 or (sample_idx + 1) == num_samples:
                print(f"[PROGRESSO] Processati {sample_idx + 1}/{num_samples} campioni completati")

        except Exception as e:
            print(f"[WARN] Errore nell'elaborazione del campione {sample_idx}: {e}")
            continue

    json_path = os.path.join(dataset_dir, "dataset_labels.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[COMPLETATO] Generati con successo {len(metadata)} campioni audio.")

def create_colab_zip(dataset_dir: str = OUTPUT_DATASET_DIR):
    zip_base_name = os.path.join(PROJECT_ROOT, "dataset")
    temp_dir = os.path.join(PROJECT_ROOT, "temp_dataset")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        json_src = os.path.join(dataset_dir, "dataset_labels.json")
        mel_src = os.path.join(dataset_dir, "mel")

        if os.path.exists(json_src):
            shutil.copy(json_src, temp_dir)
        if os.path.exists(mel_src):
            shutil.copytree(mel_src, os.path.join(temp_dir, "mel"), dirs_exist_ok=True)

        shutil.make_archive(zip_base_name, "zip", temp_dir)
        print(f"[EXPORT] Archivio dataset generato: {zip_base_name}.zip")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    print("=== GENERAZIONE DATASET PARALLELA ===")

    generate_audio_dataset_sequential(num_samples=NUM_SAMPLES, dataset_dir=OUTPUT_DATASET_DIR)
    generate_mel_spectrograms(dataset_dir=OUTPUT_DATASET_DIR, sample_rate=SAMPLE_RATE, n_mels=N_MELS)
    create_colab_zip(dataset_dir=OUTPUT_DATASET_DIR)

    print("=== ESECUZIONE COMPLETATA ===")


if __name__ == "__main__":
    main()