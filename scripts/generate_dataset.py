import os
import json
import random
import warnings
import gc
import multiprocessing as mp
import numpy as np
import soundfile as sf
import librosa
from pedalboard import VST3Plugin

import audio
import helix

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
BATCH_SIZE = 50  # Processa 50 campioni per ogni istanza del VST3 prima di riciclare il processo


def render_batch_worker(
    tasks: list,
    sample_rate: int,
    target_lufs: float,
    audio_dir: str,
    mel_dir: str,
    vst_path: str,
    queue: mp.Queue
):
    """
    Inizializza Helix Native UNA VOLTA sola per un batch di 50 campioni.
    Invia i dati alla Queue in tempo reale dopo ogni singolo campione elaborato.
    """
    try:
        plugin = VST3Plugin(vst_path)

        for sample_idx, audio_clean, selected_preset, cache_filename in tasks:
            try:
                preset_config = helix.parse_and_randomize_preset_blocks(selected_preset["preset_data"])

                audio_stereo_in = np.stack([audio_clean, audio_clean], axis=0)
                audio_processing = plugin(audio_stereo_in, sample_rate)

                if audio_processing.ndim > 1:
                    audio_processing = np.mean(audio_processing, axis=0)

                audio_final = np.nan_to_num(audio_processing, nan=0.0, posinf=0.9, neginf=-0.9)
                audio_final = audio.normalize_lufs_or_rms(audio_final, sample_rate=sample_rate, target_lufs=target_lufs)
                audio_final = np.ascontiguousarray(audio_final, dtype=np.float32)

                # 1. Salvataggio File Audio WAV
                wav_filename = f"sample_{sample_idx:05d}.wav"
                abs_audio_path = os.path.join(audio_dir, wav_filename)
                sf.write(abs_audio_path, audio_final, sample_rate)

                # 2. Calcolo e salvataggio Spettrogramma Mel (.npy)
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_final, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=N_MELS
                )
                mel_db = librosa.power_to_db(mel_spec, ref=1.0)
                mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)

                npy_filename = f"sample_{sample_idx:05d}.npy"
                abs_mel_path = os.path.join(mel_dir, npy_filename)
                np.save(abs_mel_path, mel_norm)

                # 3. Metadati per JSON
                signal_chain_label = {
                    "preset_file": selected_preset["filename"],
                    "preset_original_name": preset_config.get("preset_name", "Unknown"),
                    "config": preset_config
                }

                metadata_entry = {
                    "audio_file": os.path.join("audio", wav_filename),
                    "mel_file": os.path.join("mel", npy_filename),
                    "source_file": os.path.basename(cache_filename),
                    "signal_chain": signal_chain_label
                }

                # Invio immediato alla coda principale
                queue.put(("ITEM_SUCCESS", sample_idx, metadata_entry))

            except Exception as item_err:
                queue.put(("ITEM_ERROR", sample_idx, str(item_err)))

        del plugin
        gc.collect()

    except Exception as batch_err:
        queue.put(("BATCH_FATAL", None, str(batch_err)))


def generate_dataset_optimized(num_samples: int = NUM_SAMPLES, dataset_dir: str = OUTPUT_DATASET_DIR):
    audio_dir = os.path.join(dataset_dir, "audio")
    mel_dir = os.path.join(dataset_dir, "mel")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(PRESETS_DIR, exist_ok=True)

    if not os.path.exists(HELIX_VST3_PATH):
        raise FileNotFoundError(f"Binary VST3 non trovato: {HELIX_VST3_PATH}")

    print("[PREPROC] Inizializzazione cache audio 48 kHz...")
    cached_files = audio.prepare_48k_cache(IDMT_DATASET_DIR, CACHE_48K_DIR, SAMPLE_RATE)

    loaded_presets = helix.load_helix_presets(PRESETS_DIR)
    if not loaded_presets:
        raise FileNotFoundError(f"Nessun file .hlx trovato in {PRESETS_DIR}")
    
    print(f"[PRESET] Caricati {len(loaded_presets)} preset .hlx validati")

    # --- CONTROLLO RIPRESA E VERIFICA FILE ---
    json_path = os.path.join(dataset_dir, "dataset_labels.json")
    metadata = []
    processed_indices = set()

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            valid_metadata = []
            for item in metadata:
                # 1. Normalizzazione stringhe percorso: converte le barre Windows (\) in formato POSIX (/)
                rel_audio_path = item.get("audio_file", "").replace("\\", "/")
                rel_mel_path = item.get("mel_file", "").replace("\\", "/")

                # Aggiorna le chiavi dell'item così i percorsi salvati nel JSON saranno puliti e senza '\\'
                item["audio_file"] = rel_audio_path
                if "mel_file" in item:
                    item["mel_file"] = rel_mel_path

                # 2. Costruzione percorsi assoluti portabili cross-platform
                abs_audio_path = os.path.normpath(os.path.join(dataset_dir, rel_audio_path))
                abs_mel_path = os.path.normpath(os.path.join(dataset_dir, rel_mel_path)) if rel_mel_path else ""

                # 3. Controllo di esistenza flessibile (se mel non è richiesto/presente, controlla solo l'audio)
                audio_exists = os.path.exists(abs_audio_path)
                mel_exists = os.path.exists(abs_mel_path) if abs_mel_path else True

                if audio_exists and mel_exists:
                    valid_metadata.append(item)
                    fname = os.path.basename(rel_audio_path)
                    idx_str = fname.replace("sample_", "").replace(".wav", "")
                    if idx_str.isdigit():
                        processed_indices.add(int(idx_str))

            metadata = valid_metadata
            print(f"[RIPRESA] Verificati {len(metadata)} campioni completi nel JSON. Ripresa in corso...")

        except Exception as e:
            print(f"[WARN] Impossibile leggere JSON ({e}), reset metadati.")
            metadata = []
            processed_indices = set()

    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    missing_indices = [idx for idx in range(num_samples) if idx not in processed_indices]

    print(f"[GENERAZIONE] Rimanenti {len(missing_indices)} campioni da elaborare...")

    # Processa a blocchi di BATCH_SIZE
    for i in range(0, len(missing_indices), BATCH_SIZE):
        batch_indices = missing_indices[i:i + BATCH_SIZE]
        tasks = []

        for s_idx in batch_indices:
            audio_clean, selected_cache_file = audio.get_audio_chunk(cached_files, chunk_samples)
            selected_preset = random.choice(loaded_presets)
            tasks.append((s_idx, audio_clean, selected_preset, selected_cache_file))

        queue = mp.Queue()
        worker = mp.Process(
            target=render_batch_worker,
            args=(tasks, SAMPLE_RATE, TARGET_LUFS, audio_dir, mel_dir, HELIX_VST3_PATH, queue)
        )
        worker.start()

        # Legge i dati inviati in tempo reale durante l'esecuzione del batch
        while worker.is_alive() or not queue.empty():
            try:
                status, s_idx, payload = queue.get(timeout=1.0)

                if status == "ITEM_SUCCESS":
                    metadata.append(payload)
                    processed_indices.add(s_idx)

                    # Scrittura sincrona e immediata su JSON per ogni singolo campione
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=4)

                    print(f"[OK] Campione {s_idx:05d} salvato + label aggiornata nel JSON ({len(processed_indices)}/{num_samples})")

                elif status == "ITEM_ERROR":
                    print(f"[WARN] Errore sul campione {s_idx}: {payload}")

                elif status == "BATCH_FATAL":
                    print(f"[ERROR] Errore fatale VST nel batch: {payload}")

            except mp.queues.Empty:
                pass

        worker.join(timeout=5)

        if worker.is_alive():
            print(f"[TIMEOUT] Il batch a partire da {batch_indices[0]} è bloccato. Terminazione processo...")
            worker.terminate()
            worker.join()

    print(f"[COMPLETATO] Generazione terminata. Totale campioni nel dataset: {len(metadata)}")
    

def main():
    mp.set_start_method("spawn", force=True)
    print("=== GENERAZIONE DATASET REAL-TIME (SmartRig) ===")
    generate_dataset_optimized(num_samples=NUM_SAMPLES, dataset_dir=OUTPUT_DATASET_DIR)
    print("=== ESECUZIONE COMPLETATA ===")


if __name__ == "__main__":
    main()