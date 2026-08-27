import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
import torch

from nam.models import init_from_nam
from pedalboard import (
    Chorus,
    Delay,
    Distortion,
    HighpassFilter,
    Limiter,
    Pedalboard,
    Reverb,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

IDMT_DATASET_DIR = os.path.join(PROJECT_ROOT, "IDMT-SMT-GUITAR_V2")
AMP_HEADS_DIR = os.path.join(PROJECT_ROOT, "amp_heads")
PEDALS_DIR = os.path.join(PROJECT_ROOT, "pedals")
IR_DIR = os.path.join(PROJECT_ROOT, "ir")
OUTPUT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

SAMPLE_RATE = 48000
CHUNK_DURATION = 3.0
NUM_SAMPLES_TO_GENERATE = 5000

# Per la generazione parallela nei worker multiprocessing usiamo CPU per evitare crash CUDA
WORKER_DEVICE = torch.device("cpu")


def resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Ricampiona il segnale audio se il sample rate sorgente è diverso da target_sr."""
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio, up, down)


def parse_nam_metadata(nam_path: str, is_pedal: bool = False) -> dict:
    if not nam_path:
        return None

    filename = os.path.splitext(os.path.basename(nam_path))[0]
    capture_type = "DI"
    match_type = re.search(r'\[(.*?)\]', filename)
    if match_type:
        capture_type = match_type.group(1)
        filename = re.sub(r'\[.*?\]', '', filename).strip()

    parts = [p.strip() for p in filename.split(" - ") if p.strip()]
    brand = parts[0] if len(parts) > 0 else "UNKNOWN"
    model = parts[1] if len(parts) > 1 else "MODEL"
    raw_setting = parts[2] if len(parts) > 2 else "GENERAL"

    rel_path = os.path.relpath(nam_path, PEDALS_DIR) if (is_pedal and nam_path.startswith(PEDALS_DIR)) else ""
    
    # MODIFICA 1: Forziamo la categoria SEMPRE in maiuscolo per evitare fallimenti di matching
    category = "GENERAL" if (not rel_path or os.path.dirname(rel_path) == "") else os.path.dirname(rel_path).upper()

    knob_settings = parse_settings_to_dict(raw_setting)

    metadata = {
        "raw_file": os.path.relpath(nam_path, PROJECT_ROOT),
        "brand": brand,
        "model": model,
        "raw_setting": raw_setting,
        "settings": knob_settings
    }

    if is_pedal:
        metadata["category"] = category
    else:
        metadata["capture_type"] = capture_type

    return metadata


def parse_ir_metadata(ir_path: str) -> dict:
    if not ir_path:
        return None

    filename = os.path.splitext(os.path.basename(ir_path))[0]
    mic_or_notes = "STD"
    match_mic = re.search(r'\[(.*?)\]', filename)
    if match_mic:
        mic_or_notes = match_mic.group(1)
        filename = re.sub(r'\[.*?\]', '', filename).strip()

    parts = [p.strip() for p in filename.split(" - ") if p.strip()]
    brand = parts[0] if len(parts) > 0 else "GENERIC"
    cabinet = parts[1] if len(parts) > 1 else "CAB"

    return {
        "raw_file": os.path.relpath(ir_path, PROJECT_ROOT),
        "brand": brand,
        "cabinet": cabinet,
        "microphone_notes": mic_or_notes
    }


def parse_settings_to_dict(setting_str: str) -> dict:
    if not setting_str or setting_str == "GENERAL":
        return {}
    
    settings_dict = {}
    matches = re.findall(r'([A-Za-z]+)\s+(\d+(?:\.\d+)?)', setting_str)
    for param, val in matches:
        key = param.lower()
        val_float = float(val)
        settings_dict[key] = int(val_float) if val_float.is_integer() else val_float
        
    return settings_dict


def patch_nam_config_for_legacy(config: dict) -> dict:
    arch = config.get("architecture")
    if arch == "SlimmableContainer":
        submodels = config.get("config", {}).get("submodels", [])
        if not submodels:
            raise ValueError("Modello SlimmableContainer privo di 'submodels'.")
        
        submodel_entry = submodels[0]
        inner_model = submodel_entry.get("model", {})
        weights = (
            inner_model.get("weights") 
            or submodel_entry.get("weights") 
            or config.get("weights", [])
        )
        if not weights:
            raise ValueError("Array dei pesi vuoto in SlimmableContainer.")

        config = {
            "version": config.get("version", inner_model.get("version", "0.7.0")),
            "architecture": inner_model.get("architecture", "WaveNet"),
            "config": inner_model.get("config", {}),
            "weights": weights
        }
        arch = config.get("architecture")

    if arch == "WaveNet":
        net_config = config.get("config", {})
        layers = net_config.get("layers", [])
        for layer in layers:
            if "head" not in layer or layer["head"] is None:
                layer["head"] = {
                    "out_channels": layer.get("head_size", 1),
                    "kernel_size": 1,
                    "bias": layer.get("head_bias", True)
                }

    return config


def process_nam_native(audio_data: np.ndarray, nam_path: str) -> np.ndarray:
    try:
        with open(nam_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)

        config = patch_nam_config_for_legacy(config)
        model = init_from_nam(config)
        model.to(WORKER_DEVICE)
        model.eval()

        input_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).to(WORKER_DEVICE)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        if isinstance(output_tensor, torch.Tensor):
            output_data = output_tensor.squeeze().cpu().numpy()
        else:
            output_data = np.squeeze(output_tensor)

        if output_data.shape != audio_data.shape:
            output_data = np.pad(output_data, (0, max(0, len(audio_data) - len(output_data))))[:len(audio_data)]

        return output_data

    except Exception as e:
        print(f"\n[WARNING] Errore inferenza NAM ({os.path.basename(nam_path)}): {e}")
        return audio_data


def scan_pedals_by_category(pedals_dir: str) -> dict:
    categories = {}
    if not os.path.exists(pedals_dir):
        return categories

    for root, _, files in os.walk(pedals_dir):
        nam_files = [os.path.join(root, f) for f in files if f.lower().endswith(".nam")]
        if nam_files:
            rel_folder = os.path.relpath(root, pedals_dir)
            # MODIFICA 1: Categorie sempre ordinate e salvate in MAIUSCOLO
            cat_name = "GENERAL" if rel_folder == "." else rel_folder.upper()
            categories[cat_name] = nam_files

    return categories


def process_stage(audio: np.ndarray, category_keys: list, pedal_categories: dict, fallback_board: Pedalboard):
    nam_files = []
    for key in category_keys:
        key_upper = key.upper()
        if key_upper in pedal_categories:
            nam_files.extend(pedal_categories[key_upper])

    if nam_files:
        selected_nam = np.random.choice(nam_files)
        processed_audio = process_nam_native(audio, selected_nam)
        return processed_audio, {"nam_used": True, "nam_info": parse_nam_metadata(selected_nam, is_pedal=True)}
    elif fallback_board is not None:
        processed_audio = fallback_board(np.expand_dims(audio, axis=0), SAMPLE_RATE).squeeze()
        return processed_audio, {"nam_used": False, "nam_info": None}
    
    return audio, {"nam_used": False, "nam_info": None}


def generate_single_sample(i, clean_files, nam_amp_files, pedal_categories, ir_files, audio_dir, chunk_samples):
    """Worker per la generazione della catena di effetti reali sequenziali."""
    audio_clean = None
    random_clean_file = None
    while audio_clean is None:
        random_clean_file = np.random.choice(clean_files)
        try:
            audio_clean, sr = sf.read(random_clean_file)
            # MODIFICA 4: Controllo e resample se il file sorgente ha SR diverso
            audio_clean = resample_if_needed(audio_clean, sr, SAMPLE_RATE)
        except Exception:
            pass

    if audio_clean.ndim > 1:
        audio_clean = np.mean(audio_clean, axis=1)

    if len(audio_clean) < chunk_samples:
        repeats = int(np.ceil(chunk_samples / len(audio_clean)))
        audio_clean = np.tile(audio_clean, repeats)[:chunk_samples]
    else:
        max_start = len(audio_clean) - chunk_samples
        start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        audio_clean = audio_clean[start_idx : start_idx + chunk_samples]

    # Probabilità di attivazione blocchi
    hp_active = bool(np.random.rand() < 0.40)
    comp_active = bool(np.random.rand() < 0.40)
    drive_active = bool(np.random.rand() < 0.50)
    chorus_active = bool(np.random.rand() < 0.40)
    delay_active = bool(np.random.rand() < 0.40)
    reverb_active = bool(np.random.rand() < 0.40)

    # Parametri casuali per eventuale fallback DSP
    cutoff_freq = float(np.random.uniform(80.0, 350.0))
    drive_db = float(np.random.uniform(6.0, 30.0))
    
    chorus_rate = float(np.random.uniform(0.5, 2.2))
    chorus_depth = float(np.random.uniform(0.15, 0.50))
    chorus_delay = float(np.random.uniform(7.0, 20.0))
    chorus_feedback = float(np.random.uniform(0.10, 0.40))
    chorus_mix = float(np.random.uniform(0.20, 0.50))
    
    delay_time = float(np.random.uniform(0.18, 0.55))
    delay_feedback = float(np.random.uniform(0.15, 0.45))
    delay_mix = float(np.random.uniform(0.15, 0.45))
    
    reverb_room = float(np.random.uniform(0.2, 0.8))
    reverb_damping = float(np.random.uniform(0.2, 0.7))
    reverb_wet = float(np.random.uniform(0.15, 0.50))
    reverb_dry = float(np.random.uniform(0.8, 1.0))
    reverb_width = float(np.random.uniform(0.5, 1.0))

    audio_processing = audio_clean.copy()
    stage_logs = {}

    # 1. HIGHPASS FILTER
    if hp_active:
        board_hp = Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff_freq)])
        audio_processing, meta = process_stage(audio_processing, ["HIGHPASS"], pedal_categories, board_hp)
        dsp_params = {"cutoff_frequency_hz": cutoff_freq} if not meta["nam_used"] else {"cutoff_frequency_hz": None}
        stage_logs["highpass"] = {"enabled": True, **dsp_params, **meta}
    else:
        stage_logs["highpass"] = {"enabled": False}

    # 2. COMPRESSOR
    if comp_active:
        audio_processing, meta = process_stage(audio_processing, ["COMPRESSOR", "COMP"], pedal_categories, None)
        stage_logs["compressor"] = {"enabled": True, **meta}
    else:
        stage_logs["compressor"] = {"enabled": False}

    # 3. DRIVE / SATURATION
    if drive_active:
        board_dist = Pedalboard([Distortion(drive_db=drive_db)])
        audio_processing, meta = process_stage(audio_processing, ["OVERDRIVE", "DISTORTION", "FUZZ", "DRIVE", "GENERAL"], pedal_categories, board_dist)
        dsp_params = {"drive_db": drive_db} if not meta["nam_used"] else {"drive_db": None}
        stage_logs["drive"] = {"enabled": True, **dsp_params, **meta}
    else:
        stage_logs["drive"] = {"enabled": False}

    # STAGE AMPLIFICATORE NAM
    selected_amp_nam = None
    if nam_amp_files:
        selected_amp_nam = np.random.choice(nam_amp_files)
        audio_processing = process_nam_native(audio_processing, selected_amp_nam)

    # STAGE IR CABINET
    selected_ir_file = None
    if ir_files:
        selected_ir_file = np.random.choice(ir_files)
        try:
            ir_audio, ir_sr = sf.read(selected_ir_file)
            ir_audio = resample_if_needed(ir_audio, ir_sr, SAMPLE_RATE)
            if ir_audio.ndim > 1:
                ir_audio = np.mean(ir_audio, axis=1)
            audio_processing = fftconvolve(audio_processing, ir_audio, mode="full")[: len(audio_processing)]
        except Exception:
            pass

    # 4. CHORUS
    if chorus_active:
        board_chorus = Pedalboard([Chorus(rate_hz=chorus_rate, depth=chorus_depth, centre_delay_ms=chorus_delay, feedback=chorus_feedback, mix=chorus_mix)])
        audio_processing, meta = process_stage(audio_processing, ["CHORUS", "MODULATION"], pedal_categories, board_chorus)
        dsp_params = {
            "rate_hz": chorus_rate,
            "depth": chorus_depth,
            "centre_delay_ms": chorus_delay,
            "feedback": chorus_feedback,
            "mix": chorus_mix
        } if not meta["nam_used"] else {
            "rate_hz": None, "depth": None, "centre_delay_ms": None, "feedback": None, "mix": None
        }
        stage_logs["chorus"] = {"enabled": True, **dsp_params, **meta}
    else:
        stage_logs["chorus"] = {"enabled": False}

    # 5. DELAY
    if delay_active:
        board_delay = Pedalboard([Delay(delay_seconds=delay_time, feedback=delay_feedback, mix=delay_mix)])
        audio_processing, meta = process_stage(audio_processing, ["DELAY"], pedal_categories, board_delay)
        dsp_params = {
            "delay_seconds": delay_time,
            "feedback": delay_feedback,
            "mix": delay_mix
        } if not meta["nam_used"] else {
            "delay_seconds": None, "feedback": None, "mix": None
        }
        stage_logs["delay"] = {"enabled": True, **dsp_params, **meta}
    else:
        stage_logs["delay"] = {"enabled": False}

    # 6. REVERB
    if reverb_active:
        board_reverb = Pedalboard([Reverb(room_size=reverb_room, damping=reverb_damping, wet_level=reverb_wet, dry_level=reverb_dry, width=reverb_width)])
        audio_processing, meta = process_stage(audio_processing, ["REVERB"], pedal_categories, board_reverb)
        dsp_params = {
            "room_size": reverb_room,
            "damping": reverb_damping,
            "wet_level": reverb_wet,
            "dry_level": reverb_dry,
            "width": reverb_width
        } if not meta["nam_used"] else {
            "room_size": None, "damping": None, "wet_level": None, "dry_level": None, "width": None
        }
        stage_logs["reverb"] = {"enabled": True, **dsp_params, **meta}
    else:
        stage_logs["reverb"] = {"enabled": False}

    # Limiter Finale e Normalizzazione
    limiter = Pedalboard([Limiter(threshold_db=-1.0)])
    audio_final = limiter(np.expand_dims(audio_processing, axis=0), SAMPLE_RATE).squeeze()

    # Sanitizzazione valori NaN/Inf post-convoluzione
    audio_final = np.nan_to_num(audio_final, nan=0.0, posinf=0.9, neginf=-0.9)

    max_val = np.max(np.abs(audio_final))
    if max_val > 0:
        audio_final = audio_final / max_val * 0.90

    filename = f"sample_{i:05d}.wav"
    abs_audio_path = os.path.join(audio_dir, filename)
    sf.write(abs_audio_path, audio_final, SAMPLE_RATE)

    rel_audio_path = os.path.join("audio", filename)

    return {
        "audio_file": rel_audio_path,
        "source_file": os.path.relpath(random_clean_file, PROJECT_ROOT),
        "amplifier": parse_nam_metadata(selected_amp_nam, is_pedal=False),
        "cabinet_ir": parse_ir_metadata(selected_ir_file),
        "signal_chain": stage_logs
    }


def generate_audio_dataset(num_samples=NUM_SAMPLES_TO_GENERATE, dataset_dir=OUTPUT_DATASET_DIR, max_workers=4):
    audio_dir = os.path.join(dataset_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(IDMT_DATASET_DIR, "**", "*"), recursive=True)
    clean_files = [
        f for f in all_files 
        if f.lower().endswith(".wav") 
        and "acoustic_mic" not in f.lower() 
        and "acoustic_pickup" not in f.lower()
    ]

    if not clean_files:
        raise FileNotFoundError(f"Nessun file .wav di chitarra elettrica trovato in {IDMT_DATASET_DIR}")

    nam_amp_files = [f for f in glob.glob(os.path.join(AMP_HEADS_DIR, "**", "*"), recursive=True) if f.lower().endswith(".nam")]
    pedal_categories = scan_pedals_by_category(PEDALS_DIR)
    ir_files = [f for f in glob.glob(os.path.join(IR_DIR, "**", "*"), recursive=True) if f.lower().endswith(".wav")]

    workers = max_workers if max_workers is not None else min(4, os.cpu_count() or 1)

    print(f"--- Avvio Generazione Audio Parallela ({workers} worker su CPU) ---")
    print(f"Sorgenti audio clean (Elettriche): {len(clean_files)}")
    print(f"Modelli NAM Amplificatori: {len(nam_amp_files)}")
    print(f"Categorie Pedali NAM trovate: {list(pedal_categories.keys())}")
    for cat, files in pedal_categories.items():
        print(f"  - {cat}: {len(files)} modelli")
    print(f"File IR Cabinet: {len(ir_files)}")

    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    worker_fn = partial(
        generate_single_sample,
        clean_files=clean_files,
        nam_amp_files=nam_amp_files,
        pedal_categories=pedal_categories,
        ir_files=ir_files,
        audio_dir=audio_dir,
        chunk_samples=chunk_samples
    )

    metadata = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # MODIFICA 2: Aggiunto chunksize=10 per abbattere l'overhead multiprocessing
        results = executor.map(worker_fn, range(num_samples), chunksize=10)
        for idx, meta in enumerate(results, 1):
            metadata.append(meta)
            if idx % 100 == 0 or idx == num_samples:
                print(f"Progresso audio: {idx}/{num_samples} campioni generati.")

    json_path = os.path.join(dataset_dir, "dataset_labels.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Generazione audio e metadati completata con successo!\n")


if __name__ == "__main__":
    generate_audio_dataset(num_samples=NUM_SAMPLES_TO_GENERATE, max_workers=4)