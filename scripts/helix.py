import glob
import json
import os
import re
import numpy as np


def extract_preset_number(filepath: str) -> int:
    """Estrae l'indice numerico da 'preset12.hlx' per consentire un ordinamento naturale."""
    filename = os.path.basename(filepath)
    match = re.search(r'preset(\d+)\.hlx', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 999999


def load_helix_presets(presets_dir: str) -> list:
    """Carica in modo ordinato tutti i file .hlx validi presenti nella directory."""
    preset_files = glob.glob(os.path.join(presets_dir, "*.hlx"))
    
    # Ordina i preset in base al loro indice sequenziale (preset1, preset2, ...)
    preset_files.sort(key=extract_preset_number)

    loaded_presets = []

    for filepath in preset_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Verifica basica della struttura Helix
                if isinstance(data, dict) and "data" in data and "tone" in data["data"]:
                    loaded_presets.append({
                        "filename": os.path.basename(filepath),
                        "preset_data": data
                    })
        except Exception as e:
            print(f"[WARN] Impossibile leggere il preset {os.path.basename(filepath)}: {e}")

    return loaded_presets


def parse_and_randomize_preset_blocks(preset_obj: dict) -> dict:
    """Estrae la catena di blocchi da un preset ed esegue la randomizzazione dei parametri."""
    tone_data = preset_obj.get("data", {}).get("tone", {})
    preset_name = preset_obj.get("data", {}).get("meta", {}).get("name", "Unknown")

    chain_config = {
        "preset_name": str(preset_name),
        "blocks": {}
    }

    for dsp_key in ["dsp0", "dsp1"]:
        dsp_data = tone_data.get(dsp_key, {})
        if not isinstance(dsp_data, dict):
            continue

        for block_key, block_val in dsp_data.items():
            if not isinstance(block_val, dict) or not block_key.startswith("block"):
                continue

            model_name = str(block_val.get("@model", "Unknown"))
            is_enabled = bool(block_val.get("@enabled", True))
            
            # Randomizzazione controllata dello stato ON/OFF (salvaguardando il blocco finale)
            if not block_key.endswith("2"): 
                is_enabled = bool(np.random.choice([True, False], p=[0.75, 0.25]))

            block_params = {}
            for k, v in block_val.items():
                if k.startswith("@"):
                    continue
                
                # Perturbazione dei parametri numerici (+/- 15%)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    variation = float(np.random.uniform(-0.15, 0.15))
                    new_val = float(v + variation)
                    
                    if 0.0 <= v <= 1.0:
                        new_val = float(np.clip(new_val, 0.0, 1.0))
                    else:
                        new_val = max(0.0, new_val)
                    block_params[str(k)] = round(new_val, 4)
                elif isinstance(v, bool):
                    block_params[str(k)] = bool(np.random.choice([v, not v], p=[0.8, 0.2]))

            chain_config["blocks"][f"{dsp_key}_{block_key}"] = {
                "model": model_name,
                "enabled": is_enabled,
                "parameters": block_params
            }

    return chain_config