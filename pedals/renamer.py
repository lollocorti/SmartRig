import os
import re
import shutil

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

BRAND_MAP = {
    "HORIZON DEVICES": "HORIZON DEVICES", "HORIZON": "HORIZON DEVICES",
    "IBANEZ": "IBANEZ", "TS": "IBANEZ", 
    "BOSS": "BOSS", 
    "PROCO": "PROCO", "RAT": "PROCO", 
    "ELECTRO HARMONIX": "ELECTRO HARMONIX", "EHX": "ELECTRO HARMONIX", 
    "KLON": "KLON", 
    "BEHRINGER": "BEHRINGER",
    "XOTIC": "XOTIC", "SP": "XOTIC",
    "KEELEY": "KEELEY", 
    "MXR": "MXR", 
    "STRYMON": "STRYMON", 
    "TECH 21": "TECH 21", "SANSAMP": "TECH 21", 
    "DALLAS": "DALLAS",
    "DUNLOP": "DUNLOP", "ARBITER": "DUNLOP",
    "FULLTONE": "FULLTONE",
    "DOD": "DOD",
    "B.K. BUTLER": "BK BUTLER", "BK BUTLER": "BK BUTLER", "BUTLER": "BK BUTLER"
}

MODEL_MAP = {
    "PRECISION DRIVE": "PRECISION DRIVE", "PRECISION": "PRECISION DRIVE",
    "TS808": "TS808", "TS9": "TS9", "TUBE SCREAMER": "TS9", 
    "CENTAUR": "CENTAUR", "KLON": "CENTAUR", 
    "EP BOOSTER": "EP BOOSTER", "EPBOOSTER": "EP BOOSTER",
    "SP COMPRESSOR": "SP COMPRESSOR", "SPCOMPRESSOR": "SP COMPRESSOR", "SP COMP": "SP COMPRESSOR", "SPCOMP": "SP COMPRESSOR",
    "BB PREAMP": "BB PREAMP", "RANGEMASTER": "RANGEMASTER", "BOOST": "RANGEMASTER", 
    "RAT": "RAT", "BIG MUFF": "BIG MUFF", "DS-1": "DS1", "DS1": "DS1", "DS 1": "DS1", 
    "FUZZ FACE": "FUZZ FACE", "FUZZFACE": "FUZZ FACE",
    "DYNA COMP": "DYNA COMP", "COMPRESSOR PLUS": "COMPRESSOR PLUS", 
    "CE2": "CE2", "CARBON COPY": "CARBON COPY", "SANSAMP BASS DRIVER": "SANSAMP BASS DRIVER",
    "OCD": "OCD",
    "250": "OVERDRIVE PREAMP 250", "OVERDRIVE PREAMP 250": "OVERDRIVE PREAMP 250",
    "TUBE DRIVER": "TUBE DRIVER", "TUBEDRIVER": "TUBE DRIVER",
    "TIMMY": "TIMMY"
}

KNOB_MAP = {
    "G": "GAIN", "GN": "GAIN", "GAIN": "GAIN",
    "D": "DRIVE", "DRV": "DRIVE", "DRIVE": "DRIVE", "DIST": "DRIVE",
    "F": "FUZZ", "FZ": "FUZZ", "FUZZ": "FUZZ",
    "FLT": "FILTER", "FILTER": "FILTER",
    "T": "TREBLE", "TRB": "TREBLE", "TREBLE": "TREBLE",
    "TONE": "TONE", "TN": "TONE",
    "B": "BASS", "BS": "BASS", "BASS": "BASS",
    "V": "VOLUME", "VOL": "VOLUME", "VOLUME": "VOLUME",
    "L": "LEVEL", "LVL": "LEVEL", "LEVEL": "LEVEL",
    "S": "SUSTAIN", "SUST": "SUSTAIN", "SUSTAIN": "SUSTAIN",
    "A": "ATTACK", "AT": "ATTACK", "ATK": "ATTACK", "ATTK": "ATTACK", "ATTACK": "ATTACK",
    "BRT": "BRIGHT", "BRIGHT": "BRIGHT",
    "GATE": "GATE", "GT": "GATE",
    "BL": "BLEND", "BLND": "BLEND", "BLEND": "BLEND",
    "M": "MIX", "MIX": "MIX",
    "FB": "FEEDBACK", "FEEDBACK": "FEEDBACK",
    "DLY": "DELAY", "DELAY": "DELAY",
    "DEP": "DEPTH", "DEPTH": "DEPTH",
    "RT": "RATE", "RATE": "RATE"
}

NOISE_TOKENS = {
    "PEDAL", "NAM", "DI", "DEFAULT", "UNKNOWN", "AMP", 
    "DISTORTION", "OVERDRIVE", "COMPRESSOR", "DELAY", 
    "GENERIC", "VINTAGE", "REISSUE", "SS", "JM", "LED", "SLAMMIN", 
    "MAIN", "S", "CLONE", "SI", "A-A", "AA", "V2", "ON",
    "NUNO", "BEATO", "DRUMS", "IGNORED", "RI", "MANTOVANI", "MOD",
    "LIVESPICE", "LIVE", "SPICE"
}

SWITCH_MAP = {
    "HIGH": "GAIN 8.5",
    "HI": "GAIN 8.5",
    "MID": "GAIN 5",
    "MED": "GAIN 5",
    "LOW": "GAIN 2.5",
    "LO": "GAIN 2.5",
    "ERIC": "GAIN 6.5"
}


def oclock_to_scale_10(hh: int, mm: int) -> str:
    if 1 <= hh <= 5:
        hh += 12

    hours_decimal = hh + (mm / 60.0)
    val = max(0.0, min(10.0, hours_decimal - 7.0))
    
    if val.is_integer():
        return str(int(val))
    return str(round(val, 1))


def parse_knob_settings(text: str, is_rat=False, default_knob="GAIN") -> str:
    current_knob_map = KNOB_MAP.copy()
    if is_rat:
        current_knob_map["F"] = "FILTER"

    # Rimozione preventiva del token di rumore "TTSV10"
    text = re.sub(r'\bTTSV\d+\b', ' ', text, flags=re.IGNORECASE)

    # Pulizia dei token dello switch di clipping del Timmy (C-L, C-C, C-R, ecc.)
    text = re.sub(r'\bC[\-_][LCR]\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bC[LCR]\b', ' ', text, flags=re.IGNORECASE)

    local_noise = NOISE_TOKENS.union({"SS", "JM", "NUNO", "BEATO", "DRUMS", "IGNORED"})
    expanded_tokens = []

    # Check ed estrazione dello switch generale di gain/mode
    found_switch = None
    for sw_key, sw_val in SWITCH_MAP.items():
        pattern = r'\b' + re.escape(sw_key) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_switch = sw_val
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
            break

    # Normalizza token tipo T-Max in T 10
    text = re.sub(r'\b([a-zA-Z]+)[\-_]MAX\b', r'\1 10', text, flags=re.IGNORECASE)

    # 1. Pattern orari espliciti
    time_matches = re.finditer(r'\b([a-zA-Z]+)[\s\-_]*(\d{1,2})[\_\:\.](\d{2})\b', text)
    for match in time_matches:
        prefix, hh, mm = match.groups()
        p_upper = prefix.upper()
        hh_int, mm_int = int(hh), int(mm)
        
        if p_upper in current_knob_map and mm_int < 60 and 1 <= hh_int <= 12:
            label = current_knob_map[p_upper]
            scaled_val = oclock_to_scale_10(hh_int, mm_int)
            expanded_tokens.extend([label, scaled_val])
            text = text.replace(match.group(0), ' ')

    text = re.sub(r'([a-zA-Z]+)[\-_](\d+)[\_](\d+)', r'\1 \2.\3', text)

    # 2. Pulizia e unpacking sequenze compatte
    cleaned_text = re.sub(r'[\-_]+', ' ', text)
    
    def unpack_compact(match_obj):
        p, v = match_obj.groups()
        return f" {p} {v} "

    cleaned_text = re.sub(r'([a-zA-Z]+)(\d+(?:\.\d+)?)', unpack_compact, cleaned_text)
    tokens = cleaned_text.split()

    for token in tokens:
        token_upper = token.upper()
        if token_upper in local_noise:
            continue
        label = current_knob_map.get(token_upper, token_upper)
        expanded_tokens.append(label)

    # 3. Associazione coppie
    parsed_pairs = []
    i = 0
    while i < len(expanded_tokens):
        tok = expanded_tokens[i]

        if tok in current_knob_map.values():
            if i + 1 < len(expanded_tokens):
                next_tok = expanded_tokens[i + 1]
                if re.match(r'^\d+(\.\d+)?$', next_tok):
                    parsed_pairs.append(f"{tok} {next_tok}")
                    i += 2
                    continue
            i += 1
        elif re.match(r'^\d+(\.\d+)?$', tok):
            parsed_pairs.append(f"{default_knob} {tok}")
            i += 1
        else:
            i += 1

    if found_switch:
        parsed_pairs.append(found_switch)

    return " ".join(parsed_pairs)


def parse_and_standardize_pedal(filename, subfolder_category):
    name, ext = os.path.splitext(filename)
    if ext.lower() != '.nam':
        return None

    cleaned = re.sub(r'\[.*?\]', '', name)
    
    # Separazione automatica CamelCase / trattini
    cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # A. Modello
    model_found = None
    ds1_match = re.search(r'\bDS[\s\-]?1\b', cleaned, re.IGNORECASE)
    if ds1_match:
        model_found = "DS1"
        cleaned = re.sub(r'\bDS[\s\-]?1\b', ' ', cleaned, flags=re.IGNORECASE)
    else:
        for key in sorted(MODEL_MAP.keys(), key=len, reverse=True):
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, cleaned, re.IGNORECASE):
                model_found = MODEL_MAP[key]
                cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
                break

    # B. Marca
    brand_found = None
    for key in sorted(BRAND_MAP.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, cleaned, re.IGNORECASE):
            brand_found = BRAND_MAP[key]
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
            break

    if not brand_found and model_found:
        if model_found in ["TS9", "TS808"]:
            brand_found = "IBANEZ"
        elif model_found == "RAT":
            brand_found = "PROCO"
        elif model_found in ["DS1", "CE2"]:
            brand_found = "BOSS"
        elif model_found == "BIG MUFF":
            brand_found = "ELECTRO HARMONIX"
        elif model_found == "FUZZ FACE":
            brand_found = "DUNLOP"
        elif model_found == "CENTAUR":
            brand_found = "KLON"
        elif model_found == "RANGEMASTER":
            brand_found = "DALLAS"
        elif model_found == "OCD":
            brand_found = "FULLTONE"
        elif model_found in ["EP BOOSTER", "SP COMPRESSOR"]:
            brand_found = "XOTIC"
        elif model_found == "OVERDRIVE PREAMP 250":
            brand_found = "DOD"
        elif model_found == "TUBE DRIVER":
            brand_found = "BK BUTLER"
        elif model_found == "TIMMY":
            brand_found = "MXR"
        elif model_found == "PRECISION DRIVE":
            brand_found = "HORIZON DEVICES"

    brand_str = brand_found if brand_found else "GENERIC"
    model_str = model_found if model_found else subfolder_category.upper()

    # C. Manopole
    is_rat = (model_str == "RAT")
    setting_str = parse_knob_settings(cleaned, is_rat=is_rat, default_knob="GAIN")

    if not setting_str:
        setting_str = "GENERAL"

    return f"{brand_str} - {model_str} - {setting_str}{ext.upper()}"


def run_rename_pedals():
    pending_renames = []

    for root, _, files in os.walk(FOLDER_PATH):
        subfolder_category = os.path.basename(root)
        for filename in files:
            old_path = os.path.join(root, filename)
            if os.path.isfile(old_path) and not filename.endswith(".py"):
                new_name = parse_and_standardize_pedal(filename, subfolder_category)
                
                if new_name and new_name != filename:
                    new_path = os.path.join(root, new_name)

                    if old_path.strip().upper() == new_path.strip().upper():
                        continue

                    pending_renames.append({
                        "category": subfolder_category.upper(),
                        "old_filename": filename,
                        "old_path": old_path,
                        "new_name": new_name,
                        "new_path": new_path
                    })

    if not pending_renames:
        print("\n[INFO] Nessun file da rinominare trovato.")
        return

    print("\n" + "=" * 70)
    print("                    ANTEPRIMA RINOMINAZIONE FILE                     ")
    print("=" * 70)
    for idx, item in enumerate(pending_renames, 1):
        print(f"[{idx}] [{item['category']}]")
        print(f"    ORIGINALE : {item['old_filename']}")
        print(f"    NUOVO     : {item['new_name']}")
        print("-" * 70)

    print(f"\nTotale file da rinominare: {len(pending_renames)}\n")

    confirm = input("Vuoi applicare queste modifiche su disco? (s/n): ").strip().lower()

    if confirm in ["s", "si", "y", "yes"]:
        renamed_count = 0
        for item in pending_renames:
            shutil.move(item["old_path"], item["new_path"])
            renamed_count += 1
        print(f"\n[OK] Operazione completata! {renamed_count} file rinominati con successo.")
    else:
        print("\n[ANNULLATO] Operazione annullata. Nessun file è stato modificato su disco.")


if __name__ == "__main__":
    run_rename_pedals()