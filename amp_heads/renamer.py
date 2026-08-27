import os
import re
from abc import ABC, abstractmethod

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. CLASSE BASE PER LA GESTIONE DEI FILE NAM
# ==============================================================================

class NAMFile:
    """Rappresenta un file .nam e ne gestisce la pulizia di base."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.name_no_ext, self.ext = os.path.splitext(self.filename)
        self.upper_name = self.name_no_ext.upper()
        self.capture_type = self._detect_capture_type()

    def _detect_capture_type(self):
        if any(tag in self.upper_name for tag in ["[CAB]", " CAB", "CAPTOR"]):
            if "DI" not in self.upper_name:
                return "CAB"
        return "DI"

    def clean_text(self):
        """Pulisce parentesi, revisioni finali e simboli rumorosi."""
        cleaned = re.sub(r'\[.*?\]', '', self.upper_name)
        cleaned = re.sub(r'_\d+$', '', cleaned)
        cleaned = re.sub(r"['\-_,]+", ' ', cleaned)
        return re.sub(r'\s+', ' ', cleaned).strip()

# ==============================================================================
# 2. STRATEGIE DI PARSING PER BRAND (PLUGINS)
# ==============================================================================

class BaseParser(ABC):
    @abstractmethod
    def match(self, nam_file: NAMFile) -> bool:
        """Ritorna True se questo parser deve gestire il file."""
        pass

    @abstractmethod
    def parse(self, nam_file: NAMFile) -> str:
        """Ritorna il nome formattato del file."""
        pass


class HiwattParser(BaseParser):
    def match(self, nam_file: NAMFile) -> bool:
        return any(k in nam_file.upper_name for k in ["HIWATT", "DR103"])

    def _convert_clock(self, val_str):
        val_str = val_str.replace(',', '.')
        try:
            val = float(val_str)
            if 7 <= val <= 13:
                num = (val - 7) * 2
                return str(int(num)) if num.is_integer() else f"{num:.1f}"
            return str(int(val)) if val.is_integer() else f"{val:.1f}"
        except ValueError:
            return val_str

    def parse(self, nam_file: NAMFile) -> str:
        text = nam_file.upper_name

        if re.match(r'^HIWATT\s*-\s*DR103\s*-\s*.*\[(DI|CAB)\]\.(NAM|nam)$', nam_file.filename):
            return nam_file.filename

        cleaned = re.sub(r'\b(HIWATT|DR103|2025|2026|REV|REVYHI|CAPTOR)\b', '', nam_file.clean_text())

        labeled_matches = re.findall(r'\b(NV|BV|V|B|M|T|P|MV)\s*(\d+(?:[\.,]\d+)?)\b', cleaned)
        if labeled_matches and len(labeled_matches) >= 5:
            formatted_parts = [f"{lbl}{self._convert_clock(val)}" for lbl, val in labeled_matches]
            setting_str = " ".join(formatted_parts)
        else:
            pattern_seq = r'\b(\d{1,2}(?:[\.,]\d+)?)[- ](\d{1,2}(?:[\.,]\d+)?)[- ](\d{1,2}(?:[\.,]\d+)?)[- ](\d{1,2}(?:[\.,]\d+)?)[- ](\d{1,2}(?:[\.,]\d+)?)[- ](\d{1,2}(?:[\.,]\d+)?)(?:[- ](\d{1,2}(?:[\.,]\d+)?))?\b'
            match_seq = re.search(pattern_seq, cleaned)

            if match_seq:
                vals = [self._convert_clock(v) for v in match_seq.groups() if v is not None]
                labels = ["NV", "BV", "B", "M", "T", "P", "MV"] if len(vals) == 7 else ["V", "B", "M", "T", "P", "MV"]
                setting_str = " ".join(f"{lbl}{val}" for lbl, val in zip(labels, vals))
            else:
                master = re.search(r'\bMASTER\s*(\d+(?:\.\d+)?)\b', cleaned)
                parts = []
                if "ALL 12" in text or "ALL 5" in text:
                    parts.append("ALL 5")
                if master:
                    parts.append(f"MV{self._convert_clock(master.group(1))}")
                setting_str = " ".join(parts) if parts else "DEFAULT"

        return f"HIWATT - DR103 - {setting_str} [{nam_file.capture_type}]{nam_file.ext.upper()}"


class RolandParser(BaseParser):
    def match(self, nam_file: NAMFile) -> bool:
        return any(k in nam_file.upper_name for k in ["ROLAND", "JC 120", "JC120"])

    def parse(self, nam_file: NAMFile) -> str:
        if re.match(r'^ROLAND\s*-\s*JC120\s*-\s*.*\[(DI|CAB)\]\.(NAM|nam)$', nam_file.filename):
            return nam_file.filename

        ch = "CH2" if "CH 2" in nam_file.upper_name or "CH2" in nam_file.upper_name else "CH1"
        inp = "LOW" if "LOW" in nam_file.upper_name else ("HI" if "HI" in nam_file.upper_name else "")
        setting_str = f"{ch} {inp}".strip()
        return f"ROLAND - JC120 - {setting_str} [{nam_file.capture_type}]{nam_file.ext.upper()}"


class VoxAC15Parser(BaseParser):
    def match(self, nam_file: NAMFile) -> bool:
        return "AC15" in nam_file.upper_name

    def _clock_to_num(self, time_str: str) -> str:
        time_str = time_str.lower().strip()
        mapping = {
            "7am": "0", "8am": "1", "9am": "2", "10am": "3", 
            "11am": "4", "noon": "5", "12pm": "5", "1pm": "6", 
            "2pm": "7", "3pm": "8", "4pm": "9", "5pm": "10", "max": "10"
        }
        return mapping.get(time_str, time_str.upper())

    def parse(self, nam_file: NAMFile) -> str:
        if re.match(r'^VOX\s*-\s*AC15C1\s*-\s*.*\[(DI|CAB)\]\.(NAM|nam)$', nam_file.filename):
            return nam_file.filename

        clean_name = nam_file.clean_text()

        # 1. Canale
        channel = "NRM" if "NRM" in clean_name else ("TB" if "TB" in clean_name else "")

        # 2. Stato di default per tutti (V5, T5, B5)
        settings = {"V": "5", "T": "5", "B": "5"}

        # 3. Parsing modifiche esplicite
        setting_matches = re.findall(
            r'\b(VOL|VOLUME|TREBLE|BASS|REV)\s*(MAX|\d{1,2}(?:AM|PM)|NOON)\b', 
            clean_name
        )
        
        for ctrl, val in setting_matches:
            num_val = self._clock_to_num(val)
            if "VOL" in ctrl:
                settings["V"] = num_val
            elif "TREBLE" in ctrl:
                settings["T"] = num_val
            elif "BASS" in ctrl:
                settings["B"] = num_val
            elif "REV" in ctrl:
                settings["REV"] = num_val

        # 4. Assemblaggio
        parts = []
        if channel:
            parts.append(channel)

        for k, v in settings.items():
            parts.append(f"{k}{v}")

        if "SM57" in clean_name:
            parts.append("SM57")
        if re.search(r'\bV2\b', clean_name):
            parts.append("V2")

        capture_type = "CAB" if "SM57" in nam_file.upper_name else nam_file.capture_type
        setting_str = " ".join(parts)
        
        return f"VOX - AC15C1 - {setting_str} [{capture_type}]{nam_file.ext.upper()}"


class VoxAC30Parser(BaseParser):
    def match(self, nam_file: NAMFile) -> bool:
        return "AC30" in nam_file.upper_name

    def parse(self, nam_file: NAMFile) -> str:
        if re.match(r'^VOX\s*-\s*AC30\s*TB\s*-\s*.*\[(DI|CAB)\]\.(NAM|nam)$', nam_file.filename):
            return nam_file.filename

        clean_name = nam_file.clean_text()

        v_match = re.search(r'\bV(\d+)\b', clean_name)
        tc_match = re.search(r'\bTC(\d+)\b', clean_name)
        b_match = re.search(r'\bB(\d+)\b', clean_name)
        t_match = re.search(r'\bT(\d+)\b', clean_name)

        params = []
        if v_match: params.append(f"V{v_match.group(1)}")
        if tc_match: params.append(f"TC{tc_match.group(1)}")
        if b_match: params.append(f"B{b_match.group(1)}")
        if t_match: params.append(f"T{t_match.group(1)}")

        setting_str = " ".join(params) if params else "DEFAULT"
        return f"VOX - AC30 TB - {setting_str} [{nam_file.capture_type}]{nam_file.ext.upper()}"


class OrangeRockerverbParser(BaseParser):
    def match(self, nam_file: NAMFile) -> bool:
        return "ROCKERVERB" in nam_file.upper_name

    def parse(self, nam_file: NAMFile) -> str:
        if re.match(r'^ORANGE\s*-\s*ROCKERVERB\s*MK3\s*-\s*G\d+\s*T\d+\s*B\d+\s*\[(DI|CAB)\]\.(NAM|nam)$', nam_file.filename):
            return nam_file.filename

        clean_name = nam_file.clean_text()

        gain_match = re.search(r'\bG(\d+)\b', clean_name)
        if gain_match:
            gain_str = f"G{gain_match.group(1)}"
        elif "CLEAN" in clean_name:
            gain_str = "G0"
        else:
            gain_str = "G5"

        settings = f"{gain_str} T5 B5"
        return f"ORANGE - ROCKERVERB MK3 - {settings} [{nam_file.capture_type}]{nam_file.ext.upper()}"


class FenderPrincetonParser(BaseParser):
    def match(self, nam_file: NAMFile) -> bool:
        return any(k in nam_file.upper_name for k in ["PRINCETON", "64 PRINCETON"])

    def parse(self, nam_file: NAMFile) -> str:
        if re.match(r'^FENDER\s*-\s*64\s*PRINCETON\s*-\s*.*\[(DI|CAB)\]\.(NAM|nam)$', nam_file.filename):
            return nam_file.filename

        clean_name = nam_file.clean_text()

        vol_match = re.search(r'\bVOL(?:UME)?\s*(\d+(?:[\.,]\d+)?)\b', clean_name)
        tone_match = re.search(r'\bTONE\s*(\d+(?:[\.,]\d+)?)\b', clean_name)

        params = []
        if vol_match: params.append(f"V{vol_match.group(1)}")
        if tone_match: params.append(f"T{tone_match.group(1)}")

        setting_str = " ".join(params) if params else "DEFAULT"
        return f"FENDER - 64 PRINCETON - {setting_str} [{nam_file.capture_type}]{nam_file.ext.upper()}"


class SilvertoneParser(BaseParser):
    def match(self, nam_file: NAMFile) -> bool:
        return "SILVERTONE" in nam_file.upper_name or "1484" in nam_file.upper_name

    def parse(self, nam_file: NAMFile) -> str:
        if re.match(r'^SILVERTONE\s*-\s*1484\s*-\s*.*\[(DI|CAB)\]\.(NAM|nam)$', nam_file.filename):
            return nam_file.filename

        clean_name = nam_file.clean_text()

        ch_match = re.search(r'\b(CH\d+)\b', clean_name)
        ch_str = f"{ch_match.group(1)} " if ch_match else ""

        v_match = re.search(r'\bV(\d+)\b', clean_name)
        t_match = re.search(r'\bT(\d+)\b', clean_name)
        b_match = re.search(r'\bB(\d+)\b', clean_name)

        v_val = f"V{v_match.group(1)}" if v_match else "V5"
        t_val = f"T{t_match.group(1)}" if t_match else "T5"
        b_val = f"B{b_match.group(1)}" if b_match else "B5"

        settings = f"{ch_str}{v_val} {t_val} {b_val}".strip()
        return f"SILVERTONE - 1484 - {settings} [{nam_file.capture_type}]{nam_file.ext.upper()}"

class MarshallJCM800Parser(BaseParser):
    def match(self, nam_file: NAMFile) -> bool:
        return "JCM800" in nam_file.upper_name or "JCM 800" in nam_file.upper_name

    def parse(self, nam_file: NAMFile) -> str:
        # 1. Se il file è già formattato correttamente, lo ignora mantenendo il nome attuale
        if re.match(r'^MARSHALL\s*-\s*JCM800\s*2203\s*-\s*.*\[(DI|CAB)\]\.(NAM|nam)$', nam_file.filename):
            return nam_file.filename

        clean_name = nam_file.clean_text()

        # 2. Estrazione parametri standard Marshall (P, B, M, T, MV, G)
        p_match = re.search(r'\bP(\d+)\b', clean_name)
        b_match = re.search(r'\bB(\d+)\b', clean_name)
        m_match = re.search(r'\bM(\d+)\b', clean_name)
        t_match = re.search(r'\bT(\d+)\b', clean_name)
        mv_match = re.search(r'\bMV(\d+)\b', clean_name)
        g_match = re.search(r'\bG(\d+)\b', clean_name)

        # Fallback a valori standard (5) se non espressamente trovati nel nome
        p_val = f"P{p_match.group(1)}" if p_match else "P5"
        b_val = f"B{b_match.group(1)}" if b_match else "B5"
        m_val = f"M{m_match.group(1)}" if m_match else "M5"
        t_val = f"T{t_match.group(1)}" if t_match else "T5"
        mv_val = f"MV{mv_match.group(1)}" if mv_match else "MV5"
        g_val = f"G{g_match.group(1)}" if g_match else "G5"

        settings = f"{p_val} {b_val} {m_val} {t_val} {mv_val} {g_val}"

        return f"MARSHALL - JCM800 2203 - {settings} [{nam_file.capture_type}]{nam_file.ext.upper()}"

class BognerEcstasyParser(BaseParser):
    def match(self, nam_file: NAMFile) -> bool:
        return any(k in nam_file.upper_name for k in ["BOGNER", "ECSTASY", "101B"])

    def parse(self, nam_file: NAMFile) -> str:
        # 1. Se il file è già formattato correttamente, restituisce il nome originale intatto
        if re.match(r'^BOGNER\s*-\s*ECSTASY\s*101B\s*-\s*.*\[(DI|CAB)\]\.(NAM|nam)$', nam_file.filename):
            return nam_file.filename

        text = nam_file.upper_name

        # 2. Identificazione Canale / Modalità
        if "PLEXI" in text:
            ch_mode = "PLEXI"
        elif "RED" in text:
            ch_mode = "RED"
        elif "BLUE" in text:
            ch_mode = "BLUE"
        else:
            ch_mode = "CH"

        # 3. Gestione Boost (BST)
        boost_str = " BST" if "BOOST" in text else ""

        # 4. Impostazioni Knob (Gain al massimo a 10 come indicato dall'autore, EQ neutrale a 5)
        settings = f"{ch_mode}{boost_str} P5 B5 M5 T5 MV5 G10"

        return f"BOGNER - ECSTASY 101B - {settings} [{nam_file.capture_type}]{nam_file.ext.upper()}"

class GenericAmpParser(BaseParser):
    """Parser generico per Fender, Marshall, Mesa Boogie, ecc."""
    MODEL_MAP = {
        "DR103": ("HIWATT", "DR103"),
        "BASSMAN": ("FENDER", "BASSMAN"),
        "TWEED DELUXE 5E3": ("FENDER", "TWEED DELUXE 5E3"),
        "5E3": ("FENDER", "TWEED DELUXE 5E3"),
        "DELUXE REVERB": ("FENDER", "DELUXE REVERB"),
        "TWIN REVERB": ("FENDER", "TWIN REVERB")
    }

    CHANNEL_MAP = {
        "NORMAL": "NORM", "NORM": "NORM",
        "VIBRATO": "VIB", "VIB": "VIB",
        "BASS": "BASS", "BRIGHT": "BRIGHT"
    }

    def match(self, nam_file: NAMFile) -> bool:
        return True

    def parse(self, nam_file: NAMFile) -> str:
        cleaned = nam_file.clean_text()

        brand_str, model_str = "UNKNOWN", "AMP"
        for key, (b, m) in self.MODEL_MAP.items():
            if re.search(r'\b' + re.escape(key) + r'\b', cleaned):
                brand_str, model_str = b, m
                cleaned = re.sub(r'\b' + re.escape(key) + r'\b', '', cleaned)
                break

        channel_found = None
        for token in cleaned.split():
            if token in self.CHANNEL_MAP:
                channel_found = self.CHANNEL_MAP[token]
                break

        eq_parts = []
        for param, prefix in [("V|VOL|GAIN", "V"), ("T|TREBLE|TONE", "T"), ("M|MID|MIDDLE", "M"), ("B|BASS", "B")]:
            match = re.search(r'\b(?:' + param + r')\s*(\d+(?:\.\d+)?)\b', cleaned)
            if match:
                eq_parts.append(f"{prefix}{match.group(1)}")

        setting_parts = []
        if channel_found: setting_parts.append(channel_found)
        if eq_parts: setting_parts.append(" ".join(eq_parts))

        setting_str = " ".join(setting_parts).strip() or "DEFAULT"
        return f"{brand_str} - {model_str} - {setting_str} [{nam_file.capture_type}]{nam_file.ext.upper()}"

# ==============================================================================
# 3. MANAGER DI RINOMINA ED ESECUZIONE
# ==============================================================================

class NAMRenamerEngine:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.parsers = [
            HiwattParser(),
            RolandParser(),
            VoxAC15Parser(),
            VoxAC30Parser(),
            FenderPrincetonParser(),
            OrangeRockerverbParser(),
            SilvertoneParser(),
            MarshallJCM800Parser(),
            BognerEcstasyParser(),
            GenericAmpParser()
        ]

    def process_file(self, filepath):
        nam_file = NAMFile(filepath)
        for parser in self.parsers:
            if parser.match(nam_file):
                return parser.parse(nam_file)
        return None

    def run(self):
        rename_queue, delete_queue = [], []
        allocated_paths = set()
        ignored_count = 0

        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.nam')]

        for filename in sorted(files):
            old_path = os.path.join(self.folder_path, filename)
            new_name = self.process_file(old_path)
            
            if not new_name:
                continue

            # Se il nuovo nome è identico a quello attuale, il file è già corretto
            if filename == new_name:
                allocated_paths.add(old_path)
                ignored_count += 1
                continue

            new_path = os.path.join(self.folder_path, new_name)

            if new_path in allocated_paths or os.path.exists(new_path):
                delete_queue.append((filename, old_path))
            else:
                rename_queue.append((filename, new_name, old_path, new_path))
                allocated_paths.add(new_path)

        # Stampa Anteprima
        print("=" * 75)
        print(" ANTEPRIMA OPERAZIONI (ARCHITETTURA A PLUGIN)")
        print("=" * 75)

        if rename_queue:
            print("\n--- FILE DA RINOMINARE ---")
            for old, new, _, _ in rename_queue:
                print(f"[RINOMINA] {old}\n       -> {new}")

        if delete_queue:
            print("\n--- DUPLICATI DA ELIMINARE ---")
            for old, _ in delete_queue:
                print(f"[ELIMINA]  {old}")

        print("=" * 75)
        print(f"Rinomi: {len(rename_queue)} | Eliminati: {len(delete_queue)} | Ignorati (già corretti): {ignored_count}")
        print("=" * 75)

        if rename_queue or delete_queue:
            confirm = input("\nProcedere con le modifiche? [s/N]: ").strip().lower()
            if confirm in ["s", "si", "y", "yes"]:
                for _, _, old_p, new_p in rename_queue:
                    os.rename(old_p, new_p)
                for _, old_p in delete_queue:
                    os.remove(old_p)
                print("Operazione completata!")

if __name__ == "__main__":
    engine = NAMRenamerEngine(FOLDER_PATH)
    engine.run()