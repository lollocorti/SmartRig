import os
import shutil
from audio import generate_audio_dataset
from mel import generate_mel_spectrograms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
NUM_SAMPLES = 5000

def create_colab_zip(dataset_dir=DATASET_DIR):
    """Crea un archivio zip leggero contenente solo mel/ e dataset_labels.json per Colab."""
    print("--- Creazione archivio zip ottimizzato per Google Colab ---")
    zip_base_name = os.path.join(PROJECT_ROOT, "dataset_colab")
    
    # Cartella temporanea per assemblare solo i file leggeri necessari al training
    temp_dir = os.path.join(PROJECT_ROOT, "temp_colab_dataset")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Copia solo la cartella mel/ e il file dei metadati json
        json_src = os.path.join(dataset_dir, "dataset_labels.json")
        mel_src = os.path.join(dataset_dir, "mel")
        
        if os.path.exists(json_src):
            shutil.copy(json_src, temp_dir)
        if os.path.exists(mel_src):
            shutil.copytree(mel_src, os.path.join(temp_dir, "mel"), dirs_exist_ok=True)

        # Creazione del file dataset_colab.zip
        archive_path = shutil.make_archive(zip_base_name, "zip", temp_dir)
        print(f"Archivio creato con successo: {archive_path}")
        
    finally:
        # Pulizia cartella temporanea
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    print("==========================================")
    print("   AVVIO GENERAZIONE DATASET COMPLETO     ")
    print("==========================================")
    
    # Rilevamento automatico dei core disponibili per massimizzare la velocità
    cpu_workers = max(1, os.cpu_count() or 4)
    
    # 1. Generazione dei file audio .wav e del file dataset_labels.json
    generate_audio_dataset(num_samples=NUM_SAMPLES, dataset_dir=DATASET_DIR, max_workers=cpu_workers)
    
    # 2. Generazione degli spettrogrammi Mel .npy basati su dataset_labels.json
    generate_mel_spectrograms(dataset_dir=DATASET_DIR)

    # 3. Creazione del pacchetto zip per Google Colab
    create_colab_zip(dataset_dir=DATASET_DIR)

    print("==========================================")
    print("   DATASET GENERATO E PRONTO PER IL TRAIN  ")
    print("==========================================")

if __name__ == "__main__":
    main()