import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from mel import GuitarDataset
import matplotlib.pyplot as plt

# Importazione dei modelli modulari
from model import (
    AmpNet,
    DriveNet, 
    CabinetNet, 
    ChorusNet, 
    HighpassNet
)

def get_next_model_path(models_dir, model_prefix):
    """
    Genera percorsi dinamici per salvare checkpoints e grafici di loss 
    basati sul nome del modulo (es. amp_model1.pth).
    """
    os.makedirs(models_dir, exist_ok=True)
    existing_indices = []
    
    prefix = f"{model_prefix}_model"
    for filename in os.listdir(models_dir):
        if filename.startswith(prefix) and filename.endswith(".pth"):
            num_part = filename[len(prefix):-4]
            if num_part.isdigit():
                existing_indices.append(int(num_part))
    
    next_idx = max(existing_indices) + 1 if existing_indices else 1
    model_name = f"{prefix}{next_idx}.pth"
    plot_name = f"loss_curve_{prefix}{next_idx}.png"
    
    return os.path.join(models_dir, model_name), os.path.join(models_dir, plot_name)


def train_single_module(model, model_name, train_loader, val_loader, compute_loss_fn, device, models_dir, extra_meta=None, epochs=60, lr=5e-4):
    save_path, plot_path = get_next_model_path(models_dir, model_prefix=model_name)
    print(f"\n==========================================")
    print(f" Avvio Training: {model_name}")
    print(f" Modello: {save_path}")
    print(f"==========================================")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience, patience_counter = 12, 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Phase: Train
        model.train()
        running_loss = 0.0
        total_train_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss_fn(model, batch, device)
            loss.backward()
            optimizer.step()
            
            batch_size = batch[0].size(0)
            running_loss += loss.item() * batch_size
            total_train_samples += batch_size

        epoch_train_loss = running_loss / total_train_samples

        # Phase: Validation
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = compute_loss_fn(model, batch, device)
                batch_size = batch[0].size(0)
                val_loss += loss.item() * batch_size
                total_val_samples += batch_size

        epoch_val_loss = val_loss / total_val_samples
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        print(f"[{model_name}] Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            checkpoint = {'model_state_dict': model.state_dict()}
            if extra_meta:
                checkpoint.update(extra_meta)
            torch.save(checkpoint, save_path)
            print(f" ---> Modello salvato con successo!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] Training interrotto per {model_name}.")
                break

    #Grafico della loss
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()


def prepare_input(x, device):
    """Assicura che il tensore di input abbia 4 dimensioni [B, 1, F, T]."""
    x = x.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def train_pipeline(target_module="all", dataset_dir=None, models_dir=None):
    is_colab = os.path.exists("/content")
    if dataset_dir is None:
        dataset_dir = "/content/dataset" if is_colab else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    if models_dir is None:
        models_dir = "/content/drive/MyDrive/guitar_effects_ai/models" if is_colab else os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo attivo: {device}")

    # Caricamento del dataset
    train_dataset_full = GuitarDataset(dataset_dir=dataset_dir, is_train=True)
    val_dataset_full = GuitarDataset(dataset_dir=dataset_dir, is_train=False)

    total_samples = len(train_dataset_full)
    train_size = int(0.85 * total_samples)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=generator).tolist()

    train_subset = Subset(train_dataset_full, indices[:train_size])
    val_subset = Subset(val_dataset_full, indices[train_size:])

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    # Criteria di loss
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    # --- DEFINIZIONE LOGICA DI LOSS UNIFICATA ---
    
    def loss_amp(model, batch, device):
        x, y_amp, _, _ = batch
        x = prepare_input(x, device)
        y_amp = y_amp.to(device)
        logits = model(x)
        return criterion_ce(logits, y_amp)

    def loss_drive(model, batch, device):
        # Presuppone che y_params contenga l'indice di tipo overdrive [0] e modello [1]
        x, _, _, y_params = batch
        x = prepare_input(x, device)
        y_params = y_params.to(device)
        y_drive_type = y_params[:, 0].long()
        y_drive_model = y_params[:, 1].long()
        
        pred_type, pred_model = model(x)
        return criterion_ce(pred_type, y_drive_type) + criterion_ce(pred_model, y_drive_model)

    def loss_cabinet(model, batch, device):
        # Presuppone che il target IR cabinet risieda all'interno di y_params (es. indice 7)
        x, _, _, y_params = batch
        x = prepare_input(x, device)
        y_cab = y_params[:, 7].long() if y_params.size(1) > 7 else y_params[:, -1].long()
        y_cab = y_cab.to(device)
        
        logits = model(x)
        return criterion_ce(logits, y_cab)

    def loss_chorus(model, batch, device):
        x, _, y_onoff, y_params = batch
        x = prepare_input(x, device)
        y_onoff, y_params = y_onoff.to(device), y_params.to(device)
        
        target_active = y_onoff[:, 2:3]
        target_chorus_params = y_params[:, 2:7]
        
        pred_active, pred_params = model(x)
        l_active = criterion_bce(pred_active, target_active)
        
        # Mascheramento corretto calcolato solo sulle istanze con effetto attivo
        mask = target_active.repeat(1, pred_params.size(1))
        active_elements = mask.sum()
        
        if active_elements > 0:
            l_params = torch.sum(((pred_params - target_chorus_params) ** 2) * mask) / active_elements
        else:
            l_params = 0.0
            
        return l_active + 1.0 * l_params

    def loss_highpass(model, batch, device):
        x, _, y_onoff, y_params = batch
        x = prepare_input(x, device)
        y_onoff, y_params = y_onoff.to(device), y_params.to(device)
        
        target_active = y_onoff[:, 0:1]
        target_cutoff = y_params[:, 0:1]
        
        pred_active, pred_cutoff = model(x)
        l_active = criterion_bce(pred_active, target_active)
        
        mask = target_active
        l_cutoff = criterion_mse(pred_cutoff * mask, target_cutoff * mask)
        
        return l_active + 0.5 * l_cutoff

    # Mappatura e istanziazione modelli
    amp_meta = {
        'amp_to_id': getattr(train_dataset_full, 'amp_to_id', {}),
        'id_to_amp': getattr(train_dataset_full, 'id_to_amp', {})
    }

    modules_to_train = {
        "amp": (AmpNet(num_classes=len(train_dataset_full.amp_to_id)), loss_amp, amp_meta),
        "drive": (DriveNet(num_types=3, num_models=10), loss_drive, None),
        "cabinet": (CabinetNet(num_cabs=9), loss_cabinet, None),
        "chorus": (ChorusNet(num_params=5), loss_chorus, None),
        "highpass": (HighpassNet(), loss_highpass, None)
    }

    # Esecuzione della pipeline
    if target_module == "all":
        for name, (net, loss_fn, meta) in modules_to_train.items():
            train_single_module(net, name, train_loader, val_loader, loss_fn, device, models_dir, extra_meta=meta)
    elif target_module in modules_to_train:
        net, loss_fn, meta = modules_to_train[target_module]
        train_single_module(net, target_module, train_loader, val_loader, loss_fn, device, models_dir, extra_meta=meta)
    else:
        raise ValueError(f"Modulo '{target_module}' non valido. Scegli tra: {list(modules_to_train.keys())} o 'all'.")

if __name__ == "__main__":
    train_pipeline(target_module="all")