import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from mel import GuitarDataset
import matplotlib.pyplot as plt

# Importazione dell'architettura unificata
from model import FullSignalChainEstimator


def get_next_model_path(models_dir, prefix="full_signal_chain"):
    os.makedirs(models_dir, exist_ok=True)
    existing_indices = []
    
    for filename in os.listdir(models_dir):
        if filename.startswith(prefix) and filename.endswith(".pth"):
            num_part = filename[len(prefix):-4].strip("_")
            if num_part.isdigit():
                existing_indices.append(int(num_part))
    
    next_idx = max(existing_indices) + 1 if existing_indices else 1
    model_name = f"{prefix}_{next_idx}.pth"
    plot_name = f"loss_curve_{prefix}_{next_idx}.png"
    
    return os.path.join(models_dir, model_name), os.path.join(models_dir, plot_name)


def prepare_input(x, device):
    """Assicura che il tensore di input abbia 4 dimensioni [B, 1, F, T]."""
    x = x.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def compute_multi_task_loss(predictions, targets, criterion_bce, criterion_ce, criterion_mse, weights=None):
    """
    Calcola la loss globale multi-task per tutti i blocchi della catena.
    
    Args:
        predictions: Dict generato da FullSignalChainEstimator -> { block_id: { "active_logits", "model_logits", "params" } }
        targets: Dict proveniente dal DataLoader contenente le etichette target per blocco.
    """
    if weights is None:
        weights = {"active": 1.0, "model": 1.0, "params": 2.0}

    total_loss = 0.0

    for b_id, pred in predictions.items():
        if b_id not in targets:
            continue

        target_b = targets[b_id]
        
        # Target dei singoli Task
        target_active = target_b["active"].to(pred["active_logits"].device).float()  # [B]
        target_model = target_b["model"].to(pred["model_logits"].device).long()     # [B]
        target_params = target_b["params"].to(pred["params"].device).float()        # [B, max_params]

        # 1. Active Loss (BCE)
        loss_act = criterion_bce(pred["active_logits"], target_active)

        # Maschera per abilitare loss di modello e parametri solo sui blocchi attivi
        active_mask = (target_active > 0.5).float() # [B]
        num_active = active_mask.sum()

        # 2. Model Loss (CrossEntropy mascherata)
        if num_active > 0:
            raw_ce = criterion_ce(pred["model_logits"], target_model) # [B]
            loss_mod = (raw_ce * active_mask).sum() / num_active
        else:
            loss_mod = torch.tensor(0.0, device=pred["active_logits"].device)

        # 3. Parameter Loss (MSE mascherata)
        if num_active > 0:
            param_mask = active_mask.unsqueeze(-1).expand_as(pred["params"]) # [B, max_params]
            param_diff = ((pred["params"] - target_params) ** 2) * param_mask
            loss_par = param_diff.sum() / (param_mask.sum() + 1e-8)
        else:
            loss_par = torch.tensor(0.0, device=pred["active_logits"].device)

        # Loss pesata per il blocco corrente
        block_loss = (weights["active"] * loss_act) + (weights["model"] * loss_mod) + (weights["params"] * loss_par)
        total_loss += block_loss

    return total_loss


def train_pipeline(chain_configuration, epochs=60, lr=1e-3, dataset_dir=None, models_dir=None):
    is_colab = os.path.exists("/content")

    if dataset_dir is None:
        dataset_dir = "/content/dataset" if is_colab else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

    if models_dir is None:
        if is_colab:
            models_dir = "/content/drive/MyDrive/guitar_effects_ai/models"
        else:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo attivo per il training: {device}")

    # Caricamento del dataset
    full_dataset = GuitarDataset(dataset_dir=dataset_dir)
    if chain_configuration is None:
            chain_configuration = full_dataset.get_chain_config()
    total_samples = len(full_dataset)
    train_size = int(0.85 * total_samples)
    val_size = total_samples - train_size

    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    num_workers = 2 if is_colab else 0
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    # Inizializzazione Modello Unificato
    model = FullSignalChainEstimator(block_configs=chain_configuration).to(device)

    save_path, plot_path = get_next_model_path(models_dir)
    print(f"\n==========================================")
    print(f" Avvio Training: Full Signal Chain Estimator")
    print(f" Modello: {save_path}")
    print(f"==========================================")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Criteria di Loss
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05)
    criterion_mse = nn.MSELoss(reduction='none')

    best_val_loss = float("inf")
    patience, patience_counter = 14, 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Phase: Train
        model.train()
        running_loss = 0.0
        total_train_samples = 0

        for batch in train_loader:
            # Assume che batch sia composto da (x_spectrograms, targets_dict)
            if isinstance(batch, (list, tuple)):
                x, targets = batch[0], batch[1]
            else:
                x, targets = batch["spectrogram"], batch["targets"]

            x = prepare_input(x, device)
            optimizer.zero_grad()
            
            predictions = model(x)
            loss = compute_multi_task_loss(predictions, targets, criterion_bce, criterion_ce, criterion_mse)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            total_train_samples += batch_size

        epoch_train_loss = running_loss / total_train_samples

        # Phase: Validation
        model.eval()
        val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, targets = batch[0], batch[1]
                else:
                    x, targets = batch["spectrogram"], batch["targets"]

                x = prepare_input(x, device)
                predictions = model(x)
                loss = compute_multi_task_loss(predictions, targets, criterion_bce, criterion_ce, criterion_mse)

                batch_size = x.size(0)
                val_loss += loss.item() * batch_size
                total_val_samples += batch_size

        epoch_val_loss = val_loss / total_val_samples
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{epochs}] | LR: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'chain_configuration': chain_configuration,
                'epoch': epoch + 1,
                'val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path)
            print(f" ---> Modello salvato con successo!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] Training interrotto.")
                break

    # Grafico della Loss Curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Multi-Task')
    plt.title('Loss Curve - Full Signal Chain')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    train_pipeline(chain_configuration = None, epochs=60)