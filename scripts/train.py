import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from mel import GuitarDataset
from model import FullSignalChainEstimator


def get_next_model_path(models_dir: str, prefix: str = "full_signal_chain"):
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


def prepare_input(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Assicura la forma [B, 1, F, T] per gli spettrogrammi Mel."""
    x = x.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def compute_multi_task_loss_and_metrics(predictions, targets, criterion_bce, criterion_ce, criterion_mse, weights=None):
    """ Calcola la Multi-Task Loss bilanciata e calcola le metriche di accuratezza e MAE.
    """
    if weights is None:
        weights = {"active": 1.0, "model": 1.0, "params": 2.0}

    total_loss = 0.0
    num_blocks = len(predictions)

    total_act_correct = 0
    total_act_samples = 0
    total_mod_correct = 0
    total_mod_samples = 0
    param_mae_sum = 0.0
    param_mae_count = 0

    for b_id, pred in predictions.items():
        if b_id not in targets:
            continue

        target_b = targets[b_id]
        
        target_active = target_b["active"].to(pred["active_logits"].device).float()
        target_model = target_b["model"].to(pred["model_logits"].device).long()
        target_params = target_b["params"].to(pred["params"].device).float()

        # 1. Active Loss (BCE)
        loss_act = criterion_bce(pred["active_logits"], target_active)

        # Active Accuracy Tracking
        act_preds = (torch.sigmoid(pred["active_logits"]) > 0.5).float()
        total_act_correct += (act_preds == target_active).sum().item()
        total_act_samples += target_active.numel()

        # Maschera per abilitare loss di modello e parametri solo sui blocchi attivi
        active_mask = (target_active > 0.5).float()
        num_active = active_mask.sum()

        # 2. Model Loss (CrossEntropy mascherata mantenendo la connessione al grafo)
        raw_ce = criterion_ce(pred["model_logits"], target_model)
        if num_active > 0:
            loss_mod = (raw_ce * active_mask).sum() / (num_active + 1e-8)
            
            # Model Accuracy (solo su blocchi attivi)
            mod_preds = pred["model_logits"].argmax(dim=-1)
            active_indices = active_mask.bool()
            total_mod_correct += (mod_preds[active_indices] == target_model[active_indices]).sum().item()
            total_mod_samples += num_active.item()
        else:
            loss_mod = 0.0 * raw_ce.sum()

        # 3. Parameter Loss (MSE mascherata mantenendo la connessione al grafo)
        param_mask = active_mask.unsqueeze(-1).expand_as(pred["params"])
        raw_mse = criterion_mse(pred["params"], target_params)
        
        if num_active > 0:
            loss_par = (raw_mse * param_mask).sum() / (param_mask.sum() + 1e-8)
            
            # Parameter MAE Tracking (solo su blocchi attivi)
            abs_diff = torch.abs(pred["params"] - target_params) * param_mask
            param_mae_sum += abs_diff.sum().item()
            param_mae_count += param_mask.sum().item()
        else:
            loss_par = 0.0 * raw_mse.sum()

        # Loss del singolo blocco
        block_loss = (weights["active"] * loss_act) + (weights["model"] * loss_mod) + (weights["params"] * loss_par)
        total_loss += block_loss

    # Normalizzazione della loss rispetto al numero di blocchi nella catena
    normalized_loss = total_loss / max(num_blocks, 1)

    # Aggregazione delle metriche
    metrics = {
        "act_acc": total_act_correct / max(total_act_samples, 1),
        "mod_acc": total_mod_correct / max(total_mod_samples, 1),
        "param_mae": param_mae_sum / max(param_mae_count, 1e-8)
    }

    return normalized_loss, metrics


def train_pipeline(chain_configuration=None, epochs=60, batch_size=32, lr=1e-3, dataset_dir=None, models_dir=None):
    is_colab = os.path.exists("/content")

    if dataset_dir is None:
        dataset_dir = "/content/dataset" if is_colab else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

    if models_dir is None:
        models_dir = "/content/drive/MyDrive/guitar_effects_ai/models" if is_colab else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo attivo per il training: {device}")

    # Dataset & Dataloader setup
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
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    # Inizializzazione Modello
    model = FullSignalChainEstimator(block_configs=chain_configuration).to(device)

    save_path, plot_path = get_next_model_path(models_dir)
    print(f"\n==========================================")
    print(f" Avvio Training: Full Signal Chain Estimator")
    print(f" Modello: {save_path}")
    print(f"==========================================")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

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
        train_act_accs, train_mod_accs, train_param_maes = [], [], []

        for batch in train_loader:
            x, targets = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else (batch["spectrogram"], batch["targets"])
            x = prepare_input(x, device)
            
            optimizer.zero_grad()
            predictions = model(x)
            
            loss, metrics = compute_multi_task_loss_and_metrics(predictions, targets, criterion_bce, criterion_ce, criterion_mse)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            train_act_accs.append(metrics["act_acc"])
            train_mod_accs.append(metrics["mod_acc"])
            train_param_maes.append(metrics["param_mae"])

        epoch_train_loss = running_loss / len(train_subset)
        avg_train_act = sum(train_act_accs) / len(train_act_accs)
        avg_train_mod = sum(train_mod_accs) / len(train_mod_accs)
        avg_train_mae = sum(train_param_maes) / len(train_param_maes)

        # Phase: Validation
        model.eval()
        val_loss = 0.0
        val_act_accs, val_mod_accs, val_param_maes = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x, targets = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else (batch["spectrogram"], batch["targets"])
                x = prepare_input(x, device)
                
                predictions = model(x)
                loss, metrics = compute_multi_task_loss_and_metrics(predictions, targets, criterion_bce, criterion_ce, criterion_mse)

                val_loss += loss.item() * x.size(0)
                val_act_accs.append(metrics["act_acc"])
                val_mod_accs.append(metrics["mod_acc"])
                val_param_maes.append(metrics["param_mae"])

        epoch_val_loss = val_loss / len(val_subset)
        avg_val_act = sum(val_act_accs) / len(val_act_accs)
        avg_val_mod = sum(val_mod_accs) / len(val_mod_accs)
        avg_val_mae = sum(val_param_maes) / len(val_param_maes)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{epochs}] | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Val Act Acc: {avg_val_act*100:.1f}% | Val Mod Acc: {avg_val_mod*100:.1f}% | Val Param MAE: {avg_val_mae:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'chain_configuration': chain_configuration,
                'epoch': epoch + 1,
                'val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path)
            print(" ---> Modello e stato ottimizzatore salvati con successo!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] Nessun miglioramento per {patience} epoche. Training interrotto.")
                break

    # Grafico Loss Curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Multi-Task Normalizzata')
    plt.title('Loss Curve - Full Signal Chain Estimator')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    train_pipeline(chain_configuration=None, epochs=60)