import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import GuitarDataset
from model import FullSignalChainEstimator

def main():
    # 1. Configurazione Iniziale
    dataset_dir = "./dataset"  # Aggiorna con il percorso corretto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo di addestramento: {device}")

    # 2. Caricamento Dataset e Split
    full_dataset = GuitarDataset(dataset_dir, is_train=True)
    chain_config = full_dataset.get_chain_config()
    
    total_size = len(full_dataset)
    val_size = int(0.15 * total_size)
    train_size = total_size - val_size
    
    # Split deterministico per mantenere consistenza tra i run
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Disabilita le augmentation (masking) sul validation set
    val_data.dataset.is_train = False 

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Inizializzazione Modello
    model = FullSignalChainEstimator(chain_config).to(device)

    # 4. Funzioni di Loss e Pesi
    criterion_active = nn.BCEWithLogitsLoss()
    criterion_model = nn.CrossEntropyLoss()
    criterion_params = nn.MSELoss()

    # Pesi bilanciati: Riduciamo drasticamente l'impatto della MSELoss iniziale
    loss_weights = {"active": 2.0, "model": 2.0, "params": 0.1}

    # 5. Ottimizzatore e Scheduler per Transformer
    epochs = 150
    # AdamW è superiore ad Adam standard quando si usano i Transformer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # OneCycleLR gestisce automaticamente il "warmup" (crescita graduale del LR) per le prime epoche
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-4, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )
    
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision per velocizzare e risparmiare VRAM

    # 6. Variabili Early Stopping
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0

    # 7. Ciclo di Addestramento
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (x, targets) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(x)
                batch_loss = 0.0

                for b_id, b_targets in targets.items():
                    out = outputs[b_id]
                    
                    target_active = b_targets["active"].to(device)
                    target_model = b_targets["model"].to(device)
                    target_params = b_targets["params"].to(device)

                    loss_active = criterion_active(out["active_logits"], target_active)
                    
                    # La loss del modello e dei parametri ha senso solo se il blocco è attivo
                    mask_active = target_active > 0.5
                    if mask_active.any():
                        loss_model = criterion_model(out["model_logits"][mask_active], target_model[mask_active])
                        loss_params = criterion_params(out["params"][mask_active], target_params[mask_active])
                    else:
                        loss_model = torch.tensor(0.0, device=device)
                        loss_params = torch.tensor(0.0, device=device)

                    block_loss = (loss_weights["active"] * loss_active) + \
                                 (loss_weights["model"] * loss_model) + \
                                 (loss_weights["params"] * loss_params)
                    
                    batch_loss += block_loss

            # Scaler per Mixed Precision e Gradient Clipping
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            # Il Gradient Clipping è vitale per prevenire l'esplosione dei gradienti nei Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # Aggiorna il learning rate ad ogni step (batch)

            train_loss += batch_loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 8. Ciclo di Validazione
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, targets in val_loader:
                x = x.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(x)
                    batch_loss = 0.0
                    for b_id, b_targets in targets.items():
                        out = outputs[b_id]
                        target_active = b_targets["active"].to(device)
                        target_model = b_targets["model"].to(device)
                        target_params = b_targets["params"].to(device)

                        loss_active = criterion_active(out["active_logits"], target_active)
                        
                        mask_active = target_active > 0.5
                        if mask_active.any():
                            loss_model = criterion_model(out["model_logits"][mask_active], target_model[mask_active])
                            loss_params = criterion_params(out["params"][mask_active], target_params[mask_active])
                        else:
                            loss_model = torch.tensor(0.0, device=device)
                            loss_params = torch.tensor(0.0, device=device)

                        block_loss = (loss_weights["active"] * loss_active) + \
                                     (loss_weights["model"] * loss_model) + \
                                     (loss_weights["params"] * loss_params)
                        batch_loss += block_loss
                    
                    val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1:03d}/{epochs} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 9. Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> Modello migliorato e salvato!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping attivato all'epoca {epoch+1}. La Validation Loss non migliora da {patience} epoche.")
                break

if __name__ == "__main__":
    main()