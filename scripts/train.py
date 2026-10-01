import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import GuitarDataset
from model import FullSignalChainEstimator


def main(
    dataset_dir="/content/dataset",
    drive_models_dir="/content/drive/MyDrive/SmartRig/models",
):
  os.makedirs(drive_models_dir, exist_ok=True)
  model_save_path = os.path.join(drive_models_dir, "best_model.pth")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Dispositivo di addestramento: {device}")

  full_dataset = GuitarDataset(dataset_dir, is_train=True)
  chain_config = full_dataset.get_chain_config()

  total_size = len(full_dataset)
  val_size = int(0.15 * total_size)
  train_size = total_size - val_size

  generator = torch.Generator().manual_seed(42)
  train_data, val_data = random_split(
      full_dataset, [train_size, val_size], generator=generator
  )
  val_data.dataset.is_train = False

  train_loader = DataLoader(
      train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
  )
  val_loader = DataLoader(
      val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
  )

  model = FullSignalChainEstimator(chain_config).to(device)

  criterion_active = nn.BCEWithLogitsLoss()
  criterion_model = nn.CrossEntropyLoss()
  criterion_params = nn.HuberLoss()

  # Pesi ribilanciati per favorire l'apprendimento progressivo
  loss_weights = {"active": 2.0, "model": 0.5, "params": 1.0}

  epochs = 150
  optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs, eta_min=1e-6
  )

  scaler = torch.amp.GradScaler('cuda')

  best_val_loss = float('inf')
  patience = 25
  patience_counter = 0

  for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for x, targets in train_loader:
      x = x.to(device)
      optimizer.zero_grad(set_to_none=True)

      # TUTTO il forward e calcolo della loss all'interno dell'autocast
      with torch.amp.autocast('cuda'):
        outputs = model(x)
        accumulated_loss = torch.tensor(0.0, device=device)
        num_blocks = len(outputs)

        for b_id, b_targets in targets.items():
          out = outputs[b_id]

          target_active = b_targets['active'].to(device)
          target_model = b_targets['model'].to(device)
          target_params = b_targets['params'].to(device)

          loss_active = criterion_active(out['active_logits'], target_active)

          mask_active = target_active > 0.5
          if mask_active.any():
            loss_model = criterion_model(
                out['model_logits'][mask_active], target_model[mask_active]
            )
            loss_params = criterion_params(
                out['params'][mask_active], target_params[mask_active]
            )
          else:
            loss_model = torch.tensor(0.0, device=device)
            loss_params = torch.tensor(0.0, device=device)

          block_loss = (
              (loss_weights['active'] * loss_active)
              + (loss_weights['model'] * loss_model)
              + (loss_weights['params'] * loss_params)
          )
          accumulated_loss = accumulated_loss + block_loss

        # Normalizzazione della loss media per blocco
        batch_loss = accumulated_loss / num_blocks

      scaler.scale(batch_loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

      scaler.step(optimizer)
      scaler.update()

      train_loss += batch_loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)

    # Validazione
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for x, targets in val_loader:
        x = x.to(device)
        with torch.amp.autocast('cuda'):
          outputs = model(x)
          accumulated_loss = torch.tensor(0.0, device=device)
          num_blocks = len(outputs)

          for b_id, b_targets in targets.items():
            out = outputs[b_id]
            target_active = b_targets['active'].to(device)
            target_model = b_targets['model'].to(device)
            target_params = b_targets['params'].to(device)

            loss_active = criterion_active(out['active_logits'], target_active)

            mask_active = target_active > 0.5
            if mask_active.any():
              loss_model = criterion_model(
                  out['model_logits'][mask_active], target_model[mask_active]
              )
              loss_params = criterion_params(
                  out['params'][mask_active], target_params[mask_active]
              )
            else:
              loss_model = torch.tensor(0.0, device=device)
              loss_params = torch.tensor(0.0, device=device)

            block_loss = (
                (loss_weights['active'] * loss_active)
                + (loss_weights['model'] * loss_model)
                + (loss_weights['params'] * loss_params)
            )
            accumulated_loss = accumulated_loss + block_loss

          batch_loss = accumulated_loss / num_blocks
          val_loss += batch_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    current_lr = scheduler.get_last_lr()[0]

    print(
        f'Epoch {epoch+1:03d}/{epochs} | LR: {current_lr:.2e} | Train Loss:'
        f' {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
    )

    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      patience_counter = 0
      torch.save(model.state_dict(), model_save_path)
      print(f'  -> Modello migliorato e salvato su Drive: {model_save_path}')
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print(f'\nEarly stopping attivato all\'epoca {epoch+1}.')
        break


if __name__ == '__main__':
  main()