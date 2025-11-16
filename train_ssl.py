# train_ssl.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
import os

# Importar desde nuestros módulos
from configs import settings
from src.data import CASIAB_SSL
from src.models import GaitBackbone
from src.losses import NTXentLoss

def train_ssl(config):
    """Función principal de entrenamiento SSL"""
    
    dataset = CASIAB_SSL(
        root_path=config.ROOT_PATH,
        img_size=config.IMG_SIZE,
        use_subset=config.SSL_USE_SUBSET,
        subset_subjects=config.SSL_SUBSET_SUBJECTS,
        subset_conditions=config.SSL_SUBSET_CONDITIONS,
        subset_angles=config.SSL_SUBSET_ANGLES,
        frames_per_seq=config.SSL_SUBSET_FRAMES_PER_SEQ
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0, # 0 es mejor para debugging en Windows
        pin_memory=True if config.DEVICE == "cuda" else False,
    )
    
    model = GaitBackbone(embed_dim=256).to(config.DEVICE)
    criterion = NTXentLoss(temperature=config.SSL_TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=config.SSL_LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.SSL_EPOCHS, eta_min=1e-6)
    
    USE_MIXED_PRECISION = False 
    scaler = GradScaler('cuda') if (config.DEVICE == "cuda" and USE_MIXED_PRECISION) else None
    
    print(f"\n--- Configuración SSL ---")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.SSL_EPOCHS}")
    print(f"Learning rate: {config.SSL_LEARNING_RATE}")
    print(f"Steps/epoch: {len(loader)}")
    
    best_loss = float('inf')
    
    for epoch in range(1, config.SSL_EPOCHS + 1):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (view1, view2) in enumerate(loader):
            view1 = view1.to(config.DEVICE, non_blocking=True)
            view2 = view2.to(config.DEVICE, non_blocking=True)
            
            if scaler:
                with autocast('cuda'):
                    z1 = model(view1)
                    z2 = model(view2)
                    loss = criterion(z1, z2)
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z1 = model(view1)
                z2 = model(view2)
                loss = criterion(z1, z2)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(loader)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:2d}/{config.SSL_EPOCHS} | Loss: {avg_epoch_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, config.SSL_CHECKPOINT)
    
    print(f"\nEntrenamiento SSL finalizado")
    print(f"Mejor loss: {best_loss:.4f}")
    print(f"Modelo guardado en: {config.SSL_CHECKPOINT}")
    return model

if __name__ == "__main__":
    # 1. Verificar rutas y configuraciones
    settings.check_paths()
    
    # 2. Iniciar el entrenamiento
    train_ssl(settings)