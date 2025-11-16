# train_supervised.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# Importar desde nuestros módulos
from configs import settings
from src.data import CASIAB_Supervised
from src.models import GaitBackbone, SupervisedReIDModel
# Nota: La TripletLoss está en src.losses, pero este script usa CrossEntropy

def train_supervised(config):
    """Función principal de entrenamiento supervisado"""
    
    print("\n--- Cargando Datasets Supervisados ---")
    train_dataset = CASIAB_Supervised(
        root_path=config.ROOT_PATH,
        subject_range=config.SUPERVISED_CONFIG['train_range'],
        conditions=config.SUPERVISED_CONFIG['conditions'],
        angles=config.SUPERVISED_CONFIG['angles'],
        frames_per_seq=config.SUPERVISED_SUBSET_FRAMES_PER_SEQ,
        img_size=config.IMG_SIZE,
        augment=True
    )
    
    val_dataset = CASIAB_Supervised(
        root_path=config.ROOT_PATH,
        subject_range=config.SUPERVISED_CONFIG['val_range'],
        conditions=config.SUPERVISED_CONFIG['conditions'],
        angles=config.SUPERVISED_CONFIG['angles'],
        frames_per_seq=config.SUPERVISED_SUBSET_FRAMES_PER_SEQ,
        img_size=config.IMG_SIZE,
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    num_classes = train_dataset.get_num_classes()
    print(f"\nClases (personas) en train: {num_classes}")
    
    # --- Cargar Backbone Pre-entrenado ---
    backbone = GaitBackbone(embed_dim=256)
    
    if os.path.exists(config.SSL_CHECKPOINT):
        checkpoint = torch.load(config.SSL_CHECKPOINT, map_location=config.DEVICE)
        backbone.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Backbone SSL cargado desde: {config.SSL_CHECKPOINT}")
        print(f"    (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    else:
        print(f"  ⚠ ADVERTENCIA: No se encontró {config.SSL_CHECKPOINT}")
        print(f"    Entrenando desde cero (backbone aleatorio).")
    
    model = SupervisedReIDModel(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=False  # Fine-tuning de todo el modelo
    ).to(config.DEVICE)
    
    # --- Configuración de Entrenamiento ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.SUPERVISED_LEARNING_RATE, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"\n--- Configuración Supervisada ---")
    print(f"Epochs: {config.SUPERVISED_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.SUPERVISED_LEARNING_RATE}")
    print(f"Steps/epoch: {len(train_loader)}\n")
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, config.SUPERVISED_EPOCHS + 1):
        # ... (Tu bucle de entrenamiento y validación va aquí) ...
        # (Es exactamente el mismo que tenías en el notebook)
        
        # TRAINING
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # VALIDATION
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        scheduler.step()
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:2d}/{config.SUPERVISED_EPOCHS} | "
              f"Loss: {train_loss/len(train_loader):.3f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'history': history
            }, config.SUPERVISED_CHECKPOINT)

    print(f"\nENTRENAMIENTO SUPERVISADO COMPLETADO")
    print(f"Mejor Val Accuracy: {best_val_acc:.2f}%")
    print(f"Modelo guardado en: {config.SUPERVISED_CHECKPOINT}")
    return model, history


if __name__ == "__main__":
    # 1. Verificar rutas y configuraciones
    settings.check_paths()
    
    # 2. Iniciar el entrenamiento
    train_supervised(settings)