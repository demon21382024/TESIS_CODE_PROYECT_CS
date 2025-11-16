# train_hybrid.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# Importar desde nuestros módulos
from configs import settings
from src.data import CASIAB_Supervised
from src.models import SupervisedReIDModel, HybridGaitModel, GaitBackbone
from src.losses import TripletLoss # ¡Importamos TripletLoss!

def train_hybrid(config):
    """Función principal de entrenamiento Híbrido"""
    
    print("\n--- Cargando Datasets (para Fase Híbrida) ---")
    # Usamos la misma configuración de datos que la fase supervisada
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
    
    # --- Cargar Modelo Supervisado Pre-entrenado ---
    
    # 1. Cargar checkpoint
    if not os.path.exists(config.SUPERVISED_CHECKPOINT):
        print(f"  ERROR: No se encontró {config.SUPERVISED_CHECKPOINT}")
        print(f"  Por favor, ejecuta train_supervised.py primero.")
        return
        
    checkpoint = torch.load(config.SUPERVISED_CHECKPOINT, map_location=config.DEVICE)
    
    # 2. Re-crear la arquitectura del modelo supervisado
    #    (Necesitamos un backbone 'dummy' solo para inicializar SupervisedReIDModel)
    dummy_backbone = GaitBackbone() 
    supervised_model = SupervisedReIDModel(
        backbone=dummy_backbone,
        num_classes=checkpoint['num_classes']
    )
    
    # 3. Cargar los pesos entrenados
    supervised_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Modelo Supervisado cargado desde: {config.SUPERVISED_CHECKPOINT}")
    print(f"    (Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")

    # 4. Envolverlo en el Modelo Híbrido
    model = HybridGaitModel(supervised_model).to(config.DEVICE)

    # --- Configuración de Entrenamiento Híbrido ---
    criterion_ce = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=config.HYBRID_MARGIN)
    
    optimizer = optim.Adam(model.parameters(), lr=config.HYBRID_LEARNING_RATE, weight_decay=1e-4)
    # Usamos un scheduler más suave para fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    print(f"\n--- Configuración Híbrida ---")
    print(f"Epochs: {config.HYBRID_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate (inicio): {config.HYBRID_LEARNING_RATE}")
    print(f"Peso Triplet: {config.HYBRID_TRIPLET_WEIGHT}\n")
    
    best_val_acc = 0.0
    
    for epoch in range(1, config.HYBRID_EPOCHS + 1):
        # TRAINING
        model.train()
        train_loss_ce, train_loss_triplet, train_correct, train_total = 0, 0, 0, 0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            # El modelo ahora devuelve logits Y embeddings
            logits, embeddings = model(images)
            
            # --- Calcular Pérdida Híbrida ---
            loss_ce = criterion_ce(logits, labels)
            loss_triplet = criterion_triplet(embeddings, labels)
            
            loss = loss_ce + (config.HYBRID_TRIPLET_WEIGHT * loss_triplet)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_ce += loss_ce.item()
            train_loss_triplet += loss_triplet.item()
            
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss_ce = train_loss_ce / len(train_loader)
        avg_train_loss_triplet = train_loss_triplet / len(train_loader)
        
        # VALIDATION (Solo evaluamos la precisión de clasificación)
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                
                # Solo necesitamos los logits para la validación de Acc
                logits, _ = model(images) 
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # El scheduler usa el promedio de las pérdidas de train
        avg_total_loss = avg_train_loss_ce + avg_train_loss_triplet
        scheduler.step(avg_total_loss) 
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:2d}/{config.HYBRID_EPOCHS} | "
              f"L_CE: {avg_train_loss_ce:.3f} | "
              f"L_Trip: {avg_train_loss_triplet:.3f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
            }, config.HYBRID_CHECKPOINT)

    print(f"\nENTRENAMIENTO HÍBRIDO COMPLETADO")
    print(f"Mejor Val Accuracy: {best_val_acc:.2f}%")
    print(f"Modelo guardado en: {config.HYBRID_CHECKPOINT}")
    return model


if __name__ == "__main__":
    # 1. Verificar rutas y configuraciones
    settings.check_paths()
    
    # 2. Iniciar el entrenamiento
    train_hybrid(settings)