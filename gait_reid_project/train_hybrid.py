# train_hybrid.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import sys
import numpy as np

# Importar desde nuestros módulos
from configs import settings
from src.data import CASIAB_Supervised
from src.models import HybridGaitModel, SupervisedReIDModel, GaitBackbone
from src.losses import TripletLoss, CircleLoss
from src.samplers import RandomIdentitySampler

def compute_mini_metrics(dist_matrix, q_labels, g_labels):
    """Métrica Local de validación sin dependencias externas"""
    num_queries = len(q_labels)
    if torch.is_tensor(dist_matrix): dist_matrix = dist_matrix.cpu().numpy()
    
    hits_top1, map_sum, valid_queries = 0, 0.0, 0
    for i in range(num_queries):
        dists_i = dist_matrix[i, :]
        sorted_indices = np.argsort(dists_i)
        sorted_g_labels = g_labels[sorted_indices]
        matches = (sorted_g_labels == q_labels[i]).astype(np.int32)
        
        num_tp = np.sum(matches)
        if num_tp == 0: continue
            
        valid_queries += 1
        hits_top1 += matches[0]
        
        num_hits, sum_prec = 0, 0
        for j, match in enumerate(matches):
            if match == 1:
                num_hits += 1
                sum_prec += num_hits / (j + 1)
        map_sum += (sum_prec / num_tp)
        
    r1 = (hits_top1 / valid_queries * 100) if valid_queries > 0 else 0
    mAP = (map_sum / valid_queries * 100) if valid_queries > 0 else 0
    return r1, mAP

def train_hybrid(config):
    print("\n[+] INICIANDO ENTRENAMIENTO HÍBRIDO (VOLUMÉTRICO)\n")
    
    # 1. Cargar Datasets PKSampler (B=32 : P=8, K=4)
    train_dataset = CASIAB_Supervised(
        root_path=config.ROOT_PATH,
        subject_range=config.SUPERVISED_CONFIG['train_range'],
        conditions=config.SUPERVISED_CONFIG['conditions'],
        angles=config.SUPERVISED_CONFIG['angles'],
        seq_len=config.SUPERVISED_SUBSET_FRAMES_PER_SEQ,
        img_size=config.IMG_SIZE,
        augment=True
    )
    
    val_dataset = CASIAB_Supervised( # Mini-val
        root_path=config.ROOT_PATH,
        subject_range=config.SUPERVISED_CONFIG['val_range'],
        conditions=config.SUPERVISED_CONFIG['conditions'],
        angles=config.SUPERVISED_CONFIG['angles'],
        seq_len=config.SUPERVISED_SUBSET_FRAMES_PER_SEQ,
        img_size=config.IMG_SIZE,
        augment=False,
        return_info=True
    )

    num_classes = train_dataset.get_num_classes()
    print(f"\nClases (P limitadas por Set) en train: {num_classes}")
    
    # Muestreador PK balanceado
    num_instances = config.NUM_INSTANCES
    sampler = RandomIdentitySampler(train_dataset, batch_size=config.BATCH_SIZE, num_instances=num_instances)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Inicialización de Arquitectura Temporal (HERENCIA CLÁSICA SUPERVISADA - VÍA B)
    dummy_backbone = GaitBackbone() 
    supervised_model = SupervisedReIDModel(backbone=dummy_backbone, num_classes=num_classes)
    
    sup_path = config.SUPERVISED_CHECKPOINT
    if os.path.exists(sup_path):
        checkpoint = torch.load(sup_path, map_location=config.DEVICE, weights_only=False)
        try:
            supervised_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\n[OK] EXITO: Pesos SUPERVISADOS (Fase 2) inyectados en el Hibrido.")
            print(f"    El modelo arranca heredando el conocimiento Clasificador + SSL previo.")
        except Exception as e:
            print(f"  [X] Error de topología al cargar Supervisado: {e}")
            sys.exit(1)
    else:
        print(f"\n[X] ERROR CRÍTICO: No se encontró el modelo supervisado en {sup_path}.")
        print("  La hipótesis actual exige herencia acumulada: SSL -> Supervisado -> Híbrido.")
        print("  Arranque abortado para evitar invalidación algorítmica. Ejecuta 'python train_supervised.py' primero.\n")
        sys.exit(1)

    model = HybridGaitModel(supervised_model).to(config.DEVICE)

    # 3. Configuración del Optimizer Diferencial Dinámico
    spatial_params, head_params = [], []
    for name, param in model.named_parameters():
        if 'encoder' in name or 'spatial' in name:
            spatial_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = optim.Adam([
        {'params': spatial_params, 'lr': config.HYBRID_LEARNING_RATE * 0.1}, 
        {'params': head_params, 'lr': config.HYBRID_LEARNING_RATE}
    ], weight_decay=1e-4)

    # 4. Funciones de Pérdida
    criterion_ce = nn.CrossEntropyLoss()
    if config.USE_CIRCLE_LOSS:
        criterion_triplet = CircleLoss(m=0.25, gamma=80)
        print("[*] Pérdida métrica configurada: Circle Loss (m=0.25, gamma=80)")
    else:
        criterion_triplet = TripletLoss(margin=config.HYBRID_MARGIN)
        print(f"[*] Pérdida métrica configurada: Triplet Loss (margin={config.HYBRID_MARGIN})")

    best_map = 0.0

    print("\n--- COMENZANDO CICLO DE ENTRENAMIENTO (OPEN-SET ACTIVADO) ---")
    for epoch in range(1, config.HYBRID_EPOCHS + 1):
        model.train()
        train_loss, ce_loss_sum, trip_loss_sum, hard_triplets_sum = 0, 0, 0, 0
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            logits, embeddings = model(images)
            loss_ce = criterion_ce(logits, labels)
            loss_trip, hard_trip_frac = criterion_triplet(embeddings, labels)
            
            loss = loss_ce + (config.HYBRID_TRIPLET_WEIGHT * loss_trip)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            ce_loss_sum += loss_ce.item()
            trip_loss_sum += loss_trip.item()
            hard_triplets_sum += hard_trip_frac
        
        avg_loss = train_loss / len(train_loader)
        avg_ce = ce_loss_sum / len(train_loader)
        avg_trip = trip_loss_sum / len(train_loader)
        avg_hard = (hard_triplets_sum / len(train_loader)) * 100
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:2d}/{config.HYBRID_EPOCHS} | L_CE: {avg_ce:.3f} | L_Trip: {avg_trip:.3f} | Hard Triplets: {avg_hard:.1f}% | Time: {epoch_time:.1f}s")
        
        # Validación Asíncrona Rápida (Mini-Val OOS)
        if epoch % 5 == 0 or epoch == config.HYBRID_EPOCHS:
            model.eval()
            all_embeds, all_labels, all_conds = [], [], []
            with torch.no_grad():
                for v_data in val_loader:
                    v_img, v_lbl, v_cond, _, _ = v_data
                    v_img = v_img.to(config.DEVICE)
                    _, v_emb = model(v_img)
                    all_embeds.append(v_emb.cpu())
                    all_labels.extend(v_lbl.tolist())
                    all_conds.extend(v_cond)
                    
            if len(all_embeds) > 0:
                all_embeds = torch.cat(all_embeds, dim=0)
                all_labels = np.array(all_labels)
                all_conds = np.array(all_conds)
                
                # Partición simple
                g_mask = np.isin(all_conds, ['nm-01', 'nm-02', 'nm-03', 'nm-04'])
                q_mask = ~g_mask
                
                g_e, g_l = all_embeds[g_mask], all_labels[g_mask]
                q_e, q_l = all_embeds[q_mask], all_labels[q_mask]
                
                if len(q_e) > 0 and len(g_e) > 0:
                    dist_m = torch.cdist(q_e, g_e, p=2)
                    v_r1, v_map = compute_mini_metrics(dist_m, q_l, g_l)
                    print(f"  -> Mini-Val Signal | Rank-1: {v_r1:.1f}% | mAP: {v_map:.1f}%")
                    
                    if v_map > best_map:
                        best_map = v_map
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_map': best_map,
                            'num_classes': num_classes,
                        }, config.HYBRID_CHECKPOINT)

    print(f"\n[+] ENTRENAMIENTO HÍBRIDO PRO FINALIZADO. Mejor mAP en Señal Temprana: {best_map:.2f}%")
    if "--no-eval" not in sys.argv:
        print(f"[*] Lanzando Evaluación Exhaustiva (Matriz 11x3) automáticamente...")
        import subprocess
        subprocess.run([sys.executable, "evaluate.py"])

if __name__ == "__main__":
    settings.check_paths()
    train_hybrid(settings)
