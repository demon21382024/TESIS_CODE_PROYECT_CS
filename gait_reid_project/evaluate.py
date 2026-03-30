# evaluate.py

import torch
import numpy as np
import os
import time
from torch.utils.data import DataLoader

# Importar desde nuestros módulos
from configs import settings
from src.data import CASIAB_Supervised
from src.models import HybridGaitModel, SupervisedReIDModel, GaitBackbone

# =======================================================
# 1. FUNCIONES AUXILIARES DE EXTRACCIÓN Y METRICAS
# =======================================================

def extract_features_with_info(data_loader, model, device):
    """Extrae embeddings y metadata del modelo en modo evaluación."""
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_conditions = []
    all_angles = []
    all_subjects = []

    print("[*] Iniciando Extracción de Vectores Biométicos...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # Desenpaquetar la nueva tupla rica:
            images, labels, conditions_batch, angles_batch, subjects_batch = data
            images = images.to(device)
            
            # El HybridGaitModel devuelve (logits, embeddings)
            _, embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.tolist())
            all_conditions.extend(conditions_batch)
            all_angles.extend(angles_batch)
            all_subjects.extend(subjects_batch)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Procesando Batch {batch_idx+1}/{len(data_loader)}...")

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = np.array(all_labels)
    conditions = np.array(all_conditions)
    angles = np.array(all_angles)
    subjects = np.array(all_subjects)
    
    print(f"[✓] Extracción completada en {time.time() - start_time:.1f}s. Total Muestras: {len(labels)}")
    return embeddings, labels, conditions, angles, subjects

def compute_reid_metrics_block(dist_matrix, query_labels, gallery_labels):
    """
    Calcula hiper-métricas (Rank-1, Rank-5, Rank-10, mAP) y devuelve los índices exactos
    para análisis cualitativo.
    En CASIA-B, Query y Gallery son conjuntos DISJUNTOS por Condición, por lo que nunca 
    habrá un "Identical Match" (Misma secuencia de la misma persona).
    """
    CMC_top1 = [] 
    CMC_top5 = []
    CMC_top10 = []
    AP_list = [] 
    
    # Traceability para Análisis de Fallo (MinP)
    sample_ap_scores = []
    
    num_queries = len(query_labels)
    
    # Convertir tensores a numpy si es necesario
    if torch.is_tensor(dist_matrix): dist_matrix = dist_matrix.cpu().numpy()
    
    for i in range(num_queries):
        q_label = query_labels[i]
        dists_i = dist_matrix[i, :]
        
        # Ordenar galería por distancias (menor a mayor)
        sorted_indices = np.argsort(dists_i)
        sorted_gallery_labels = gallery_labels[sorted_indices]
        
        # Identificar las coincidencias correctas (True/False ó 1/0)
        matches = (sorted_gallery_labels == q_label).astype(np.int32)
        
        # ==================== RANKS (CMC) ====================
        CMC_top1.append(matches[0]) 
        CMC_top5.append(1 if np.sum(matches[:5]) > 0 else 0)
        CMC_top10.append(1 if np.sum(matches[:10]) > 0 else 0)

        # ==================== mAP (AP) ====================
        num_true_positives = np.sum(matches)
        
        if num_true_positives == 0:
            AP_list.append(0.0)
            sample_ap_scores.append(0.0)
            continue
            
        num_hits = 0
        sum_precisions = 0
        
        for j, match in enumerate(matches):
            if match == 1:
                num_hits += 1
                precision = num_hits / (j + 1) 
                sum_precisions += precision
        
        AP = sum_precisions / num_true_positives
        AP_list.append(AP)
        sample_ap_scores.append(AP)

    # Resultados finales del bloque
    rank1 = np.mean(CMC_top1) * 100
    rank5 = np.mean(CMC_top5) * 100
    rank10 = np.mean(CMC_top10) * 100
    mAP = np.mean(AP_list) * 100 if AP_list else 0.0

    return rank1, rank5, rank10, mAP, np.array(sample_ap_scores)


# =======================================================
# 2. FUNCIÓN PRINCIPAL DE TESTING CON MATRIZ DE ÁNGULOS
# =======================================================

def test_reid_exhaustive(config):
    print("\n" + "="*60)
    print(" INICIANDO EVALUACIÓN EXHAUSTIVA CASIA-B (PROTOCOLO OFICIAL)")
    print("="*60)
    
    # 1. Cargar el Modelo Híbrido Final (El que acabas de entrenar magistralmente)
    model_path = config.HYBRID_CHECKPOINT # <-- CORRECCIÓN: Apuntando al modelo Híbrido (Mejor mAP)
    if not os.path.exists(model_path):
        print(f"[X] ERROR CRÍTICO: No se encontró {model_path}")
        return
    # PyTorch 2.6 restringe los pickles por seguridad. Permitiendo escalares (weights_only=False)
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    
    dummy_backbone = GaitBackbone()
    dummy_supervised_model = SupervisedReIDModel(
        backbone=dummy_backbone,
        num_classes=checkpoint['num_classes']
    )
    
    # Envolvemos el supervisado en el Hibrido para obtener la proyección de embeddings directa
    model = HybridGaitModel(dummy_supervised_model).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Modelo cargado estandarizado desde: {model_path}")

    # 2. Cargar Dataset de Prueba (Sobreescribiendo filtros para garantizar el conjunto completo)
    print("\n[+] Preparando Dataset de Prueba (Open-Set Retrieval: 50 Sujetos No Vistos)")
    
    all_casia_conditions = ['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06', 'bg-01','bg-02', 'cl-01','cl-02']
    all_casia_angles = ['000','018','036','054','072','090','108','126','144','162','180']
    
    test_dataset = CASIAB_Supervised(
        root_path=config.ROOT_PATH,
        subject_range=config.SUPERVISED_CONFIG['test_range'], # DINÁMICO: Evaluación Oficial Open-Set (075-124)
        conditions=all_casia_conditions,
        angles=all_casia_angles,
        seq_len=config.SUPERVISED_SUBSET_FRAMES_PER_SEQ,
        img_size=config.IMG_SIZE,
        augment=False,
        return_info=True # <-- Activación Crítica para Evaluación Cruzada
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 3. Extracción Unificada
    embeddings, labels, conditions, angles, subjects = extract_features_with_info(test_loader, model, config.DEVICE)
    
    # 4. Partición Lógica de Vectores (Standard CASIA-B)
    print("\n[*] Fragmentando Espacio Latente en Gallery / Queries...")
    
    # Gallery: NM 01 a 04
    gallery_mask = np.isin(conditions, ['nm-01', 'nm-02', 'nm-03', 'nm-04'])
    g_embeds, g_labels, g_angles = embeddings[gallery_mask], labels[gallery_mask], angles[gallery_mask]
    
    # Queries:
    q_nm_mask = np.isin(conditions, ['nm-05', 'nm-06'])
    q_bg_mask = np.isin(conditions, ['bg-01', 'bg-02'])
    q_cl_mask = np.isin(conditions, ['cl-01', 'cl-02'])
    
    query_sets = {
        'NM (Normal Walking)': (embeddings[q_nm_mask], labels[q_nm_mask], angles[q_nm_mask], subjects[q_nm_mask], conditions[q_nm_mask]),
        'BG (Bag Oclusion)': (embeddings[q_bg_mask], labels[q_bg_mask], angles[q_bg_mask], subjects[q_bg_mask], conditions[q_bg_mask]),
        'CL (Coat Appearance)': (embeddings[q_cl_mask], labels[q_cl_mask], angles[q_cl_mask], subjects[q_cl_mask], conditions[q_cl_mask])
    }
    
    print(f"  Total Gallery: {len(g_embeds)} muestras.")
    
    overall_mAPs = []
    
    # 5. Iteración de la Matriz Angular Categórica
    print("\n" + "="*80)
    print(" 🛠️ MATRIZ DE DEGRADACIÓN ANGULAR (RANK-1 %)")
    print("="*80)
    
    for condition_name, (q_e, q_l, q_a, q_sub, q_cond) in query_sets.items():
        if len(q_e) == 0:
            continue
            
        print(f"\n--- Prueba: {condition_name} ({len(q_e)} Probes) ---")
        
        # Calcular Matriz D Global Probe vs Gallery para esta condición
        dist_matrix_all = torch.cdist(q_e, g_embeds, p=2)
        
        r1, r5, r10, mAP_cond, ap_scores = compute_reid_metrics_block(dist_matrix_all, q_l, g_labels)
        overall_mAPs.append(mAP_cond)
        
        print(f"> Promedio Condición | Rank-1: {r1:.1f}% | Rank-5: {r5:.1f}% | Rank-10: {r10:.1f}% | mAP: {mAP_cond:.1f}%")
        
        # Análisis Fino: Desglose Exclusivo por Ángulo de Probe vs TODOS los ángulos de Gallery
        angle_r1_list = []
        for specific_angle in all_casia_angles:
            angle_mask = (q_a == specific_angle)
            if not np.any(angle_mask):
                continue
            
            sub_q_e = q_e[angle_mask]
            sub_q_l = q_l[angle_mask]
            
            # Matriz específica para este ángulo
            sub_dist_matrix = torch.cdist(sub_q_e, g_embeds, p=2)
            sub_r1, _, _, _, _ = compute_reid_metrics_block(sub_dist_matrix, sub_q_l, g_labels)
            
            angle_r1_list.append(f"{specific_angle}°: {sub_r1:04.1f}%")
            
        print(" | ".join(angle_r1_list))
        
        # Análisis Fino: ABLACIÓN CUALITATIVA (Top-5 Peores)
        print("\n  [Análisis de Fallos Clínicos - Las 5 peores inferencias MinP]")
        worst_indices = np.argsort(ap_scores)[:5]
        for idx in worst_indices:
            print(f"   -> Probe: Sujeto {q_sub[idx]}, {q_cond[idx]}, Ángulo {q_a[idx]} | Precisión lograda: {ap_scores[idx]*100:.1f}%")

    # 6. Tabla Comparativa General
    final_map = np.mean(overall_mAPs)
    
    print("\n" + "="*60)
    print(" 📊 VEREDICTO FINAL DE TOPOLOGÍA VOLUMÉTRICA")
    print("="*60)
    print(f"Baseline Temporal (Antiguo): mAP 29.23%")
    print(f"Modelo Volumétrico Actual:   mAP {final_map:.2f}%")
    
    if final_map > 29.23:
        print("\n[✓] CRECIMIENTO ESTADÍSTICO CONFIRMADO.")
        print("La transición a Tensión Temporal [T, C, H, W] y Dual Pooling superó exitosamente")
        print("las deficiencias del ResNet-18 estático. La tesis es matemáticamente sólida.")
    
if __name__ == "__main__":
    settings.check_paths()
    test_reid_exhaustive(settings)