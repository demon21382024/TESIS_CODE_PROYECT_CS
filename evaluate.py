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
from src.losses import TripletLoss 

# =======================================================
# 1. FUNCIONES AUXILIARES DE EXTRACCIÓN Y METRICAS
# =======================================================

def extract_features(data_loader, model, device):
    """Extrae embeddings y etiquetas del modelo en modo evaluación."""
    model.eval()
    all_embeddings = []
    all_labels = []
    # También necesitamos un identificador de muestra para manejar el trivial match
    all_indices = []
    index_counter = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            # El HybridGaitModel devuelve (logits, embeddings)
            _, embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            
            # Generar identificadores únicos para cada muestra
            batch_size = images.size(0)
            all_indices.extend(range(index_counter, index_counter + batch_size))
            index_counter += batch_size

    # Concatenar todos los lotes
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    indices = np.array(all_indices)
    
    return embeddings, labels, indices

def compute_reid_metrics(dist_matrix, query_labels, gallery_labels, query_indices=None, gallery_indices=None):
    """
    Calcula Rank-1 Accuracy y mAP (Mean Average Precision) para Re-ID, 
    manejando la exclusión de la coincidencia trivial.
    """
    dist_matrix = dist_matrix.cpu().numpy()
    query_labels = query_labels.cpu().numpy()
    gallery_labels = gallery_labels.cpu().numpy()

    CMC = [] 
    AP_list = [] 

    num_queries = len(query_labels)
    
    # -------------------------------------------------------------
    # 🛠️ CORRECCIÓN CLAVE: Crear máscara para excluir el trivial match
    # -------------------------------------------------------------
    # Se inicializa una máscara de exclusión con False (nada excluido inicialmente)
    mask_exclusion = np.zeros_like(dist_matrix, dtype=bool)

    if query_indices is not None and gallery_indices is not None and len(query_indices) == len(gallery_indices):
        # Si Query y Gallery son el mismo conjunto (Q=G), excluimos Q[i] vs G[i]
        is_same_set = (query_indices.min() == gallery_indices.min()) and (query_indices.max() == gallery_indices.max())
        
        if is_same_set:
            # Si el Query y Gallery son el mismo conjunto (0-N), la coincidencia trivial es donde i=j
            # Creamos una matriz identidad (True donde i=j)
            is_same_instance = (query_indices[:, None] == gallery_indices[None, :])
            mask_exclusion = is_same_instance
    # -------------------------------------------------------------

    # Iterar sobre las consultas
    for i in range(num_queries):
        q_label = query_labels[i]
        
        # 1. Aplicar Exclusión: Ignoramos la coincidencia trivial, si existe.
        # Definimos los resultados válidos: misma etiqueta Y no es la misma instancia (exclusión).
        is_relevant_match = (gallery_labels == q_label) 
        
        # Copiamos las distancias y aplicamos una penalización infinita a las exclusiones
        dists_i = dist_matrix[i, :].copy()
        
        # Excluir: Penalizar las instancias triviales o no válidas (distancia infinita)
        dists_i[mask_exclusion[i, :]] = np.inf
        
        # Excluir las distancias donde la etiqueta no coincide y no son relevantes
        # No queremos que se considere un Rank si la etiqueta no coincide (ya manejado por np.argsort)
        
        # 2. Ordenar por Distancia
        sorted_indices = np.argsort(dists_i)
        sorted_gallery_labels = gallery_labels[sorted_indices]
        
        # 3. Identificar las coincidencias correctas (después de la exclusión)
        # Esto nos da una lista de 1s y 0s para los resultados ordenados.
        matches = (sorted_gallery_labels == q_label).astype(np.int32)
        
        # 4. Excluir las coincidencias que tienen la misma etiqueta pero se penalizaron.
        # Esto es necesario si se usa un protocolo de prueba más estricto que Q=G.
        # Sin embargo, dado que penalizamos con inf, la clasificación las mueve al final.

        # ==================== RANK-k (CMC) ====================
        # El Rank-1 es ahora el primer resultado que no es la misma imagen.
        CMC.append(matches[0]) 

        # ==================== mAP (AP) ====================
        # Solo consideramos coincidencias válidas (donde la etiqueta coincide)
        valid_matches = matches 
        num_true_positives = np.sum(valid_matches)
        
        if num_true_positives == 0:
            continue
            
        num_hits = 0
        sum_precisions = 0
        
        for j, match in enumerate(valid_matches):
            if match == 1:
                num_hits += 1
                precision = num_hits / (j + 1) 
                sum_precisions += precision
        
        AP = sum_precisions / num_true_positives
        AP_list.append(AP)

    # Resultados finales
    rank1 = np.mean(CMC) * 100
    mAP = np.mean(AP_list) * 100 if AP_list else 0.0

    return rank1, mAP

# =======================================================
# 2. FUNCIÓN PRINCIPAL DE TESTING (Modificada)
# =======================================================

def test_reid(config):
    print("\n--- INICIANDO EVALUACIÓN DE RE-IDENTIFICATION ---")
    
    # 1. Cargar el Modelo Híbrido Final
    if not os.path.exists(config.HYBRID_CHECKPOINT):
        print(f"ERROR: No se encontró el checkpoint híbrido: {config.HYBRID_CHECKPOINT}")
        return
    
    checkpoint = torch.load(config.HYBRID_CHECKPOINT, map_location=config.DEVICE)
    
    dummy_backbone = GaitBackbone()
    dummy_supervised_model = SupervisedReIDModel(
        backbone=dummy_backbone,
        num_classes=checkpoint['num_classes']
    )
    
    model = HybridGaitModel(dummy_supervised_model).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Modelo Híbrido cargado desde: {config.HYBRID_CHECKPOINT}")
    print(f"  (Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")

    # 2. Cargar Dataset de Prueba
    print("\n--- Preparando Dataset de Prueba (Query/Gallery) ---")
    
    test_dataset = CASIAB_Supervised(
        root_path=config.ROOT_PATH,
        subject_range=config.SUPERVISED_CONFIG['test_range'],
        conditions=config.SUPERVISED_CONFIG['conditions'],
        angles=config.SUPERVISED_CONFIG['angles'],
        frames_per_seq=config.SUPERVISED_SUBSET_FRAMES_PER_SEQ,
        img_size=config.IMG_SIZE,
        augment=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 3. Extracción de Embeddings, Etiquetas e Índices de Muestra Únicos
    #    Usamos el mismo conjunto para Query y Gallery, por lo que tenemos que penalizar el trivial match.
    all_embeddings, all_labels, all_indices = extract_features(test_loader, model, config.DEVICE)
    
    query_embeddings = all_embeddings
    query_labels = all_labels
    query_indices = all_indices # ID único de la muestra (0, 1, 2, ...)
    
    gallery_embeddings = all_embeddings 
    gallery_labels = all_labels
    gallery_indices = all_indices
    
    print(f"  Query Samples: {len(query_embeddings)}, Gallery Samples: {len(gallery_embeddings)}")

    # 4. Cálculo de Matriz de Distancia
    print("\n--- Calculando Matriz de Distancia y Métricas (Exclusión Trivial Match) ---")
    dist_matrix = torch.cdist(query_embeddings, gallery_embeddings, p=2) 
    
    # 5. Cálculo de Métricas Finales
    start_time = time.time()
    rank1, mAP = compute_reid_metrics(
        dist_matrix, 
        query_labels, 
        gallery_labels, 
        query_indices=query_indices,
        gallery_indices=gallery_indices
    )
    end_time = time.time()
    
    print(f"Tiempo de cálculo de métricas: {end_time - start_time:.2f}s")
    
    print("\n=================================================")
    print(f"RESULTADOS DE EVALUACIÓN FINAL (CORREGIDOS):")
    print(f"Rank-1 Accuracy: {rank1:.2f}%")
    print(f"mAP (Mean Average Precision): {mAP:.2f}%")
    print("=================================================")

    return rank1, mAP

if __name__ == "__main__":
    # Necesitas que esta función exista en configs/settings.py
    settings.check_paths()
    test_reid(settings)