# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """NT-Xent Loss (SimCLR) para InfoNCE Temporal."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        
        z1 = z1.float()
        z2 = z2.float()
        
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Evitar match diagonal (con sigo mismo)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -1e9)
        
        # Etiquetas: V1 debe cruzar con V2 para la misma secuencia
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        loss = self.criterion(sim_matrix, labels)
        return loss

class TripletLoss(nn.Module):
    """
    Vectorized Batch Hard Triplet Loss (OPTMIZADA en CUDA).
    Devuelve la pérdida promedio y la Fracción de Hard Triplets activas
    para el Protocolo de Monitoreo de Señales Tempranas.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        # embeddings: Tensor [B, Features]
        # labels: Tensor [B]
        B = embeddings.size(0)
        device = embeddings.device
        
        # 1. Matriz Dinámica de Distancias Euclidias L2
        dist_matrix = torch.cdist(embeddings, embeddings, p=2) # [B, B]
        
        # 2. Máscaras Lógicas Vectorizadas
        labels_mat = labels.expand(B, B) # [B, B]
        
        # Máscara de Positivos (misma clase)
        is_pos = labels_mat.eq(labels_mat.t())
        
        # Máscara de Negativos (distinta clase)
        is_neg = labels_mat.ne(labels_mat.t())
        
        # Excluir la auto-referencia diagonal (Ancla == Positivo)
        identity_mask = torch.eye(B, dtype=torch.bool, device=device)
        is_pos = is_pos & ~identity_mask
        
        # 3. Minería Hardest Positive: Máxima distancia entre Ancla y todos sus Positivos
        # Se anula la distancia para no-positivos.
        dist_ap = dist_matrix * is_pos.float()
        hardest_positive_dist, _ = dist_ap.max(dim=1) # [B]
        
        # 4. Minería Hardest Negative: Mínima distancia entre Ancla y todos sus Negativos
        # Se infla a "infinito" la distancia para no-negativos.
        max_dist_val = dist_matrix.max() + 1e5
        dist_an = dist_matrix + (~is_neg).float() * max_dist_val
        hardest_negative_dist, _ = dist_an.min(dim=1) # [B]
        
        # 5. Cálculo Simultáneo de Pérdida
        loss_components = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        # 6. Monitor de Señal Temprana: Estadísticas de Triplets
        active_triplets = loss_components[loss_components > 0]
        num_active = active_triplets.size(0)
        fraction_active = float(num_active) / float(B)
        
        if num_active > 0:
            loss = active_triplets.mean()
        else:
            # Previene rupturas de la gráfica computacional PyTorch
            loss = loss_components.sum() * 0.0 
            
        return loss, fraction_active

class CircleLoss(nn.Module):
    """
    Circle Loss para aprendizaje métrico (pairwise similarity).
    Optimizado y vectorizado en PyTorch utilizando operaciones de tensores estables.
    """
    def __init__(self, m=0.25, gamma=80):
        super().__init__()
        self.m = m
        self.gamma = gamma
        
    def forward(self, embeddings, labels):
        # embeddings: [B, D] (ya normalizados L2)
        # labels: [B]
        B = embeddings.size(0)
        device = embeddings.device
        
        # 1. Matriz de similitud coseno (al estar normalizados L2, dot product == similitud coseno)
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        
        # 2. Máscaras de positivos y negativos
        labels_mat = labels.expand(B, B)
        is_pos = labels_mat.eq(labels_mat.t())
        is_neg = labels_mat.ne(labels_mat.t())
        
        # Excluir la auto-similitud diagonal
        identity_mask = torch.eye(B, dtype=torch.bool, device=device)
        is_pos = is_pos & ~identity_mask
        
        # 3. Parámetros de margen y pesos
        op = 1 + self.m
        on = -self.m
        dp = 1 - self.m
        dn = self.m
        
        # Calcular pesos alpha_p y alpha_n
        ap = torch.clamp(op - sim_matrix, min=0.0)
        an = torch.clamp(sim_matrix - on, min=0.0)
        
        # Calcular logits para positivos y negativos
        logit_p = -self.gamma * ap * (sim_matrix - dp)
        logit_n = self.gamma * an * (sim_matrix - dn)
        
        # Enmascarar elementos no válidos con un valor muy bajo para evitar influir en logsumexp
        logit_p = logit_p.masked_fill(~is_pos, -1e9)
        logit_n = logit_n.masked_fill(~is_neg, -1e9)
        
        # 4. Cálculo numéricamente estable con LogSumExp por fila
        logsum_p = torch.logsumexp(logit_p, dim=1)
        logsum_n = torch.logsumexp(logit_n, dim=1)
        
        # Usamos softplus(x) = log(1 + exp(x)) para máxima estabilidad numérica
        loss_components = F.softplus(logsum_p + logsum_n)
        
        # Filtrar filas que no tengan positivos o negativos (evitar pérdidas rotas en batches desbalanceados)
        has_pos = is_pos.sum(dim=1) > 0
        has_neg = is_neg.sum(dim=1) > 0
        valid_rows = has_pos & has_neg
        
        if valid_rows.sum() > 0:
            loss = loss_components[valid_rows].mean()
        else:
            loss = loss_components.sum() * 0.0
            
        return loss, 0.0