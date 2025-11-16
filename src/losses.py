# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """NT-Xent Loss (SimCLR)."""
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
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -1e9)
        
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        loss = self.criterion(sim_matrix, labels)
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        batch_size = embeddings.size(0)
        loss = 0.0
        num_valid = 0
        
        for i in range(batch_size):
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
            neg_mask = labels != labels[i]
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            hard_positive = dist_matrix[i][pos_mask].max()
            hard_negative = dist_matrix[i][neg_mask].min()
            
            loss += F.relu(hard_positive - hard_negative + self.margin)
            num_valid += 1
        
        return loss / max(num_valid, 1)