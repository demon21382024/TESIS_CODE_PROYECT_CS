# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaitBackbone(nn.Module):
    """Backbone usado para SSL (SimCLR)"""
    def __init__(self, embed_dim=256):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head para SimCLR
        self.projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        z = self.projection(x)
        z = F.normalize(z, dim=1)
        return z

class SupervisedReIDModel(nn.Module):
    """Modelo para Re-ID (clasificación) que usa el encoder del backbone"""
    def __init__(self, backbone, num_classes, freeze_backbone=False):
        super().__init__()
        
        # Copiamos las capas del encoder del backbone pre-entrenado
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.conv2,
            backbone.conv3,
            backbone.pool
        )
        
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("  ✓ Backbone congelado")
        
        # Capas de clasificación
        self.bn = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        features = self.bn(features)
        logits = self.classifier(features)
        return logits
    

# src/models.py (Añadir al final)

class HybridGaitModel(nn.Module):
    """
    Modelo Híbrido que usa el modelo supervisado pre-entrenado
    y devuelve tanto logits como embeddings para la pérdida dual.
    """
    def __init__(self, supervised_model: SupervisedReIDModel):
        super().__init__()
        
        # Heredamos el encoder y el clasificador del modelo supervisado
        self.encoder = supervised_model.encoder
        self.bn = supervised_model.bn
        self.classifier = supervised_model.classifier
    
    def forward(self, x):
        # 1. Obtener características (features)
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        # 2. Obtener embeddings (para TripletLoss)
        # Usamos las características después de Batch Norm, como en el modelo supervisado
        embeddings = self.bn(features)
        
        # 3. Obtener logits (para CrossEntropyLoss)
        logits = self.classifier(embeddings)
        
        # 4. Normalizar embeddings para TripletLoss (mejora la estabilidad)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return logits, normalized_embeddings