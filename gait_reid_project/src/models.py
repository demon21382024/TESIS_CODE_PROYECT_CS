# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TemporalPooling(nn.Module):
    """
    Realiza GAP y GMP a lo largo de la dimensión temporal T y los concatena.
    Entrada: (B, T, Features)
    Salida: (B, Features * 2)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x: [B, T, F]
        gap = torch.mean(x, dim=1) # [B, F] - Promedio Morfológico
        gmp = torch.max(x, dim=1)[0] # [B, F] - Amplitud Topológica Máxima
        return torch.cat([gap, gmp], dim=1) # [B, F*2]

class GaitBackbone(nn.Module):
    """Backbone usando ResNet-18 modificado para soportar volumetría temporal [B, T, C, H, W]"""
    def __init__(self, embed_dim=256):
        super().__init__()
        
        # Cargar ResNet-18 pre-entrenada
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modificar la primera capa para aceptar 1 canal (siluetas L)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # HPP: Spatial Average Pooling por franja horizontal (2 partes: Cuerpo Superior e Inferior)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 1))
        
        # Proyección HPP: Unificar el espacio latente de ambas franjas
        self.hpp_project = nn.Linear(2048, 1024)
        nn.init.normal_(self.hpp_project.weight.data, 0.0, 0.01)
        nn.init.constant_(self.hpp_project.bias.data, 0.0)
        
        # Pooling Dual a través de la secuencia en el tiempo (GAP + GMP)
        self.temporal_pool = TemporalPooling()
        
        # Capa de proyección InfoNCE (Dimensiones x2 por GAP+GMP = 512*2 = 1024)
        self.projection = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim)
        )
        
    def spatial_forward(self, x_fp):
        # Procesamiento Espacial Puro 2D
        x = self.conv1(x_fp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) # [B*T, 512, 2, 1]
        x = x.view(x.size(0), 512, 2) # [B*T, 512, 2]
        return x
        
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        
        # 1. Plegar el lote en una lista masiva transaccional [B*T, C, H, W]
        x_fold = x.view(B * T, C, H, W)
        
        # 2. Extracción espacial frame a frame (con HPP)
        features_2d = self.spatial_forward(x_fold) # [B*T, 512, 2]
        
        # 3. Restaurar dimensionalidad temporal [B, T, 512, 2]
        features_3d = features_2d.view(B, T, 512, 2) 
        
        # 4. Pooling Dual Temporal Integrado por parte:
        z_upper = self.temporal_pool(features_3d[:, :, :, 0]) # [B, 1024]
        z_lower = self.temporal_pool(features_3d[:, :, :, 1]) # [B, 1024]
        
        # 5. Concatenar ambas franjas:
        z_hpp = torch.cat([z_upper, z_lower], dim=1) # [B, 2048]
        
        # 6. Unificación a 1024 dimensiones
        temporal_embedding = self.hpp_project(z_hpp) # [B, 1024]
        
        # 7. Pipeline SSL InfoNCE
        z = self.projection(temporal_embedding)
        z = F.normalize(z, dim=1)
        return z

class SupervisedReIDModel(nn.Module):
    """Modelo para Re-ID (Clasificación/Métrica) que hereda la topología temporal"""
    def __init__(self, backbone, num_classes, freeze_backbone=False):
        super().__init__()
        
        # Conservamos únicamente las capas del decodificador espacial local con HPP
        self.spatial_encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool # AdaptiveAvgPool2d((2,1))
        )
        self.temporal_pool = backbone.temporal_pool
        self.hpp_project = backbone.hpp_project
        
        if freeze_backbone:
            for param in self.spatial_encoder.parameters(): param.requires_grad = False
            for param in self.temporal_pool.parameters(): param.requires_grad = False
            for param in self.hpp_project.parameters(): param.requires_grad = False
            print("  ✓ Backbone Espacial/Temporal/HPP congelado")
            
        feature_dim = 1024 # 512(GAP) + 512(GMP)
        
        # Cabeceras de Métrica
        self.bn = nn.BatchNorm1d(feature_dim)
        nn.init.normal_(self.bn.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn.bias.data, 0.0)
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(self.classifier.weight.data, 0.0, 0.001)
        nn.init.constant_(self.classifier.bias.data, 0.0)
    
    def forward(self, x):
        B, T, C, H, W = x.size()
        
        x_fold = x.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder(x_fold) # [B*T, 512, 2, 1]
        spatial_features = spatial_features.view(B * T, 512, 2)
        
        temporal_features = spatial_features.view(B, T, 512, 2) # [B, T, 512, 2]
        
        # Pooling Temporal Dual por parte
        z_upper = self.temporal_pool(temporal_features[:, :, :, 0]) # [B, 1024]
        z_lower = self.temporal_pool(temporal_features[:, :, :, 1]) # [B, 1024]
        
        z_hpp = torch.cat([z_upper, z_lower], dim=1) # [B, 2048]
        pooled_features = self.hpp_project(z_hpp) # [B, 1024]
        
        embeddings = self.bn(pooled_features)
        logits = self.classifier(embeddings)
        return logits

    def load_state_dict(self, state_dict, strict=True):
        # Si faltan las claves de hpp_project (checkpoint antiguo), cargar de forma parcial
        missing_keys = []
        for key in ["hpp_project.weight", "hpp_project.bias"]:
            if key not in state_dict:
                missing_keys.append(key)
        
        if len(missing_keys) > 0:
            print(f"\n  [!] Advertencia: Cargando checkpoint antiguo. Claves HPP faltantes: {missing_keys}.")
            print("      Inicializando hpp_project de forma aleatoria (se optimizará durante el entrenamiento).")
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

class HybridGaitModel(nn.Module):
    """
    Controlador Maestro de Topología de Secuencia que devuelve [Logits, Embeddings] 
    para la CrossEntropyLoss y la Vectorized Batch Hard Triplet Loss simultáneamente.
    """
    def __init__(self, supervised_model: SupervisedReIDModel):
        super().__init__()
        
        self.spatial_encoder = supervised_model.spatial_encoder
        self.temporal_pool = supervised_model.temporal_pool
        self.hpp_project = supervised_model.hpp_project
        self.bn = supervised_model.bn
        self.classifier = supervised_model.classifier
    
    def forward(self, x):
        B, T, C, H, W = x.size()
        
        x_fold = x.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder(x_fold) # [B*T, 512, 2, 1]
        spatial_features = spatial_features.view(B * T, 512, 2)
        
        temporal_features = spatial_features.view(B, T, 512, 2)
        
        # Pooling Temporal Dual por parte
        z_upper = self.temporal_pool(temporal_features[:, :, :, 0]) # [B, 1024]
        z_lower = self.temporal_pool(temporal_features[:, :, :, 1]) # [B, 1024]
        
        z_hpp = torch.cat([z_upper, z_lower], dim=1) # [B, 2048]
        pooled_features = self.hpp_project(z_hpp) # [B, 1024]
        
        embeddings = self.bn(pooled_features)
        logits = self.classifier(embeddings)
        
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return logits, normalized_embeddings

    def load_state_dict(self, state_dict, strict=True):
        # Si faltan las claves de hpp_project (checkpoint antiguo), cargar de forma parcial
        missing_keys = []
        for key in ["hpp_project.weight", "hpp_project.bias"]:
            if key not in state_dict:
                missing_keys.append(key)
        
        if len(missing_keys) > 0:
            print(f"\n  [!] Advertencia: Cargando checkpoint antiguo en Hybrid. Claves HPP faltantes: {missing_keys}.")
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)