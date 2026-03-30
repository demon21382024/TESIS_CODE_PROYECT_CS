# src/data.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F_vision
from PIL import Image
from pathlib import Path
import numpy as np
import random

def center_of_mass_align(img_tensor):
    """
    Centra la silueta binaria basada en su Centro de Masa (CoM).
    img_tensor: Tensor de forma (C, H, W). Se asume fondo negro (0) y silueta blanca (>0).
    """
    assert img_tensor.dim() == 3 and img_tensor.size(0) == 1
    # Convertimos a formato numpy para cálculos rápidos
    np_img = img_tensor.squeeze().cpu().numpy()
    
    # Calcular centro de masa
    y_coords, x_coords = np.nonzero(np_img > 0.1) # Umbral de activación para siluetas continuas
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        return img_tensor # Si la silueta está vacía (ruido extremo / frame negro), devolvemos original
        
    cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
    
    # Centro geométrico deseado
    H, W = np_img.shape
    center_y, center_x = H // 2, W // 2
    
    # Desplazamientos geométricos (Alineamiento Espacial)
    dy = center_y - cy
    dx = center_x - cx
    
    # La afín traslada usando dx, dy
    translated = F_vision.affine(img_tensor, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0], interpolation=F_vision.InterpolationMode.BILINEAR)
    
    return translated

class CASIAB_SSL(Dataset):
    """
    Data structure para InfoNCE Temporal: Devuelve DOS sub-secuencias volumétricas del MISMO video.
    """
    def __init__(self, root_path, img_size=(64, 64), use_subset=True, subset_subjects=10, 
                 subset_conditions=None, subset_angles=None, seq_len=15):
        
        self.root = Path(root_path)
        self.seq_len = seq_len
        self.img_size = img_size
        self.sequences = [] # Array principal que referencia secuencias completas, no frames
        
        all_subjects = sorted([d for d in self.root.iterdir() if d.is_dir()])
        subjects = all_subjects[:subset_subjects] if use_subset else all_subjects
        
        print(f"\n[Dataset SSL Volumétrico] Cargando desde: {root_path}")
        
        for subject_dir in subjects:
            for condition_dir in subject_dir.iterdir():
                if not condition_dir.is_dir(): continue
                if subset_conditions and condition_dir.name not in subset_conditions: continue
                    
                for angle_dir in condition_dir.iterdir():
                    if not angle_dir.is_dir(): continue
                    if subset_angles and angle_dir.name not in subset_angles: continue
                    
                    frames = sorted([f for f in angle_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
                    
                    # Filtramos por validéz temporal (ignorando caminatas super truncadas)
                    if len(frames) >= self.seq_len:
                        self.sequences.append(frames)
        
        print(f"Secuencias Volumétricas (>= {self.seq_len} frames): {len(self.sequences)}")
        
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ])
        
    def _sample_frames(self, frames, start_idx):
        sampled_frames = []
        for i in range(self.seq_len):
            idx = min(start_idx + i, len(frames) - 1)
            img = Image.open(frames[idx]).convert("L")
            img_t = self.base_transform(img)
            # PASO ESENCIAL: Alineamiento CoM por frame para destruir oscilaciones de segmentación 2D.
            img_centered = center_of_mass_align(img_t)
            sampled_frames.append(img_centered)
        # Apilamos en dimensión 0, produciendo Tensor(T, C, H, W)
        return torch.stack(sampled_frames, dim=0) 

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        frames = self.sequences[idx]
        total_len = len(frames)
        
        # InfoNCE Contrastivo Temporal:
        # Extraer dos muestras temporales (V1 y V2) cortadas en lugares distintos
        # de la misma trayectoria para promover invarianza de ciclo de marcha.
        if total_len > self.seq_len:
            start1 = random.randint(0, total_len - self.seq_len)
            start2 = random.randint(0, total_len - self.seq_len)
        else:
            start1 = 0
            start2 = 0
            
        view1 = self._sample_frames(frames, start1)
        view2 = self._sample_frames(frames, start2)
        
        return view1, view2

class CASIAB_Supervised(Dataset):
    """
    Data structure PKSampler-Friendly: Devuelve Tensor Volumétrico [T, C, H, W] y etiqueta escalar.
    """
    def __init__(self, root_path, subject_range, conditions, angles=None, 
                 seq_len=15, img_size=(64, 64), augment=False, return_info=False):
        
        self.root = Path(root_path)
        self.seq_len = seq_len
        self.augment = augment
        self.return_info = return_info
        self.samples = [] 
        self.subject_to_label = {}
        
        all_subjects = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        start_idx, end_idx = subject_range
        subjects = all_subjects[start_idx:end_idx]
        
        print(f"\n[Dataset Supervised Volumétrico] Rango: {start_idx+1:03d}-{end_idx:03d} ({len(subjects)} sujetos)")
        if angles: print(f"Ángulos: {angles}")
        
        for label_id, subject_name in enumerate(subjects):
            self.subject_to_label[subject_name] = label_id
            
        for subject_name in subjects:
            subject_dir = self.root / subject_name
            label_id = self.subject_to_label[subject_name]
            
            for condition_dir in subject_dir.iterdir():
                if not condition_dir.is_dir() or condition_dir.name not in conditions: continue
                
                for angle_dir in condition_dir.iterdir():
                    if not angle_dir.is_dir(): continue
                    if angles and angle_dir.name not in angles: continue
                    
                    frames = sorted([f for f in angle_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
                    
                    # Añadir TODA la secuencia secuencialmente a muestras de Identidad P
                    if len(frames) >= self.seq_len:
                        self.samples.append({
                            'frames': frames, 
                            'label': label_id, 
                            'subject': subject_name,
                            'condition': condition_dir.name, 
                            'angle': angle_dir.name
                        })
                        
        print(f"  Secuencias Totales de P(Id) disponibles: {len(self.samples)}, Clases Aprobadas: {len(self.subject_to_label)}")
        
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = sample['frames']
        total_len = len(frames)
        
        # En Entrenamiento (augment=True): Extraemos fase aleatoria.
        # En Validación/Prueba: Extraemos el medio de la trayectoria para consistencia de evaluación mAP.
        if self.augment:
            start_idx = random.randint(0, total_len - self.seq_len)
        else:
            start_idx = (total_len - self.seq_len) // 2 
            
        sampled_tensors = []
        for i in range(self.seq_len):
            idx_f = min(start_idx + i, total_len - 1)
            img = Image.open(frames[idx_f]).convert("L")
            img_t = self.base_transform(img)
            
            # Algoritmo Base de Invarianza Bidimensional
            img_aligned = center_of_mass_align(img_t)
            sampled_tensors.append(img_aligned)
            
        seq_tensor = torch.stack(sampled_tensors, dim=0) # Generando Volumetría [T, C, H, W]
        
        # NOTA: Ahora el Dataloader devuelve diccionarios para que funcionen elegantemente 
        # con el PKSampler, garantizando la meta-data de identidad estricta.
        if self.return_info:
            return seq_tensor, sample['label'], sample['condition'], sample['angle'], sample['subject']
        return seq_tensor, sample['label']
    
    def get_num_classes(self):
        return len(self.subject_to_label)