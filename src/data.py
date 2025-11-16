# src/data.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class CASIAB_SSL(Dataset):
    """
    Data structure: subject_id/condition(bg-01,nm-01,...)/angle(000,018,...)/frames
    """
    def __init__(self, root_path, img_size=(64, 64), use_subset=True, subset_subjects=10, 
                 subset_conditions=None, subset_angles=None, frames_per_seq=15):
        
        self.root = Path(root_path)
        self.images = []
        
        all_subjects = sorted([d for d in self.root.iterdir() if d.is_dir()])
        subjects = all_subjects[:subset_subjects] if use_subset else all_subjects
        
        print(f"\n[Dataset SSL] Cargando desde: {root_path}")
        print(f"Sujetos: {len(subjects)}/{len(all_subjects)}")
        if subset_conditions:
            print(f"Condiciones: {subset_conditions}")
        if subset_angles:
            print(f"Ángulos: {subset_angles}")
        
        total_sequences = 0
        for subject_dir in subjects:
            for condition_dir in subject_dir.iterdir():
                if not condition_dir.is_dir(): continue
                if subset_conditions and condition_dir.name not in subset_conditions: continue
                    
                for angle_dir in condition_dir.iterdir():
                    if not angle_dir.is_dir(): continue
                    if subset_angles and angle_dir.name not in subset_angles: continue
                    
                    total_sequences += 1
                    frames = sorted([f for f in angle_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
                    
                    if len(frames) > frames_per_seq:
                        step = len(frames) / frames_per_seq
                        frames = [frames[int(i * step)] for i in range(frames_per_seq)]
                    
                    self.images.extend(frames)
        
        print(f"Secuencias: {total_sequences}")
        print(f"Total frames: {len(self.images)}")
        
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ])
        
        self.ssl_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("L")
        img = self.base_transform(img)
        view1 = self.ssl_transform(img)
        view2 = self.ssl_transform(img)
        return view1, view2

class CASIAB_Supervised(Dataset):
    def __init__(self, root_path, subject_range, conditions, angles=None, 
                 frames_per_seq=20, img_size=(64, 64), augment=False):
        
        self.root = Path(root_path)
        self.augment = augment
        self.samples = []
        self.subject_to_label = {}
        
        all_subjects = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        start_idx, end_idx = subject_range
        subjects = all_subjects[start_idx:end_idx]
        
        print(f"\n[Dataset Supervised] Rango: {start_idx+1:03d}-{end_idx:03d} ({len(subjects)} sujetos)")
        print(f"Condiciones: {conditions}")
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
                    
                    if len(frames) > frames_per_seq:
                        step = len(frames) / frames_per_seq
                        frames = [frames[int(i * step)] for i in range(frames_per_seq)]
                    
                    for frame_path in frames:
                        self.samples.append({
                            'path': frame_path, 'label': label_id, 'subject': subject_name,
                            'condition': condition_dir.name, 'angle': angle_dir.name
                        })
        
        print(f"  Samples: {len(self.samples)}, Clases: {len(self.subject_to_label)}")
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(img_size, antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size, antialias=True),
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert("L")
        img = self.transform(img)
        return img, sample['label']
    
    def get_num_classes(self):
        return len(self.subject_to_label)