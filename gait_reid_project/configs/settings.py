# configs/settings.py
import torch
import os

# --- Configuración General ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32      # Protegiendo agresivamente la VRAM (RTX 3060 Ti)
NUM_INSTANCES = 4    # Para PKSampler (K secuencias por identidad)

# --- Ruta al Dataset ---
ROOT_PATH = "C:/output"

# --- Configuración SSL (Self-Supervised Learning) ---
SSL_EPOCHS = 40
SSL_LEARNING_RATE = 3e-4
SSL_TEMPERATURE = 0.07
SSL_CHECKPOINT = "models/backbone_ssl_best.pth"

# Configuración del subset para SSL
SSL_USE_SUBSET = True # Limita la ingesta para evitar fuga de los 50 de prueba
SSL_SUBSET_SUBJECTS = 74 # Sujetos 001 al 074 estrictamente
SSL_SUBSET_CONDITIONS = ['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02']
SSL_SUBSET_ANGLES = ['000','018','036','054','072','090','108','126','144','162','180']
SSL_SUBSET_FRAMES_PER_SEQ = 24

# --- Configuración Supervisada ---
SUPERVISED_EPOCHS = 40
SUPERVISED_LEARNING_RATE = 1e-4
SUPERVISED_MARGIN = 0.3
SUPERVISED_CHECKPOINT = "models/supervised_model.pth"

# Configuración del subset para Supervisado
SUPERVISED_USE_SUBSET = True

# --- Configuración Oficial Open-Set CASIA-B ---
SUPERVISED_SUBSET_TRAIN_SUBJECTS = 74  # 001 al 074
SUPERVISED_SUBSET_VAL_SUBJECTS = 10     
SUPERVISED_SUBSET_TEST_SUBJECTS = 50   # 075 al 124
SUPERVISED_SUBSET_FRAMES_PER_SEQ = 24  # T>=24

# Todas las condiciones y todos los ángulos disponibles en CASIA-B
SUPERVISED_CONFIG = {
    'conditions': ['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02'],
    'angles': ['000','018','036','054','072','090','108','126','144','162','180'],
    'train_range': (0, 74),   # 74 sujetos vistos
    'val_range': (64, 74),    # Mini-Val sobre los últimos 10 de Train para rastreo de convergencia
    'test_range': (74, 124),  # 50 Sujetos COMPLETAMENTE DESCONOCIDOS (Evaluación Final Open-Set)
}

# --- Verificación de Directorios ---
def check_paths():
    if not os.path.exists(ROOT_PATH):
        raise FileNotFoundError(f"La ruta del dataset no existe: {ROOT_PATH}")
    
    os.makedirs("models", exist_ok=True)
    
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset Root: {ROOT_PATH}")

# --- Configuración Híbrida ---
HYBRID_EPOCHS = 40
HYBRID_LEARNING_RATE = 1e-4
HYBRID_TRIPLET_WEIGHT = 0.5
HYBRID_MARGIN = 0.3
HYBRID_CHECKPOINT = "models/hybrid_model_final_PRO.pth"