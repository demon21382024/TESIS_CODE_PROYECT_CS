# configs/settings.py
import torch
import os

# --- Configuración General ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (64, 64)
BATCH_SIZE = 64

# --- Ruta al Dataset ---
# ¡IMPORTANTE! Es mejor definir la ruta aquí o pasarla como argumento al script.
# Evita rutas absolutas (hardcoded) si es posible.
# Por ejemplo, podrías establecer una variable de entorno.
ROOT_PATH = "C:/output"

# --- Configuración SSL (Self-Supervised Learning) ---
SSL_EPOCHS = 80
SSL_LEARNING_RATE = 3e-4
SSL_TEMPERATURE = 0.07
SSL_CHECKPOINT = "models/backbone_ssl_best.pth"

# Configuración del subset para SSL
SSL_USE_SUBSET = True
SSL_SUBSET_SUBJECTS = 90
SSL_SUBSET_CONDITIONS = ['nm-01','nm-02','nm-03','nm-04','bg-01','cl-01']
SSL_SUBSET_ANGLES = ['000', '054', '090', '126', '180']
SSL_SUBSET_FRAMES_PER_SEQ = 24

# --- Configuración Supervisada ---
SUPERVISED_EPOCHS = 60
SUPERVISED_LEARNING_RATE = 1e-4  # Más bajo que SSL (fine-tuning)
SUPERVISED_MARGIN = 0.3  # Para triplet loss
SUPERVISED_CHECKPOINT = "models/supervised_model.pth"

# Configuración del subset para Supervisado
SUPERVISED_USE_SUBSET = True
SUPERVISED_SUBSET_TRAIN_SUBJECTS = 20
SUPERVISED_SUBSET_VAL_SUBJECTS = 10
SUPERVISED_SUBSET_TEST_SUBJECTS = 10
SUPERVISED_SUBSET_FRAMES_PER_SEQ = 20

"""
SUPERVISED_CONFIG = {
    'conditions': ['nm-01','nm-02','nm-03','nm-04'],
    'angles': ['000','054','090'],
    'train_range': (0, SUPERVISED_SUBSET_TRAIN_SUBJECTS if SUPERVISED_USE_SUBSET else 74),
    'val_range': (74, 74 + SUPERVISED_SUBSET_VAL_SUBJECTS if SUPERVISED_USE_SUBSET else 99),
    'test_range': (99, 99 + SUPERVISED_SUBSET_TEST_SUBJECTS if SUPERVISED_USE_SUBSET else 124),
}
"""

SUPERVISED_CONFIG = {
    'conditions': ['nm-01','nm-02','nm-03','nm-04'],
    'angles': ['000','054','090'],
    'train_range': (0, SUPERVISED_SUBSET_TRAIN_SUBJECTS),
    'val_range': (
        SUPERVISED_SUBSET_TRAIN_SUBJECTS,
        SUPERVISED_SUBSET_TRAIN_SUBJECTS + SUPERVISED_SUBSET_VAL_SUBJECTS
    ),
    'test_range': (
        SUPERVISED_SUBSET_TRAIN_SUBJECTS + SUPERVISED_SUBSET_VAL_SUBJECTS,
        SUPERVISED_SUBSET_TRAIN_SUBJECTS +
        SUPERVISED_SUBSET_VAL_SUBJECTS +
        SUPERVISED_SUBSET_TEST_SUBJECTS
    ),
}

# --- Verificación de Directorios ---
def check_paths():
    if not os.path.exists(ROOT_PATH):
        raise FileNotFoundError(f"La ruta del dataset no existe: {ROOT_PATH}")
    
    # Crear carpeta de modelos si no existe
    os.makedirs("models", exist_ok=True)
    
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset Root: {ROOT_PATH}")

# --- Configuración Híbrida (Supervisada + Triplet) ---
HYBRID_EPOCHS = 20
HYBRID_LEARNING_RATE = 1e-4  # Tasa de aprendizaje muy baja para fine-tuning
HYBRID_TRIPLET_WEIGHT = 1.5  # Peso para balancear las dos pérdidas
HYBRID_MARGIN = 0.5          # Margen para TripletLoss (mismo que SUPERVISED_MARGIN)
HYBRID_CHECKPOINT = "models/hybrid_model_best.pth"