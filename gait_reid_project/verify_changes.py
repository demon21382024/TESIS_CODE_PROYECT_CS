import torch
from src.models import GaitBackbone, SupervisedReIDModel
from src.samplers import RandomIdentitySampler
from configs import settings

def verify_model():
    print("Verifying Model Architecture...")
    try:
        backbone = GaitBackbone()
        model = SupervisedReIDModel(backbone, num_classes=10)
        x = torch.randn(2, 1, 128, 128)
        logits = model(x)
        print(f"  ✓ Model forward pass successful. Output shape: {logits.shape}")
        if logits.shape == (2, 10):
            print("  ✓ Output shape is correct.")
        else:
            print(f"  X Output shape mismatch. Expected (2, 10), got {logits.shape}")
    except Exception as e:
        print(f"  X Model verification failed: {e}")

def verify_sampler():
    print("\nVerifying RandomIdentitySampler...")
    # Mock data: 10 identities, 10 images each
    data_source = []
    for pid in range(10):
        for i in range(10):
            data_source.append((f"img_{pid}_{i}.png", pid))
            
    batch_size = 16
    num_instances = 4
    sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
    
    print(f"  Data source size: {len(data_source)}")
    print(f"  Batch size: {batch_size}, Num instances: {num_instances}")
    
    iterator = iter(sampler)
    batch = []
    try:
        for _ in range(batch_size):
            batch.append(next(iterator))
            
        print(f"  Sampled batch indices: {batch}")
        
        # Check P x K property
        sampled_pids = [data_source[i][1] for i in batch]
        print(f"  Sampled PIDs: {sampled_pids}")
        
        from collections import Counter
        counts = Counter(sampled_pids)
        print(f"  PID Counts: {counts}")
        
        if all(c == num_instances for c in counts.values()):
            print("  ✓ Sampler produced correct P x K distribution.")
        else:
            print("  X Sampler distribution incorrect.")
            
    except Exception as e:
        print(f"  X Sampler verification failed: {e}")

if __name__ == "__main__":
    verify_model()
    verify_sampler()
