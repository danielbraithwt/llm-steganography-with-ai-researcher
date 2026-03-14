import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Memory: {mem:.1f} GB")
import transformers
print(f"Transformers: {transformers.__version__}")
from transformers import DynamicCache
dc = DynamicCache()
print(f"DynamicCache has .layers: {hasattr(dc, 'layers')}")
from scipy import stats
print("scipy.stats OK")
