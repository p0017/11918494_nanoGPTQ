import os
import torch
from model import GPT
from data_preprocessing import encode, decode
from config import sample_config, VOCABULARY

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = os.path.join(
    "checkpoints", f"{sample_config.experiment_name}_best.pt"
)
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

print(f"Loading model from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)
model = GPT(checkpoint["model_config"])
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

print(
    f"Model loaded (Iter: {checkpoint['iteration']}, Loss: {checkpoint['best_val_loss']:.4f})"
)
print("-" * 50)

start_ids = encode(sample_config.start_prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    y = model.generate(
        x,
        sample_config.max_new_tokens,
        temperature=sample_config.temperature,
        top_k=sample_config.top_k,
    )
    print(decode(y[0].tolist()))
    print("-" * 50)
