import os
from beartype import beartype
import torch
from src.model import GPT
from src.data_preprocessing import encode, decode
from src.quantize import QuantizedLinear
from src.config import sample_config, validate_sample_config
import torch.nn as nn

validate_sample_config()


@beartype
def replace_with_dummy_quantized(module: nn.Module):
    """Before loading a quantized model, we need to replace all nn.Linear layers with
    QuantizedLinear layers, such that the state_dict can be loaded correctly.

    Args:
        module (nn.Module): The module to replace layers in.
    """

    for name, child in module.named_children():
        if name == "linear_mapping_head":
            continue  # Skip the linear mapping head

        # Iterate through child modules
        if isinstance(child, nn.Linear):
            # Only quantize nn.Linear layers
            # Replacing with dummy quantized weights and scales for now
            # Actual weights and scales will be loaded from the checkpoint
            dummy_weight = torch.zeros_like(child.weight, dtype=torch.int8)
            dummy_scale = torch.zeros((child.out_features, 1), dtype=child.weight.dtype)

            quantized_layer = QuantizedLinear(child, dummy_weight, dummy_scale)
            setattr(module, name, quantized_layer)
        else:
            # If the child module has children, recurse
            replace_with_dummy_quantized(child)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join("checkpoints", f"{sample_config.experiment_name}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GPT(checkpoint["model_config"])

    # If quantized, we need to replace nn.Linear with QuantizedLinear before loading state_dict
    if "quantized" in sample_config.experiment_name:
        print("Model is quantized. Swapping layers.")
        replace_with_dummy_quantized(model)

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(
        f"Model loaded (Iter: {checkpoint['iteration']}, Loss: {checkpoint['best_val_loss']:.4f})"
    )

    print("_" * 50)
    # Sampling from the model
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
        print("_" * 50)
