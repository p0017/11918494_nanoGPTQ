import os
import torch
import torch.nn as nn
from model import GPT
from config import quantization_config


class QuantizedLinear(torch.nn.Module):
    """A linear layer with quantized weights, replaces nn.Linear in quantized models."""

    def __init__(self, original_linear, weight_int8, scale):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # Registering buffers such that they are saved and loaded with the state_dict
        self.register_buffer("weight", weight_int8)
        self.register_buffer("scale", scale)

        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """Dequantize weights on-the-fly for the forward pass."""

        w_dequantized = self.weight.to(x.dtype) * self.scale
        return torch.nn.functional.linear(x, w_dequantized, self.bias)


def get_row_wise_scaling_factors(weight_matrix):
    """Computes row-wise scaling factors of a weight matrix for symmetric quantization."""

    return weight_matrix.abs().amax(dim=1, keepdim=True) / 127


def naive_symmetric_quantization(weight_matrix, row_wise_scaling_factors):
    """Takes in a weight matrix and row-wise scaling factors, returns int8 quantized weight matrix."""

    return torch.clamp(
        torch.round(weight_matrix / row_wise_scaling_factors), -127, 127
    ).to(torch.int8)


def replace_linear_with_quantized(module: nn.Module):
    """Recursively replaces all nn.Linear layers in the module with QuantizedLinear layers.
    If the module has children, it moves down through them recursively."""

    for name, child_module in module.named_children():
        # Iterate through child modules
        if isinstance(child_module, nn.Linear):
            # Only quantize nn.Linear layers

            weight_matrix = child_module.weight.data
            row_wise_scaling_factors = get_row_wise_scaling_factors(weight_matrix)
            quantized_weights = naive_symmetric_quantization(
                weight_matrix, row_wise_scaling_factors
            )

            quantized_layer = QuantizedLinear(
                child_module, quantized_weights, row_wise_scaling_factors
            )
            # Replace the original nn.Linear layer with the quantized version
            setattr(module, name, quantized_layer)
        # If its not a linear layer, recurse and check its children
        else:
            replace_linear_with_quantized(child_module)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(
        "checkpoints", f"{quantization_config.experiment_name}.pt"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GPT(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])

    replace_linear_with_quantized(model)
    # Save the quantized model with its original config and training state
    quantized_checkpoint = {
        "model": model.state_dict(),
        "model_config": checkpoint["model_config"],
        "iteration": checkpoint.get("iteration", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
    }

    torch.save(
        quantized_checkpoint,
        os.path.join(
            "checkpoints", f"{quantization_config.experiment_name}_quantized_naive.pt"
        ),
    )
