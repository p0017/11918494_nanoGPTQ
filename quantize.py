import os
from beartype import beartype
import torch
import torch.nn as nn
from model import GPT
from config import quantization_config, validate_quantization_config
from train import get_batch

validate_quantization_config()
device = "cuda" if torch.cuda.is_available() else "cpu"


@beartype
class QuantizedLinear(torch.nn.Module):
    """A linear layer with quantized weights, replaces nn.Linear in quantized models.
    Dequantizes weights on-the-fly during the forward pass.

    Attributes:
        original_linear (nn.Linear): The original linear layer to be quantized.
        weight_int8 (torch.Tensor): The quantized weights in int8 format.
        scale (torch.Tensor): The scaling factors for dequantization.
    """

    def __init__(
        self, original_linear: nn.Linear, weight_int8: torch.Tensor, scale: torch.Tensor
    ):
        """Initializes the QuantizedLinear layer.

        Args:
            original_linear (nn.Linear): The original linear layer to be quantized.
            weight_int8 (torch.Tensor): The quantized weights in int8 format.
            scale (torch.Tensor): The scaling factors for dequantization.
        """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize weights on-the-fly for the forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """

        w_dequantized = self.weight.to(x.dtype) * self.scale
        return torch.nn.functional.linear(x, w_dequantized, self.bias)


@beartype
def get_row_wise_scaling_factors(weight_matrix: torch.Tensor) -> torch.Tensor:
    """Computes row-wise scaling factors of a weight matrix for symmetric quantization.

    Args:
        weight_matrix (torch.Tensor): The weight matrix to compute scaling factors for.
    Returns:
        torch.Tensor: Row-wise scaling factors.
    """

    return weight_matrix.abs().amax(dim=1, keepdim=True) / 127


@beartype
def naive_quantization(module: nn.Module):
    """Recursively replaces all nn.Linear layers in the module with QuantizedLinear layers,
    and sets their weights to quantized weights. If the module has children, it moves down through them recursively.

    Args:
        module (nn.Module): The module to quantize.
    """

    for name, child_module in module.named_children():
        # Iterate through child modules
        if isinstance(child_module, nn.Linear):
            # Only quantize nn.Linear layers

            weight_matrix = child_module.weight.data
            scaling_factors = get_row_wise_scaling_factors(weight_matrix)

            # Quantizing the weights to int8
            quantized_weights = torch.clamp(
                torch.round(weight_matrix / scaling_factors), -127, 127
            )
            quantized_weights = quantized_weights.to(torch.int8)

            quantized_layer = QuantizedLinear(
                child_module, quantized_weights, scaling_factors
            )
            # Replace the original nn.Linear layer with the quantized version
            setattr(module, name, quantized_layer)
        # If its not a linear layer, recurse and check its children
        else:
            naive_quantization(child_module)


@beartype
def gptq_math(
    layer: nn.Linear, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs the actual GPTQ algorithm on a single layer.

    Args:
        layer (nn.Linear): The linear layer to quantize.
        inputs (torch.Tensor): The inputs to the layer used to compute the Hessian.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized weights and scaling factors.

    """
    # W is the weight matrix of shape [out_features, in_features]
    # We quantize it column by column, so iterating over in_features
    W = layer.weight.data.clone().float()
    cols = W.shape[1]

    # Computing the Hessian (H = 2 * X.T @ X)
    # The derivation of this can be found in the OBS paper by Hassibi et al.
    H = 2 * torch.matmul(inputs.T, inputs)

    # Dampening for numerical stability
    # Adding a small value to the diagonal of H
    # Instead of just dampening with a constant, we base it on the magnitude of H
    dampening_factor = 0.01 * torch.mean(torch.diag(H))
    diagonal = torch.arange(cols, device=W.device)
    H[diagonal, diagonal] += dampening_factor

    # Inverting the Hessian using Cholesky decomposition
    # Any positive definite matrix can be decomposed into H = L @ L.T
    # Where L is a lower triangular matrix
    # Therefore H_inv = L_inv.T @ L_inv
    try:
        L = torch.linalg.cholesky(H)
        L_inv = torch.linalg.solve_triangular(
            L, torch.eye(cols, device=W.device), upper=False
        )
        H_inv = torch.matmul(L_inv.T, L_inv)
    # If not positive definite, fall back to standard inverse
    except RuntimeError:
        print("Hessian not positive definite, falling back to standard inverse")
        H_inv = torch.inverse(H)

    # We iterate over columns (in_features), as in the GPTQ paper
    W_quantized = torch.zeros_like(W)
    # But we are still using row-wise scaling factors
    scaling_factors = get_row_wise_scaling_factors(W)

    for i in range(cols):
        # Getting the current weight column
        # This is the column we will quantize now
        current_w_column = W[:, i]

        # We will first quantize this column naively
        scaling_factors_flat = scaling_factors.flatten()
        current_w_column_quantized = torch.clamp(
            torch.round(current_w_column / scaling_factors_flat), -127, 127
        )
        # Then dequantize it again using the scaling factors
        current_w_column_quantized = current_w_column_quantized * scaling_factors_flat
        W_quantized[:, i] = current_w_column_quantized

        # Because we just need it to get the quantization error
        # By which we will adjust the remaining weights
        quantization_error = current_w_column - current_w_column_quantized

        # Update the remaining weights using the quantization error
        # And second order information from the inverse Hessian
        # The derivation of this can be found in the GPTQ paper by Frantar et al.
        H_inverse_diagonal = H_inv[i, i]
        H_inverse_remaining_cols = H_inv[i, i:]

        # We update all future columns to compensate for the error in the current column
        if i < cols - 1:
            weight_update = (
                torch.outer(quantization_error, H_inverse_remaining_cols[1:])
                / H_inverse_diagonal
            )
            W[:, i + 1 :] -= weight_update

    # After quantizing and updating all columns, we do a final quantization step
    # to get the final quantized weights and scaling factors
    W_final_quantized = torch.clamp(
        torch.round(W_quantized / scaling_factors), -127, 127
    )
    W_final_quantized = W_final_quantized.to(torch.int8)
    return W_final_quantized, scaling_factors


@beartype
def gptq_quantization(model: nn.Module):
    """
    Main loop for GPTQ. Iterates through blocks, quantizes them, and updates the data.

    Args:
        model (nn.Module): The model to quantize.
    """

    # Getting four batches for calibration
    input = [get_batch("train")[0] for _ in range(4)]
    input = torch.cat(input, dim=0).to(device)
    context_length = input.shape[1]

    # Getting the initial embeddings for the initial forward pass
    with torch.no_grad():
        positions = torch.arange(0, context_length, device=device)
        token_embeddings = model.transformer.token_embedding(input)
        positional_embeddings = model.transformer.positional_embedding(positions)
        input_embeddings = token_embeddings + positional_embeddings

    for i, transformer_block in enumerate(model.transformer.transformer_blocks):
        print(
            f"Quantizing Block {i+1} out of {len(model.transformer.transformer_blocks)}..."
        )

        # All linear layers of the block in the order in which they are called
        linear_layers = {
            "attention.c_attention": transformer_block.attention.c_attention,
            "attention.c_projection": transformer_block.attention.c_projection,
            "mlp.first_linear": transformer_block.mlp.first_linear,
            "mlp.second_linear": transformer_block.mlp.second_linear,
        }

        # Iterating though each linear layer and quantizing it
        for name, layer in linear_layers.items():

            # Using hooks to get the input to the linear layer
            # The input will be stored in this dictionary
            hook_data = {"layer_input": None}

            def hook(module, input, output):
                input = input[0]
                # Detaching the input such that we dont interfere with gradients
                hook_data["layer_input"] = input.view(-1, input.shape[-1]).detach()

            # Putting the hook on the layer
            current_hook = layer.register_forward_hook(hook)

            # Forward pass through the entire block
            # It would be more efficient to only pass the data through the necessary submodules,
            # But this would be more complex to implement
            with torch.no_grad():
                transformer_block(input_embeddings)

            # Checking if the hook captured the input
            if hook_data["layer_input"] is None:
                raise RuntimeError("Hook failed to capture layer inputs.")

            # Remove the hook since we don't need it anymore
            current_hook.remove()

            # Run the math
            # The Hessian is computed using the layer inputs
            q_weight, scale = gptq_math(layer, hook_data["layer_input"])
            # Create the new QuantizedLinear layer
            # And replace the original layer in the model with it
            quantized_layer = QuantizedLinear(layer, q_weight, scale)
            parent_name, child_name = name.split(".")
            parent_module = getattr(transformer_block, parent_name)
            setattr(parent_module, child_name, quantized_layer)

            # Deleting the stored inputs to free up memory
            del hook_data
            torch.cuda.empty_cache()

        # Forward pass to update input_embeddings for the next block
        with torch.no_grad():
            input_embeddings = transformer_block(input_embeddings)


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

    if quantization_config.method == "naive":
        print("Applying naive symmetric quantization...")
        naive_quantization(model)

    elif quantization_config.method == "gptq":
        print("Applying GPTQ quantization, this can take a few seconds...")
        model.to(device)
        gptq_quantization(model)

    else:
        raise ValueError(
            f"Quantization method {quantization_config.method} not recognized."
        )

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
            "checkpoints",
            f"{quantization_config.experiment_name}_quantized_{quantization_config.method}.pt",
        ),
    )
