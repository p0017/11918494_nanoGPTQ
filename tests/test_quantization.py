import torch
import torch.nn as nn
from quantize import naive_quantization, QuantizedLinear


def test_layer_replacement():
    """Test that Linear layers are replaced with QuantizedLinear layers."""
    # Creating a dummy model
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    # Apply quantization
    naive_quantization(model)

    # Check if layers were swapped
    assert isinstance(model[0], QuantizedLinear)  # Should be QuantizedLinear
    assert isinstance(model[1], nn.ReLU)  # Should remain ReLU
    assert isinstance(model[2], QuantizedLinear)  # Should be QuantizedLinear


def test_quantized_linear_forward():
    """Test forward pass of QuantizedLinear layer."""
    original = nn.Linear(4, 2, bias=False)
    # Creating a dummy quantized weight and scale
    weight_matrix_int8 = torch.zeros_like(original.weight, dtype=torch.int8)
    scale = torch.ones((2, 1), dtype=torch.float32)

    quantized_layer = QuantizedLinear(original, weight_matrix_int8, scale)

    input_tensor = torch.randn(1, 4)
    output = quantized_layer(input_tensor)

    assert output.shape == (1, 2)
