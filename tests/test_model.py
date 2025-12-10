import torch
import pytest
from model import GPT


class dummy_model_config:
    """A dummy model config for testing the GPT model."""

    context_length: int = 256  # Length of the context window
    n_layers: int = 8  # Number of Transformer blocks
    n_attention_heads: int = 6  # Number of attention heads
    embedding_dimension: int = (
        384  # Embedding dimension, must be divisible by n_attention_heads
    )
    dropout: float = (
        0.4  # Chose a slightly lower dropout for the small model and dataset
    )
    bias: bool = False  # Whether to use bias
    vocabulary: str = "ABCabc"  # Vocabulary string


def test_gpt_output_shape():
    """Testing the GPT model output shape and loss computation."""
    model = GPT(dummy_model_config)
    batch_size = 2
    # Creating a dummy input [batch, context_length]
    x = torch.randint(
        0,
        len(dummy_model_config.vocabulary),
        (batch_size, dummy_model_config.context_length),
    )
    logits, loss = model(x, targets=x)

    assert logits.shape == (
        batch_size,
        dummy_model_config.context_length,
        len(dummy_model_config.vocabulary),
    )
    assert loss is not None  # Targets provided, so loss should be computed
