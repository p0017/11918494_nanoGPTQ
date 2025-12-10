import torch
import pytest
from unittest.mock import patch
import train
from model import GPT


class dummy_model_config:
    context_length = 8  # Length of the context window
    n_layers = 1  # Number of Transformer blocks
    n_attention_heads = 2  # Number of attention heads
    embedding_dimension = (
        4  # Embedding dimension, must be divisible by n_attention_heads
    )
    dropout = 0.0  # No dropout for testing
    bias = False
    vocabulary = "ABCabc"  # Small vocabulary string


class dummy_train_config:
    learning_rate = 1e-3
    min_learning_rate = 1e-4
    betas = (0.9, 0.95)
    weight_decay = 0.1
    batch_size = 2
    max_iters = 2  # Only run 2 iterations
    warmup_iters = 0
    eval_interval = 100  # Set high to avoid eval and checkpointing during test
    iters_per_eval = 1
    train_from_scratch = True
    name = "test_run"


def test_training_loop_cpu():
    """
    Smoke test: Runs a few iterations of the training loop on CPU
    with dummy data to ensure the pipeline doesn't crash.
    """
    model_config = dummy_model_config()
    train_config = dummy_train_config()
    device = "cpu"  # Force CPU for CI
    model_config.vocabulary_size = len(model_config.vocabulary)

    model = GPT(model_config)
    model.to(device)

    optimizer = model.configure_optimizers(
        train_config.weight_decay, train_config.learning_rate, train_config.betas
    )

    # Ensuring get_learning_rate uses the dummy config, not the global one
    with patch("train.train_config", train_config):

        # Set model to training mode
        model.train()

        # Create dummy data of shape [batch_size, context_length]
        X = torch.randint(
            0,
            model_config.vocabulary_size,
            (train_config.batch_size, model_config.context_length),
        )
        Y = torch.randint(
            0,
            model_config.vocabulary_size,
            (train_config.batch_size, model_config.context_length),
        )

        for i in range(train_config.max_iters):
            # Forward
            logits, loss = model(X, Y)

            # Check output shapes
            assert logits.shape == (
                train_config.batch_size,
                model_config.context_length,
                model_config.vocabulary_size,
            )
            # Check loss is computed
            assert loss is not None

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
