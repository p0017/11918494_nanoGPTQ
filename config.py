
class GPTConfig:
    """Configuration for the GPT model."""
    vocab_size: int = 65                                    # Make sure this matches the dataset
    context_length: int = 128                               # Length of the context window
    n_layer: int = 4                                        # Number of Transformer blocks
    n_attention_heads: int = 8                              # Number of attention heads
    embedding_dimension: int = 256                          # Embedding dimension, must be divisible by n_attention_heads
    dropout_rate: float = 0.1                               # Chose a slightly lower dropout for the small model and dataset
    learning_rate: float = 3e-4                             # Learning rate
    batch_size: int = 64                                    # Batch size
    betas: tuple = (0.9, 0.95)                              # AdamW optimizer betas
    train_from_scratch: bool = True                         # Whether to train the model from scratch
    name: str = "tinyshakespeare_V0.0"                      # Name of the experiment for logging and checkpoints