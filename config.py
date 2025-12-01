class GPTConfig:
    """Configuration for the GPT model."""

    vocabulary_size: int = 65  # Make sure this matches the dataset
    context_length: int = 128  # Length of the context window
    n_layers: int = 4  # Number of Transformer blocks
    n_attention_heads: int = 8  # Number of attention heads
    embedding_dimension: int = (
        256  # Embedding dimension, must be divisible by n_attention_heads
    )
    dropout: float = (
        0.1  # Chose a slightly lower dropout for the small model and dataset
    )
    bias: bool = False  # Whether to use bias
    learning_rate: float = 3e-4  # Learning rate
    min_learning_rate: float = 1e-5  # Minimum learning rate for cosine decay
    batch_size: int = 64  # Batch size
    betas: tuple = (0.9, 0.95)  # AdamW optimizer betas
    weight_decay: float = 0.1  # Weight decay for AdamW optimizer
    max_iters: int = 50  # Total number of training iterations
    warmup_iters: int = 5  # Number of warmup iterations
    iters_per_eval: int = 5  # Number of iterations for evaluation
    eval_interval: int = 20  # Interval for evaluation and checkpointing
    train_from_scratch: bool = True  # Whether to train the model from scratch
    name: str = (
        "tinyshakespeare_V0"  # Name of the experiment for logging and checkpoints
    )
