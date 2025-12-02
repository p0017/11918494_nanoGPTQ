VOCABULARY = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class model_config:
    """Configuration for the GPT model."""

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
    vocabulary: str = VOCABULARY  # Vocabulary string


class train_config:
    """Configuration for training parameters."""

    learning_rate: float = 3e-4  # Learning rate
    min_learning_rate: float = 5e-5  # Minimum learning rate for cosine decay
    betas: tuple = (0.9, 0.95)  # AdamW optimizer betas
    weight_decay: float = 0.1  # Weight decay for AdamW optimizer
    batch_size: int = 64  # Batch size
    max_iters: int = 2000  # Total number of training iterations
    warmup_iters: int = 200  # Number of warmup iterations
    iters_per_eval: int = 10  # Number of iterations for evaluation
    eval_interval: int = 250  # Interval for evaluation and checkpointing
    train_from_scratch: bool = True  # Whether to train the model from scratch
    name: str = (
        "tinyshakespeare_V0.1"  # Name of the experiment for logging and checkpoints
    )


class sample_config:
    """Configuration for sampling parameters."""

    experiment_name = (
        "tinyshakespeare_V0.1"  # Name of the experiment to load the model from
    )
    start_prompt = "\n"  # Can be any string, e.g., "To be, or not to be"
    max_new_tokens = 500  # How many characters to generate
    temperature = 0.8  # 1.0 = standard, < 1.0 = more conservative, > 1.0 = more random
    top_k = 200  # Retain only the top_k most likely tokens (clamp distribution)
