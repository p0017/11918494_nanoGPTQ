from beartype import beartype
import os

VOCABULARY = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class model_config:
    """Configuration for the GPT model."""

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
    vocabulary: str = VOCABULARY  # Vocabulary string


class train_config:
    """Configuration for training parameters."""

    learning_rate: float = 3e-4  # Learning rate
    min_learning_rate: float = 5e-5  # Minimum learning rate for cosine decay
    betas: tuple = (0.9, 0.95)  # AdamW optimizer betas
    weight_decay: float = 0.2  # Weight decay for AdamW optimizer
    batch_size: int = 32  # Batch size
    max_iters: int = 14000  # Total number of training iterations
    warmup_iters: int = 500  # Number of warmup iterations
    iters_per_eval: int = 15  # Number of iterations for evaluation
    eval_interval: int = 500  # Interval for evaluation and checkpointing
    train_from_scratch: bool = True  # Whether to train the model from scratch
    name: str = (
        "tinyshakespeare_8x384_full_run_highreg"  # Name of the experiment for logging and checkpoints
    )


class sample_config:
    """Configuration for sampling parameters."""

    experiment_name = "tinyshakespeare_8x384_full_run_highreg_quantized_gptq"  # Name of the experiment to load the model from
    start_prompt = "To be, or "  # Can be any string, e.g., "To be, or not to be"
    max_new_tokens = 500  # How many characters to generate
    temperature = 0.8  # 1.0 = standard, < 1.0 = more conservative, > 1.0 = more random
    top_k = 200  # Retain only the top_k most likely tokens (clamp distribution)


class quantization_config:
    """Configuration for quantization parameters."""

    experiment_name = "tinyshakespeare_8x384_full_run_highreg"  # Name of the experiment to load the model from
    method: str = "gptq"  # Quantization method: 'naive' or 'gptq'


@beartype
def validate_model_config():
    """Check if model hyperparameters are valid."""
    assert model_config.context_length > 0, "Context length must be positive"
    assert model_config.n_layers > 0, "Number of layers must be positive"
    assert (
        model_config.n_attention_heads > 0
    ), "Number of attention heads must be positive"
    assert model_config.embedding_dimension > 0, "Embedding dimension must be positive"
    # Embedding dim must be divisible by heads
    assert (
        model_config.embedding_dimension % model_config.n_attention_heads == 0
    ), "Embedding dimension must be divisible by number of attention heads"
    assert 0 <= model_config.dropout < 1.0, "Dropout must be in [0, 1)"


@beartype
def validate_train_config():
    """Check if training parameters are valid."""
    assert train_config.learning_rate > 0, "Learning rate must be positive"
    assert (
        train_config.min_learning_rate >= 0
    ), "Minimum learning rate must be non-negative"
    assert (
        train_config.min_learning_rate <= train_config.learning_rate
    ), "Minimum learning rate must be less than or equal to learning rate"
    assert train_config.batch_size > 0, "Batch size must be positive"
    assert train_config.max_iters > 0, "Max iterations must be positive"
    assert train_config.warmup_iters >= 0, "Warmup iterations must be non-negative"
    assert (
        train_config.warmup_iters <= train_config.max_iters
    ), "Warmup iterations must be less than or equal to max iterations"
    assert train_config.iters_per_eval > 0, "Iters per eval must be positive"
    assert train_config.eval_interval > 0, "Eval interval must be positive"
    assert (
        train_config.eval_interval <= train_config.max_iters
    ), "Eval interval must be less than or equal to max iterations"


@beartype
def validate_sample_config():
    """Check if sampling parameters are valid."""
    checkpoint_path = os.path.join("checkpoints", f"{sample_config.experiment_name}.pt")
    assert os.path.isfile(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"
    assert sample_config.max_new_tokens > 0, "Max new tokens must be positive"
    assert sample_config.temperature > 0, "Temperature must be positive"
    assert sample_config.top_k > 0, "Top-k must be positive"


@beartype
def validate_quantization_config():
    """Check if quantization parameters are valid."""
    checkpoint_path = os.path.join(
        "checkpoints", f"{quantization_config.experiment_name}.pt"
    )
    assert os.path.isfile(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"
    assert quantization_config.method in [
        "naive",
        "gptq",
    ], "Quantization method must be 'naive' or 'gptq'"
