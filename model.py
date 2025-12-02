import math
import inspect
from beartype import beartype
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. PyTorch's LayerNorm always has bias, this one gives the option to disable it."""

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.bias = None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Causal self-attention mechanism with optional flash attention support.
    Attends only to previous tokens and itself (no peeking into the future)."""

    def __init__(self, config):
        super().__init__()
        assert config.embedding_dimension % config.n_attention_heads == 0
        # one big linear layer for all three learning query, key, value matrices at once
        self.c_attention = nn.Linear(
            config.embedding_dimension, 3 * config.embedding_dimension, bias=config.bias
        )
        # linear layer for combining the output of the different attention heads
        self.c_projection = nn.Linear(
            config.embedding_dimension, config.embedding_dimension, bias=config.bias
        )
        # regularization
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.n_attention_heads = config.n_attention_heads
        self.embedding_dimension = config.embedding_dimension
        self.dropout = config.dropout
        # optional flash attention for faster training, only supported by PyTorch >= 2.0
        self.flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.flash_attention:
            print("Flash attention not available, requires PyTorch >= 2.0")
            # Creating a lower triangular mask for causal masking
            # Such that each token can only attend to previous tokens and itself
            # No attention to future tokens
            causal_mask = torch.tril(
                torch.ones(config.context_length, config.context_length)
            )
            # Adding batch and head dimensions
            causal_mask = causal_mask.view(
                1, 1, config.context_length, config.context_length
            )
            # Registering the mask as a buffer such that its not a learnable parameter
            self.register_buffer("bias", causal_mask)

    def forward(self, x):
        batch_size, sequence_length, embedding_dimension = x.size()

        # Before, we initialized one big linear layer for query, key, value matrices
        # Now we split them into separate matrices
        qkv = self.c_attention(
            x
        )  # (batch_size, sequence_length, 3 * embedding_dimension)
        query, key, value = qkv.split(
            self.embedding_dimension, dim=2
        )  # each is (batch_size, sequence_length, embedding_dimension)

        # Reshape q, k, v for multi-head attention
        # We split the embedding dimension into multiple heads
        key = key.view(
            batch_size,
            sequence_length,
            self.n_attention_heads,
            embedding_dimension // self.n_attention_heads,
        )
        query = query.view(
            batch_size,
            sequence_length,
            self.n_attention_heads,
            embedding_dimension // self.n_attention_heads,
        )
        value = value.view(
            batch_size,
            sequence_length,
            self.n_attention_heads,
            embedding_dimension // self.n_attention_heads,
        )

        # Transpose such that the head dimension comes before the sequence length
        # Each is now (batch_size, n_attention_heads, sequence_length, head_dimension)
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.flash_attention:
            if self.training:
                y = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=self.dropout,
                    is_causal=True,
                )
            else:
                y = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
                )

        else:
            # Manual implementation of scaled dot-product attention
            # Computing raw attention scores then scaling
            attention = query @ key.transpose(-2, -1)
            attention = attention / math.sqrt(key.size(-1))

            # Applying the causal mask to prevent attending to future tokens
            # Setting those scores to -inf before softmax
            attention = attention.masked_fill(
                self.bias[:, :, :sequence_length, :sequence_length] == 0, float("-inf")
            )

            # Applying softmax to get attention weights
            attention = F.softmax(attention, dim=-1)
            attention = self.attention_dropout(attention)

            # Computing the weighted sum of values
            y = attention @ value

        # Re-assemble ass head outputs side by side
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, embedding_dimension)
        )

        # Output projection
        y = self.c_projection(y)
        y = self.residual_dropout(y)
        return y


class MLP(nn.Module):
    """Feed-forward neural network (multi-layer perceptron, MLP) with GELU activation and dropout."""

    def __init__(self, config):
        super().__init__()
        # First linear layer expands the embedding dimension to 4 times its size
        # To let model learn more complex representations
        self.first_linear = nn.Linear(
            config.embedding_dimension, 4 * config.embedding_dimension, bias=config.bias
        )
        self.gelu = nn.GELU()
        # Second linear layer reduces the dimension back to the original embedding size
        self.second_linear = nn.Linear(
            4 * config.embedding_dimension, config.embedding_dimension, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.first_linear(x)
        x = self.gelu(x)
        x = self.second_linear(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """A single Transformer block consisting of causal self-attention and MLP with residual connections and layer normalization."""

    def __init__(self, config):
        super().__init__()
        self.first_layer_norm = LayerNorm(config.embedding_dimension, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.second_layer_norm = LayerNorm(config.embedding_dimension, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # Residual connections around attention and MLP
        x = x + self.attention(self.first_layer_norm(x))
        x = x + self.mlp(self.second_layer_norm(x))
        return x


class GPT(nn.Module):
    """GPT autoregressive language model consisting of token and positional embeddings, multiple Transformer blocks, and a linear mapping head."""

    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        # Getting vocabulary size from the vocabulary string set in config
        self.config.vocabulary_size = len(model_config.vocabulary)
        assert self.config.vocabulary_size is not None
        assert self.config.context_length is not None

        # Main transformer is a dictionary of modules
        self.transformer = nn.ModuleDict(
            dict(
                # Converts input tokens to vectors
                token_embedding=nn.Embedding(
                    self.config.vocabulary_size, self.config.embedding_dimension
                ),
                # Adds an unique vector for every position in the input sequence
                positional_embedding=nn.Embedding(
                    self.config.context_length, self.config.embedding_dimension
                ),
                dropout=nn.Dropout(self.config.dropout),
                # Stack of transformer blocks
                transformer_blocks=nn.ModuleList(
                    [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
                ),
                # Final layer norm before output
                final_layer_norm=LayerNorm(
                    self.config.embedding_dimension, bias=self.config.bias
                ),
            )
        )
        # Linear layer for mapping the final embeddings to logits over the vocabulary
        self.linear_mapping_head = nn.Linear(
            self.config.embedding_dimension, self.config.vocabulary_size, bias=False
        )
        # Weight tying for better generalization and reduced parameters
        self.transformer.token_embedding.weight = self.linear_mapping_head.weight

        self.apply(self._initialize_weights)
        # Apply special initialization to c_projection weights as per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_projection.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

        print(f"Number of parameters: %.2fM" % (self.get_number_of_parameters() / 1e6,))

    def get_number_of_parameters(self, non_embedding: bool = True) -> int:
        """Returns the total number of parameters in the model."""
        number_of_parameters = sum(p.numel() for p in self.parameters())
        if non_embedding:
            number_of_parameters -= self.transformer.token_embedding.weight.numel()
        return number_of_parameters

    def _initialize_weights(self, module):
        """Initialize the weights of linear and embedding layers with a normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        else:
            pass  # No initialization for other modules

    def forward(self, input_token_indices, targets=None):
        device = input_token_indices.device
        batch_size, sequence_length = input_token_indices.size()

        # Ensure the input sequence length does not exceed the model's context length
        assert (
            sequence_length <= self.config.context_length
        ), "Cannot forward, sequence length exceeds context length"

        # Creating a position index tensor [0, 1, 2, ..., sequence_length-1] for positional embeddings
        positional_indices = torch.arange(
            0, sequence_length, dtype=torch.long, device=device
        )

        token_embeddings = self.transformer.token_embedding(
            input_token_indices
        )  # (batch_size, sequence_length, embedding_dimension)
        positional_embeddings = self.transformer.positional_embedding(
            positional_indices
        )  # (sequence_length, embedding_dimension)
        # Adding token and positional embeddings upon each other and applying dropout
        x = self.transformer.dropout(token_embeddings + positional_embeddings)

        # Passing through all transformer blocks
        for block in self.transformer.transformer_blocks:
            x = block(x)
        x = self.transformer.final_layer_norm(x)

        if targets is not None:  # Training mode
            # If there are desired targets, compute the loss
            logits = self.linear_mapping_head(
                x
            )  # (batch_size, sequence_length, vocabulary_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:  # Inference mode
            logits = self.linear_mapping_head(
                x[:, [-1], :]
            )  # Only return logits for the last token
            loss = None

        return logits, loss

    @beartype
    def crop_context_length(self, context_length: int):
        """Crop the model's context length to a new, smaller value."""
        assert (
            context_length <= self.config.context_length
        ), "New context length must be less than or equal to the original context length."
        self.config.context_length = context_length
        self.transformer.positional_embedding = nn.Parameter(
            self.transformer.positional_embedding.weight[:context_length]
        )
        for block in self.transformer.transformer_blocks:
            if hasattr(block.attention, "bias"):
                block.attention.bias = block.attention.bias[
                    :, :, :context_length, :context_length
                ]

    @beartype
    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str
    ):
        parameter_dict = {pn: p for pn, p in self.named_parameters()}
        # Filtering out parameters that do not require grad
        parameter_dict = {pn: p for pn, p in parameter_dict.items() if p.requires_grad}
        # Creating optimizer groups
        # Any two dimensional parameter will be weight decayed, otherwise not
        # Weight tensors in matmuls and embeddings are 2D, biases and LayerNorm/BatchNorm parameters are not
        weight_decay_parameters = [p for n, p in parameter_dict.items() if p.ndim >= 2]
        no_weight_decay_parameters = [
            p for n, p in parameter_dict.items() if p.ndim < 2
        ]
        optimizer_grouped_parameters = [
            {"params": weight_decay_parameters, "weight_decay": weight_decay},
            {"params": no_weight_decay_parameters, "weight_decay": 0.0},
        ]

        # Checking if fused AdamW optimizer is available
        # This is a GPU-optimized version of AdamW for faster training and lower memory usage
        fused_adamw_available = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
            and device_type == "cuda"
        )
        extra_arguments = dict(fused=True) if fused_adamw_available else dict()
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            **extra_arguments,
        )
        print(
            f"Using {'fused ' if fused_adamw_available else 'non-fused '}AdamW optimizer"
        )

        return optimizer

    @beartype
    @torch.no_grad()
    def generate(
        self,
        input_token_indices,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        """
        Autoregressively generate new tokens given a prompt.

        input_token_indices: (batch_size, sequence_length) tensor of token indices as prompt
        max_new_tokens: number of new tokens to generate
        temperature: controls randomness of sampling
        top_k: optionally restricts sampling to the top k most probable tokens
        """

        for _ in range(max_new_tokens):
            # If the input sequence fits into the context length, use it as is
            if input_token_indices.size(1) <= self.config.context_length:
                input_token_indices_condition = input_token_indices
            # Otherwise, crop the input
            else:
                input_token_indices_condition = input_token_indices[
                    :, -self.config.context_length :
                ]

            logits, _ = self.forward(input_token_indices_condition)
            # We only need the logits for the last token, all previous tokens are in the input
            # And adjust by temperature
            logits = logits[:, -1, :] / temperature

            # Optionally only keep the top k logits
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all logits not in the top k to -inf
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Logits to probabilities using softmax
            probabilities = F.softmax(logits, dim=-1)
            # Sample the next token according to the probabilities
            next_token_index = torch.multinomial(probabilities, num_samples=1)
            # Append the newly generated token to the input sequence
            input_token_indices = torch.cat(
                (input_token_indices, next_token_index), dim=1
            )

        return input_token_indices
