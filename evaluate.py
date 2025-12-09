import torch
import os
from config import evaluation_config, validate_evaluation_config
from train import get_batch
from model import GPT
from sample import replace_with_dummy_quantized

validate_evaluation_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)


def evaluate_model(checkpoint_name: str) -> float:
    """
    Loads a model, calculates average loss on the test set, returns the Perplexity.

    Args:
        checkpoint_name (str): Name of the experiment checkpoint to load.
    Returns:
        float: The calculated Perplexity on the test set.
    """

    checkpoint_path = os.path.join("checkpoints", f"{checkpoint_name}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GPT(checkpoint["model_config"])

    # If quantized, we need to replace nn.Linear with QuantizedLinear before loading state_dict
    if "quantized" in checkpoint_name:
        print("Model is quantized. Swapping layers.")
        replace_with_dummy_quantized(model)

    model.load_state_dict(checkpoint["model"])
    model.to(device)

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for _ in range(evaluation_config.eval_batches):
            x, y = get_batch("test")
            _, loss = model(x, y)
            total_loss += loss.item()

    average_loss = total_loss / evaluation_config.eval_batches
    perplexity = torch.exp(torch.tensor(average_loss)).item()

    # Free up memory
    del model
    torch.cuda.empty_cache()

    return perplexity


if __name__ == "__main__":
    # Evaluate baseline and quantized models one after another
    perplexity_baseline = evaluate_model(evaluation_config.baseline_experiment_name)
    perplexity_naive = evaluate_model(evaluation_config.naive_quantized_experiment_name)
    perplexity_gptq = evaluate_model(evaluation_config.gptq_quantized_experiment_name)

    print("_" * 50)
    print(f"Baseline Perplexity: {perplexity_baseline:.4f}")
    print(f"Naive Perplexity:    {perplexity_naive:.4f}")
    print(f"GPTQ Perplexity:     {perplexity_gptq:.4f}")
