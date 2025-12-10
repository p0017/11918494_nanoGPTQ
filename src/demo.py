import os
from beartype import beartype
import torch
import gradio as gr
from src.model import GPT
from src.data_preprocessing import encode, decode
from src.sample import replace_with_dummy_quantized
from src.config import demo_config, validate_demo_config

validate_demo_config()
device = "cuda" if torch.cuda.is_available() else "cpu"


@beartype
def load_model(experiment_name: str) -> GPT:
    """
    Loads a model from the specified checkpoint.

    Args:
        experiment_name (str): Name of the experiment checkpoint to load.
    Returns:
        GPT: The loaded GPT model.
    """

    checkpoint_path = os.path.join("checkpoints", f"{experiment_name}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GPT(checkpoint["model_config"])

    # If quantized, we need to replace nn.Linear with QuantizedLinear before loading state_dict
    if "quantized" in experiment_name:
        print("Model is quantized. Swapping layers.")
        replace_with_dummy_quantized(model)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


@beartype
def generate_demo_text(model: GPT, prompt: str) -> str:
    """Generates text from the model given a prompt.

    Args:
        model (GPT): The GPT model to use for generation.
        prompt (str): The input prompt string.
    Returns:
        str: The generated text.
    """

    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(
            x,
            demo_config.max_new_tokens,
            temperature=demo_config.temperature,
            top_k=demo_config.top_k,
        )
    return decode(y[0].tolist())


# We want to compare the responses from three models
baseline_model = load_model(demo_config.baseline_experiment_name)
naive_quantized_model = load_model(demo_config.naive_quantized_experiment_name)
gptq_quantized_model = load_model(demo_config.gptq_quantized_experiment_name)

# If possible, load models to GPU for faster inference
try:
    baseline_model.to(device)
    naive_quantized_model.to(device)
    gptq_quantized_model.to(device)
except RuntimeError:
    print("Cannot load all models to GPU due to memory constraints. Running on CPU.")
    device = "cpu"

if __name__ == "__main__":

    with gr.Blocks() as demo:
        gr.Markdown("# NanoGPTQ Demo")

        prompt_input = gr.Textbox(label="Enter your prompt", lines=2)
        generate_button = gr.Button("Generate")

        # Display outputs from all three models side by side
        with gr.Row():
            baseline_output = gr.Textbox(label="Baseline Model", lines=10)
            naive_output = gr.Textbox(label="Naive Quantized Model", lines=10)
            gptq_output = gr.Textbox(label="GPTQ Quantized Model", lines=10)

        def demo_function_blocks(prompt: str):
            """Generates text from all three models given a prompt.

            Args:
                prompt (str): The input prompt string.
            """

            baseline_text = generate_demo_text(baseline_model, prompt)
            naive_text = generate_demo_text(naive_quantized_model, prompt)
            gptq_text = generate_demo_text(gptq_quantized_model, prompt)
            return baseline_text, naive_text, gptq_text

        generate_button.click(
            fn=demo_function_blocks,
            inputs=prompt_input,
            outputs=[baseline_output, naive_output, gptq_output],
        )

    demo.launch()
