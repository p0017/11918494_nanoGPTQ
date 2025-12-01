import os
import numpy as np
import torch
import time
import math

from config import GPTConfig
from model import GPT

config = GPTConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)


def get_batch(split: str):
    if split == "train":
        path = "./data/tinyshakespeare_train.bin"
    elif split == "val":
        path = "./data/tinyshakespeare_val.bin"
    elif split == "test":
        path = "./data/tinyshakespeare_test.bin"
    else:
        raise ValueError("Invalid split name. Use 'train', 'val' or 'test'.")

    # Loading the entire dataset into RAM, since its only about 2MB
    data = np.fromfile(path, dtype=np.uint16).astype(np.int16)
    # Randomly select starting indices for the batch
    starting_index = torch.randint(
        len(data) - config.context_length, size=(config.batch_size,)
    )
    # For each starting index, extract input sequences of length context_length
    # And the corresponding target sequences shifted by one position
    input = torch.stack(
        [
            torch.tensor(
                data[i.item() : i.item() + config.context_length], dtype=torch.long
            )
            for i in starting_index
        ]
    )
    target = torch.stack(
        [
            torch.tensor(
                data[i.item() + 1 : i.item() + config.context_length + 1],
                dtype=torch.long,
            )
            for i in starting_index
        ]
    )

    if device == "cuda":
        # Pin memory for faster transfer from CPU to GPU
        input, target = input.pin_memory().to(
            "cuda", non_blocking=True
        ), target.pin_memory().to("cuda", non_blocking=True)
    else:
        input, target = input.to(device), target.to(device)
    return input, target


if __name__ == "__main__":
    current_iteration = 0
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join("checkpoints", f"{config.name}_best.pt")

    # Either we train from scratch
    if config.train_from_scratch:
        print("Training model from scratch...")
        model = GPT(config)
        optimizer = model.configure_optimizers(
            config.weight_decay, config.learning_rate, config.betas, device_type=device
        )

    # Or we resume from the best previous checkpoint
    else:
        print(f"Resuming training model from {best_checkpoint_path}...")
        if not os.path.exists(best_checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {best_checkpoint_path}")

        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        config = checkpoint["config"]

        current_iteration = checkpoint["iteration"]
        best_val_loss = checkpoint["best_val_loss"]

        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        optimizer = model.configure_optimizers(
            config.weight_decay, config.learning_rate, config.betas, device_type=device
        )
        # Free up memory
        checkpoint = None

    model.to(device=device)

    @torch.no_grad()
    def get_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            # Evaluating on iters_per_eval batches
            losses = torch.zeros(config.iters_per_eval)
            for k in range(config.iters_per_eval):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            # Average loss over the evaluated batches
            out[split] = losses.mean()
        model.train()
        return out

    def get_learning_rate(iteration: int) -> float:
        # First some warmup
        if iteration < config.warmup_iters:
            return config.learning_rate * (iteration + 1) / (config.warmup_iters + 1)

        # Then cosine decay down to min_learning_rate
        else:
            decay_ratio = (iteration - config.warmup_iters) / (
                config.max_iters - config.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return config.min_learning_rate + coeff * (
                config.learning_rate - config.min_learning_rate
            )

    print("Starting training loop...")
    X, Y = get_batch("train")
    t0 = time.time()

    while current_iteration <= config.max_iters:

        # Setting the current learning rate
        lr = get_learning_rate(current_iteration)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation and checkpointing every eval_interval iterations
        if current_iteration % config.eval_interval == 0 and current_iteration > 0:
            # Estimate time remaining
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            iters_left = config.max_iters - current_iteration
            sec_per_iter = dt / config.eval_interval if current_iteration > 0 else 0
            eta_sec = iters_left * sec_per_iter
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

            losses = get_loss()
            print(
                f"step {current_iteration}/{config.max_iters} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"lr {lr:.2e} | "
                f"ETA {eta_str}"
            )

            # Checkpointing the new best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "iteration": current_iteration,
                    "best_val_loss": best_val_loss,
                }

                print(f"Saving best checkpoint to {best_checkpoint_path}")
                torch.save(checkpoint, best_checkpoint_path)

        _, loss = model(X, Y)
        # Preloading the next batch
        X, Y = get_batch("train")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        current_iteration += 1
