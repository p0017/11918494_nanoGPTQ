import os
from beartype import beartype
import numpy as np
import torch
import time
import math
from config import (
    model_config,
    train_config,
    validate_model_config,
    validate_train_config,
)
from model import GPT

validate_model_config()
validate_train_config()
device = "cuda" if torch.cuda.is_available() else "cpu"


@beartype
def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a batch of data for training or evaluation.

    Args:
        split (str): The data split to use ('train', 'val', or 'test').
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing input and target tensors.
    """

    if split == "train":
        path = "./data/tinystories_train.bin"
    elif split == "val":
        path = "./data/tinystories_val.bin"
    elif split == "test":
        path = "./data/tinystories_test.bin"
    else:
        raise ValueError("Invalid split name. Use 'train', 'val' or 'test'.")

    # Loading the entire dataset into RAM, since its only about 2MB
    data = np.fromfile(path, dtype=np.uint16).astype(np.int16)
    # Randomly select starting indices for the batch
    starting_index = torch.randint(
        len(data) - model_config.context_length, size=(train_config.batch_size,)
    )
    # For each starting index, extract input sequences of length context_length
    # And the corresponding target sequences shifted by one position
    input = torch.stack(
        [
            torch.tensor(
                data[i.item() : i.item() + model_config.context_length],
                dtype=torch.long,
            )
            for i in starting_index
        ]
    )
    target = torch.stack(
        [
            torch.tensor(
                data[i.item() + 1 : i.item() + model_config.context_length + 1],
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
    best_checkpoint_path = os.path.join("checkpoints", f"{train_config.name}.pt")

    # Either we train from scratch
    if train_config.train_from_scratch:
        print("Training model from scratch...")
        model = GPT(model_config)
        optimizer = model.configure_optimizers(
            train_config.weight_decay,
            train_config.learning_rate,
            train_config.betas
        )

    # Or we resume from the best previous checkpoint
    else:
        print(f"Resuming training model from {best_checkpoint_path}...")
        if not os.path.exists(best_checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {best_checkpoint_path}")

        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model_config = checkpoint["model_config"]
        train_config = checkpoint["train_config"]

        current_iteration = checkpoint["iteration"]
        best_val_loss = checkpoint["best_val_loss"]

        model = GPT(model_config)
        model.load_state_dict(checkpoint["model"])
        optimizer = model.configure_optimizers(
            train_config.weight_decay,
            train_config.learning_rate,
            train_config.betas,
            device_type=device,
        )
        # Free up memory
        checkpoint = None

    model.to(device=device)

    @torch.no_grad()
    @beartype
    def get_loss() -> dict[str, torch.Tensor]:
        """Calculates the average loss over the training and validation sets.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing average losses for 'train' and 'val' splits.
        """

        out = {}
        model.eval()
        for split in ["train", "val"]:
            # Evaluating on iters_per_eval batches
            losses = torch.zeros(train_config.iters_per_eval)
            for k in range(train_config.iters_per_eval):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            # Average loss over the evaluated batches
            out[split] = losses.mean()
        model.train()
        return out

    @beartype
    def get_learning_rate(iteration: int) -> float:
        """Calculates the learning rate based on the current iteration using a warmup and cosine decay schedule.

        Args:
            iteration (int): The current training iteration.
        Returns:
            float: The calculated learning rate.
        """

        # First some warmup
        if iteration < train_config.warmup_iters:
            return (
                train_config.learning_rate
                * (iteration + 1)
                / (train_config.warmup_iters + 1)
            )

        # Then cosine decay down to min_learning_rate
        else:
            decay_ratio = (iteration - train_config.warmup_iters) / (
                train_config.max_iters - train_config.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return train_config.min_learning_rate + coeff * (
                train_config.learning_rate - train_config.min_learning_rate
            )

    print("Starting training loop...")
    X, Y = get_batch("train")
    t0 = time.time()

    while current_iteration <= train_config.max_iters:

        # Setting the current learning rate
        lr = get_learning_rate(current_iteration)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation and checkpointing every eval_interval iterations
        if (
            current_iteration % train_config.eval_interval == 0
            and current_iteration > 0
        ):
            # Estimate time remaining
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            iters_left = train_config.max_iters - current_iteration
            sec_per_iter = (
                dt / train_config.eval_interval if current_iteration > 0 else 0
            )
            eta_sec = iters_left * sec_per_iter
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

            losses = get_loss()
            print(
                f"step {current_iteration}/{train_config.max_iters} | "
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
                    "model_config": model_config,
                    "train_config": train_config,
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
