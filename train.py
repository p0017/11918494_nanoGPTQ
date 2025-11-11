import os
import numpy as np
import torch

from config import GPTConfig
from model import GPT

config = GPTConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split: str):
    if split == 'train':
        path = "./data/tinyshakespeare_train.bin"
    elif split == 'val':
        path = "./data/tinyshakespeare_val.bin"
    elif split == 'test':
        path = "./data/tinyshakespeare_test.bin"
    else:
        raise ValueError("Invalid split name. Use 'train', 'val' or 'test'.")
    
    # Loading the entire dataset into RAM, since its only about 2MB
    data = np.fromfile(path, dtype=np.uint16)
    # Randomly select starting indices for the batch
    starting_index = torch.randint(len(data) - config.context_length, size=(config.batch_size,))
    # For each starting index, extract input sequences of length context_length
    # And the corresponding target sequences shifted by one position
    input = torch.stack([torch.from_numpy(data[i:i+config.context_length]) for i in starting_index])
    target = torch.stack([torch.from_numpy(data[i+1:i+config.context_length+1]) for i in starting_index])

    if device == 'cuda':
        # Pin memory for faster transfer from CPU to GPU
        input, target = input.pin_memory().to('cuda', non_blocking=True), target.pin_memory().to('cuda', non_blocking=True)
    else:
        input, target = input.to(device), target.to(device)
    return input, target

if __name__ == "__main__":
    current_iteration = 0
    best_val_loss = float('inf')

    if config.train_from_scratch:
        print("Training model from scratch...")
        model = GPT(config)
        optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, config.betas)

    else:
        print("Resuming training model from ./checkpoints/...")
        checkpoint_path = "./checkpoints/latest.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        current_iteration = checkpoint['iteration']
        best_val_loss = checkpoint['best_val_loss']

        model = GPT(config)
        model.load_state_dict(checkpoint['model'])
        optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, config.betas)