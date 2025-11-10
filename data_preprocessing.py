import os
import requests
from beartype import beartype
import numpy as np

# Download the Tiny Shakespeare dataset if not already present
tinyshakespeare_path = "./data/tinyshakespeare.txt"
if not os.path.exists(tinyshakespeare_path):
    tinyshakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(tinyshakespeare_path, "w") as f:
        f.write(requests.get(tinyshakespeare_url).text)

with open(tinyshakespeare_path, "r") as f:
    text = f.read()

# Analyze the dataset
characters = sorted(list(set(text)))
vocabulary_size = len(characters)
print(f"Length of dataset in characters: {len(text):,}")
print(f"All the unique characters in the dataset: {''.join(characters)}")
print(f"Vocabulary size: {vocabulary_size:,}")

# Encoding and decoding as the model can not work with raw text
@beartype
def encode(text: str) -> list[int]:
    '''Convert a string to a list of integers.
     Each character is mapped to a unique integer.
     For example, "abc" -> [0, 1, 2]'''
    
    string_to_integer = {char:i for i, char in enumerate(characters)}
    return [string_to_integer[char] for char in text]

@beartype
def decode(encoded: list[int]) -> str:
    '''Convert a list of integers to a string.
     Each integer is mapped to a unique character.
     For example, [0, 1, 2] -> "abc"'''
    integer_to_string = {i:char for i, char in enumerate(characters)}
    return ''.join([integer_to_string[i] for i in encoded])

# Split the dataset into training and validation sets
# Validation set comes after the training set to avoid data leakage
train_text = text[:int(0.9*len(text))]
val_text = text[int(0.9*len(text)):]
train_data = encode(train_text)
val_data = encode(val_text)

print(f"Number of tokens in training data: {len(train_data):,}")
print(f"Number of tokens in validation data: {len(val_data):,}")

# Save the encoded datasets as binary files
train_data = np.array(train_data, dtype=np.uint16)
val_data = np.array(val_data, dtype=np.uint16)
train_data.tofile("./data/tinyshakespeare_train.bin")
val_data.tofile("./data/tinyshakespeare_val.bin")