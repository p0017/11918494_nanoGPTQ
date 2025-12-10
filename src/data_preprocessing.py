import os
import requests
from beartype import beartype
import numpy as np
from src.config import VOCABULARY


# Encoding and decoding as the model can not work with raw text
@beartype
def encode(text: str) -> list[int]:
    """Convert a string to a list of integers.
    Each character is mapped to a unique integer.
    For example, "abc" -> [0, 1, 2]

    Args:
        text (str): The input string to encode.
    Returns:
        list[int]: The encoded list of integers.
    """

    string_to_integer = {char: i for i, char in enumerate(VOCABULARY)}
    return [string_to_integer[char] for char in text]


@beartype
def decode(encoded: list[int]) -> str:
    """Convert a list of integers to a string.
    Each integer is mapped to a unique character.
    For example, [0, 1, 2] -> "abc"

    Args:
        encoded (list[int]): The list of integers to decode.
    Returns:
        str: The decoded string.
    """

    integer_to_string = {i: char for i, char in enumerate(VOCABULARY)}
    return "".join([integer_to_string[i] for i in encoded])


if __name__ == "__main__":
    # Download the TinyStories V2 dataset if not already present
    tinystories_path = "./data/tinystories.txt"
    target_chars = int(10 * 1e6)  # Targeting 10 million characters

    if not os.path.exists(tinystories_path):
        print(f"Downloading TinyStories V2 (first {target_chars:,} characters)...")
        tinystories_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"

        # Stream the download so we dont download the full 2GB file
        response = requests.get(tinystories_url, stream=True)

        collected_text = []
        total_len = 0

        # Iterate over chunks of data
        for data_chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if data_chunk:
                text_chunk = data_chunk.decode("utf-8", errors="ignore")
                collected_text.append(text_chunk)
                total_len += len(text_chunk)

                if total_len >= target_chars:
                    break

        full_text = "".join(collected_text)[:target_chars]

        with open(tinystories_path, "w", encoding="utf-8") as f:
            f.write(full_text)

    with open(tinystories_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Analyze the dataset
    characters = sorted(list(set(text)))
    vocabulary_size = len(characters)
    print(f"Length of dataset in characters: {len(text):,}")
    print(f"All the unique characters in the dataset: {''.join(characters)}")
    print(f"Vocabulary size: {vocabulary_size:,}")
    assert all(
        c in VOCABULARY for c in characters
    ), "Update the VOCABULARY in config.py to include all above characters, then rerun this script."

    # Split the dataset into training, validation and test sets
    # Validation and test sets come after the training set to avoid data leakage
    train_text = text[: int(0.8 * len(text))]
    val_text = text[int(0.8 * len(text)) : int(0.9 * len(text))]
    test_text = text[int(0.9 * len(text)) :]
    train_data = encode(train_text)
    val_data = encode(val_text)
    test_data = encode(test_text)

    print(f"Number of tokens in training data: {len(train_data):,}")
    print(f"Number of tokens in validation data: {len(val_data):,}")
    print(f"Number of tokens in test data: {len(test_data):,}")

    # Save the encoded datasets as binary files
    train_data = np.array(train_data, dtype=np.uint16)
    val_data = np.array(val_data, dtype=np.uint16)
    test_data = np.array(test_data, dtype=np.uint16)
    train_data.tofile("./data/tinystories_train.bin")
    val_data.tofile("./data/tinystories_val.bin")
    test_data.tofile("./data/tinystories_test.bin")
