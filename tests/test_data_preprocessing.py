import pytest
import numpy as np

from data_preprocessing import encode, decode, train_data, val_data, vocabulary_size, text


def test_encode_decode_consistency():
    '''Test that encoding followed by decoding returns the original string.'''
    sample = "Hello, world!"
    encoded = encode(sample)
    decoded = decode(encoded)
    assert decoded == sample, "Decoded string does not match original"
    assert all(0 <= i < vocabulary_size for i in encoded), "Encoded values out of vocab range"

def test_encode_type_error():
    '''Test that encoding a non-string raises a TypeError.'''
    with pytest.raises(TypeError):
        encode(123)  # not a string

def test_train_val_split():
    '''Test that the training and validation data split is correct.'''
    total_len = len(train_data) + len(val_data)
    assert total_len == len(text), "Train + validation length does not match total text length"
