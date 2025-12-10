import pytest
from src.data_preprocessing import encode, decode
from src.config import VOCABULARY


def test_encode_decode_consistency():
    """Test that encoding followed by decoding returns the original text,
    and that the encoded output is a list of integers."""

    original_text = "Hello, World!"
    # Ensure all chars are in vocabulary for this test
    original_text = "".join([c for c in original_text if c in VOCABULARY])

    encoded = encode(original_text)
    decoded = decode(encoded)

    assert decoded == original_text
    assert isinstance(encoded, list)
    assert isinstance(encoded[0], int)


def test_vocabulary_completeness():
    """Ensure vocabulary is not empty."""
    assert len(VOCABULARY) > 0
