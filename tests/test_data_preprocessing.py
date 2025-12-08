import pytest
from data_preprocessing import encode, decode
from config import VOCABULARY


def test_encode_decode_consistency():
    """Test that encoding and then decoding returns the original string."""
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
