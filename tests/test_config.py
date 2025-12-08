import pytest
import os
from config import (
    validate_model_config,
    validate_train_config,
    validate_sample_config,
    validate_quantization_config,
)


def test_complete_config():
    """Test that all configuration validations pass."""
    validate_model_config()
    validate_train_config()
    validate_sample_config()
    validate_quantization_config()
