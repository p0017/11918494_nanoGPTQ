import pytest
import os
from src.config import (
    validate_model_config,
    validate_train_config,
    validate_sample_config,
    validate_quantization_config,
    validate_evaluation_config,
)


def test_complete_config():
    """Test that all configuration validations pass."""
    validate_model_config()
    validate_train_config()
    validate_sample_config()
    validate_quantization_config()
    validate_evaluation_config()
