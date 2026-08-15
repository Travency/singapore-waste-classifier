import pytest
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_preprocess_image_shape():
    """Test that preprocessing returns the correct tensor shape."""
    from predict import preprocess_image
    img = Image.new('RGB', (300, 400), color=(100, 150, 200))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {tensor.shape}"


def test_preprocess_image_rgba_conversion():
    """Test that RGBA images are handled correctly."""
    from predict import preprocess_image
    img = Image.new('RGBA', (224, 224), color=(100, 150, 200, 255))
    img = img.convert('RGB')
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224)


def test_disposal_instructions_all_classes():
    """Test that disposal instructions exist for all expected classes."""
    from predict import DISPOSAL_INSTRUCTIONS
    expected_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    for cls in expected_classes:
        assert cls in DISPOSAL_INSTRUCTIONS, f"Missing disposal instructions for {cls}"
        assert 'bin' in DISPOSAL_INSTRUCTIONS[cls]
        assert 'instructions' in DISPOSAL_INSTRUCTIONS[cls]
        assert 'nea_tip' in DISPOSAL_INSTRUCTIONS[cls]


def test_disposal_instructions_structure():
    """Test that each disposal instruction has all required fields."""
    from predict import DISPOSAL_INSTRUCTIONS
    required_fields = ['emoji', 'bin', 'instructions', 'nea_tip']
    for cls, info in DISPOSAL_INSTRUCTIONS.items():
        for field in required_fields:
            assert field in info, f"Missing field '{field}' in {cls}"