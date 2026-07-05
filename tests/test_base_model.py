import pytest

from training.models import BaseModel


def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel()
