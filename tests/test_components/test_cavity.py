from zpgenerator.elements.nonlinear.cavity import ShapedCavitySystem
import pytest


def test_shaped_cavity_rejects_unsupported_shape_type():
    with pytest.raises(TypeError, match="requested pulse shape"):
        ShapedCavitySystem(shape="not-a-shape")
