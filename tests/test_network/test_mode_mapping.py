import pytest

from zpgenerator.network.mode_mapping import ModeMapping, normalize_mode_position


def test_normalize_mode_position_accepts_int_and_str():
    assert normalize_mode_position(2) == 2
    assert normalize_mode_position("left") == "left"


def test_normalize_mode_position_rejects_invalid_values():
    with pytest.raises(ValueError, match="non-negative"):
        normalize_mode_position(-1)
    with pytest.raises(ValueError, match="non-empty"):
        normalize_mode_position("")
    with pytest.raises(TypeError, match="integer index or named port string"):
        normalize_mode_position(1.5)
    with pytest.raises(TypeError, match="integer index or named port string"):
        normalize_mode_position(True)


def test_mode_mapping_from_input_supports_single_and_multiple_positions():
    assert ModeMapping.from_input(0).positions == (0,)
    assert ModeMapping.from_input("left").positions == ("left",)
    assert ModeMapping.from_input([0, 2, "right"]).positions == (0, 2, "right")
    assert ModeMapping.from_input((1, "left")).positions == (1, "left")


def test_mode_mapping_from_input_rejects_empty_mapping():
    with pytest.raises(ValueError, match="cannot be empty"):
        ModeMapping.from_input([])


def test_mode_mapping_single_requires_exactly_one_position():
    with pytest.raises(ValueError, match="exactly one position"):
        _ = ModeMapping.from_input([0, 1]).single
