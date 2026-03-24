import numpy as np
import pytest
from qutip import Qobj, basis

from zpgenerator.simulate._processor_helpers import (
    build_channels_from_results,
    build_simulation_context,
    invert_tensors,
    prepare_channel_basis,
)


def test_build_simulation_context_normalises_nested_values():
    parameters = {
        "alpha": np.array([1, 2]),
        "nested": {"beta": [3, 4]},
    }
    context = build_simulation_context(parameters=parameters, bin_list=["B0"], basis=[Qobj([[1]])])
    assert context["bin_list"] == ("B0",)
    assert context["parameters"][0][0] == "alpha"
    assert context["parameters"][0][1][0] == "ndarray"


def test_prepare_channel_basis_requires_non_empty_basis():
    with pytest.raises(ValueError, match="Please provide a state basis"):
        prepare_channel_basis([])


def test_prepare_channel_basis_builds_outer_products():
    psi0 = basis(2, 0)
    psi1 = basis(2, 1)
    expanded = prepare_channel_basis([psi0, psi1])
    assert len(expanded) == 4
    assert expanded[0].shape == (2, 2)


def test_invert_tensors_accumulates_with_logical_or():
    class DummyTensor:
        def __init__(self, value):
            self.value = value

        def invert(self):
            return self.value

    assert invert_tensors([DummyTensor(False), DummyTensor(False)]) is False
    assert invert_tensors([DummyTensor(False), DummyTensor(True), DummyTensor(False)]) is True


def test_build_channels_from_results_does_not_mutate_source_state_dims():
    source_state = Qobj([[1, 0], [0, 0]], dims=[[2], [2]])
    results = [{(0,): source_state} for _ in range(4)]
    channel_basis = prepare_channel_basis([basis(2, 0), basis(2, 1)])

    channels = build_channels_from_results(results=results, basis=channel_basis, dims=[1, 2], select=None)

    assert source_state.dims == [[2], [2]]
    assert (0,) in channels
    assert channels[(0,)].shape == (4, 4)
