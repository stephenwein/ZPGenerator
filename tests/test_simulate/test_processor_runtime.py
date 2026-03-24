import pytest
from qutip import basis

from zpgenerator.network import DetectorGate, TimeBin
from zpgenerator.simulate._processor_runtime import (
    assert_continue_allowed,
    build_propagation_times,
    initialise_or_resume_grove,
)
from zpgenerator.virtual import MeasurementBranch


def test_assert_continue_allowed_requires_existing_grove():
    with pytest.raises(RuntimeError, match="No simulation to continue"):
        assert_continue_allowed(grove=None, current_time=0.0, existing_context=None, simulation_context=None)


def test_assert_continue_allowed_requires_current_time():
    with pytest.raises(RuntimeError, match="No current time"):
        assert_continue_allowed(grove=object(), current_time=None, existing_context=None, simulation_context=None)


def test_assert_continue_allowed_rejects_context_changes():
    with pytest.raises(ValueError, match="Cannot continue simulation"):
        assert_continue_allowed(
            grove=object(),
            current_time=0.0,
            existing_context={"parameters": ("a", 1)},
            simulation_context={"parameters": ("a", 2)},
        )


def test_initialise_or_resume_grove_initializes_when_no_current_time():
    branches = [MeasurementBranch(time_bins=[TimeBin(detector=DetectorGate(resolution=None))])]
    current_time, grove, branch_order = initialise_or_resume_grove(
        current_time=None,
        grove=None,
        branch_order=[],
        initial_time=0.0,
        basis_states=[basis(2, 0)],
        branches=branches,
    )
    assert current_time == 0.0
    assert grove is not None
    assert isinstance(branch_order, list)


def test_initialise_or_resume_grove_reuses_existing_state():
    grove = object()
    branch_order = [0, 1]
    current_time, returned_grove, returned_order = initialise_or_resume_grove(
        current_time=1.2,
        grove=grove,
        branch_order=branch_order,
        initial_time=0.0,
        basis_states=[],
        branches=[],
    )
    assert current_time == 1.2
    assert returned_grove is grove
    assert returned_order == branch_order


def test_build_propagation_times_filters_internal_grid():
    times = build_propagation_times(current_time=1.0, component_times=[0.5, 1.0, 1.5, 2.0, 2.5], final_time=2.2)
    assert times == [1.0, 1.5, 2.0, 2.2]
