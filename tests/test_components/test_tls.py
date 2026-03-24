from zpgenerator.elements import Emitter
from zpgenerator.components.sources import SourceComponent
from zpgenerator.components import Source, Circuit
from zpgenerator.dynamic import Pulse
import pytest


def test_initial_state_tls_source():
    emitter = Emitter.two_level()
    emitter.initial_state = emitter.states['|g>']
    source = SourceComponent(emitter)
    assert source.initial_state == emitter.states['|g>']
    assert source.subdims == [2]

    source = Source.two_level()
    assert source.subdims == [2]


def test_fock_source_preserves_explicit_gate_when_shape_is_provided():
    source = Source.fock(1, gate=[1, 2], shape=Pulse.gaussian())
    times = source.times()

    assert 1 in times
    assert 2 in times


def test_circuit_phase_shifter_preserves_explicit_parameters():
    circuit = Circuit.ps(parameters={'phase': 0.3})

    assert circuit.default_parameters['phase'] == 0.3


def test_source_quality_processor_invalidates_after_topology_change():
    source = Source.fock(1)
    source.mu()
    old_processor = source._quality_processor

    source.add(0, Circuit.bs())

    assert source._quality_processor is None
    source.mu(1)
    assert source._quality_processor is not None
    assert source._quality_processor is not old_processor


def test_source_quality_processor_invalidates_after_parameter_update():
    source = Source.two_level()
    source.mu()
    old_processor = source._quality_processor

    source.update_default_parameters({'efficiency': 0.5})

    assert source._quality_processor is None
    source.mu()
    assert source._quality_processor is not None
    assert source._quality_processor is not old_processor


def test_circuit_from_perceval_requires_compute_unitary():
    with pytest.raises(TypeError, match="compute_unitary"):
        Circuit.from_perceval(object())
