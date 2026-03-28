from zpgenerator.elements import Emitter
from zpgenerator.components.sources import SourceComponent
from zpgenerator.components import Source, Circuit
from zpgenerator.dynamic import Pulse
from zpgenerator.time import TimeOperator
from qutip import destroy, num
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


def test_source_from_master_equation_requires_monitored_channel():
    with pytest.raises(ValueError, match="monitored collapse operator"):
        Source.from_master_equation(hamiltonian=num(2), gate=[0, 1])


def test_source_from_master_equation_infers_gate_from_finite_time_support():
    pulse = Pulse.gaussian(parameters={'width': 1})
    hamiltonian = TimeOperator(operator=num(2), functions=pulse)

    source = Source.from_master_equation(
        hamiltonian=hamiltonian,
        monitored=[destroy(2)],
        initial_state=None,
        infer_gate=True,
    )

    times = source.times()
    assert min(times) in times
    assert max(times) in times
    assert source.input.open_modes == 0
    assert source.output.open_modes == 1


def test_source_from_master_equation_requires_explicit_gate_for_time_independent_models():
    with pytest.raises(ValueError, match="explicit gate"):
        Source.from_master_equation(
            hamiltonian=num(2),
            monitored=[destroy(2)],
        )


def test_source_from_master_equation_requires_opt_in_for_gate_inference():
    pulse = Pulse.gaussian(parameters={'width': 1})
    hamiltonian = TimeOperator(operator=num(2), functions=pulse)

    with pytest.raises(ValueError, match="infer_gate=True"):
        Source.from_master_equation(
            hamiltonian=hamiltonian,
            monitored=[destroy(2)],
        )


def test_source_from_master_equation_supports_port_layout_policy():
    source = Source.from_master_equation(
        hamiltonian=num(2),
        monitored=[destroy(2), destroy(2)],
        gate=[0, 1],
        close_outputs=[0],
        mask_outputs=True,
    )

    assert source.output.ports[0].is_closed


def test_source_from_master_equation_propagates_named_monitored_channels():
    source = Source.from_master_equation(
        hamiltonian=num(2),
        monitored={'signal': destroy(2), 'idler': destroy(2)},
        gate=[0, 1],
        close_outputs=['signal'],
        mask_outputs=True,
    )

    assert source.output.port_names == ['signal', 'idler']
    assert source.output.open_port_names == ['idler']
    assert source.output.ports[0].is_closed


def test_catalogue_sources_expose_stable_optional_port_names():
    two_level = Source.two_level()
    exciton = Source.exciton()
    biexciton = Source.biexciton()
    trion = Source.trion()

    assert two_level.output.port_names == ['direct']
    assert exciton.output.port_names == ['x', 'y']
    assert biexciton.output.port_names == ['x', 'y', 'bx', 'by']
    assert trion.output.port_names == ['direct_h', 'direct_v']


def test_circuit_from_perceval_requires_compute_unitary():
    with pytest.raises(TypeError, match="compute_unitary"):
        Circuit.from_perceval(object())
