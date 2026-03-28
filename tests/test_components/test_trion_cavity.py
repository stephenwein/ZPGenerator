from zpgenerator.components import Source
from zpgenerator.components.sources import TrionCavitySource
from zpgenerator.elements import Emitter
from zpgenerator.dynamic import Pulse
import pytest
from math import sqrt


def test_trion_cavity_emitter_builds_exact_tensor_model():
    emitter = Emitter.trion_cavity()

    assert emitter.subdims == [2, 2, 2, 2]
    assert emitter.dim == 16
    assert emitter.modes == 4
    assert emitter.coupling.subdims == [2, 2, 2, 2]
    assert set(emitter.subsystems) == {'trion', 'cavity_h', 'cavity_v'}


def test_trion_cavity_source_exposes_only_cavity_outputs():
    source = Source.trion_cavity()

    assert source.is_masked
    assert source.input.open_modes == 0
    assert source.output.open_modes == 2


def test_trion_cavity_factory_matches_component_semantics():
    factory_source = Source.trion_cavity()
    component_source = TrionCavitySource()

    assert factory_source.is_masked == component_source.is_masked
    assert [port.is_closed for port in factory_source.output.ports] == \
        [port.is_closed for port in component_source.output.ports]
    assert [port.is_closed for port in factory_source.input.ports] == \
        [port.is_closed for port in component_source.input.ports]


def test_trion_cavity_source_supports_port_resolved_quality_metrics():
    source = Source.trion_cavity(pulse=Pulse.gaussian(parameters={'width': 1, 'area': 3.14159}))

    assert source.beta(0) > 0
    assert source.beta(1) > 0


def test_trion_cavity_keyword_defaults_derive_mode_parameters():
    emitter = Emitter.trion_cavity(
        purcell_factor_h=4,
        purcell_factor_v=6,
        regime_h=0.2,
        regime_v=0.3,
        timescale=2,
    )

    defaults = emitter.default_parameters

    assert defaults['trion/decay'] == pytest.approx(1 / 20)
    assert defaults['coupling_h'] == pytest.approx(1 / sqrt(2))
    assert defaults['coupling_v'] == pytest.approx(1 / sqrt(2))
    assert defaults['cavity_h/decay'] == pytest.approx(5)
    assert defaults['cavity_v/decay'] == pytest.approx(10 / 3)


def test_trion_cavity_keyword_defaults_still_overwrite_explicit_dependents():
    emitter = Emitter.trion_cavity(purcell_factor_h=4, purcell_factor_v=4, regime=0.2, timescale=2)

    parameters = emitter.set_parameters({'coupling_h': 999, 'coupling_v': 999}).dict

    assert parameters['coupling_h'] == emitter.default_parameters['coupling_h']
    assert parameters['coupling_v'] == emitter.default_parameters['coupling_v']


def test_trion_cavity_default_gate_tracks_timescale_like_purcell_source():
    fast = Source.trion_cavity(timescale=1)
    slow = Source.trion_cavity(timescale=2)

    assert max(slow.times()) > max(fast.times())
    assert max(slow.times()) == pytest.approx(2 * max(fast.times()))


def test_trion_cavity_total_purcell_factor_splits_symmetrically_between_modes():
    split = Emitter.trion_cavity(purcell_factor=10, regime=0.05, timescale=200)
    explicit = Emitter.trion_cavity(purcell_factor_h=5, purcell_factor_v=5, regime_h=0.05, regime_v=0.05,
                                    timescale=200)

    assert split.default_parameters['trion/decay'] == pytest.approx(explicit.default_parameters['trion/decay'])
    assert split.default_parameters['coupling_h'] == pytest.approx(explicit.default_parameters['coupling_h'])
    assert split.default_parameters['coupling_v'] == pytest.approx(explicit.default_parameters['coupling_v'])
    assert split.default_parameters['cavity_h/decay'] == pytest.approx(explicit.default_parameters['cavity_h/decay'])
    assert split.default_parameters['cavity_v/decay'] == pytest.approx(explicit.default_parameters['cavity_v/decay'])


def test_trion_cavity_single_mode_detuning_matches_split_total_purcell_convention():
    source = Source.trion_cavity(
        pulse=Pulse.gaussian(parameters={'area': 1.4142135623730951 * 3.141592653589793, 'width': 10}),
        purcell_factor=10,
        timescale=200,
        regime=0.05,
        parameters={'theta': 0, 'phi': 0},
    )

    total_brightness = (
        source.beta(0, parameters={'cavity_h/resonance': 20}) +
        source.beta(1, parameters={'cavity_h/resonance': 20})
    )

    assert total_brightness == pytest.approx(5 / 6, rel=2e-2)


def test_trion_cavity_explicit_mode_purcell_factors_preserve_stronger_single_mode_collection():
    source = Source.trion_cavity(
        pulse=Pulse.gaussian(parameters={'area': 1.4142135623730951 * 3.141592653589793, 'width': 10}),
        purcell_factor_h=10,
        purcell_factor_v=10,
        timescale=200,
        regime_h=0.05,
        regime_v=0.05,
        parameters={'theta': 0, 'phi': 0},
    )

    total_brightness = (
        source.beta(0, parameters={'cavity_h/resonance': 20}) +
        source.beta(1, parameters={'cavity_h/resonance': 20})
    )

    assert total_brightness == pytest.approx(10 / 11, rel=2e-2)
