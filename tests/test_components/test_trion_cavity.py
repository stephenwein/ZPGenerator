from zpgenerator.components import Source
from zpgenerator.components.sources import TrionCavitySource
from zpgenerator.elements import Emitter
from zpgenerator.dynamic import Pulse
import pytest


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
    assert defaults['coupling_h'] == pytest.approx(1 / 2)
    assert defaults['coupling_v'] == pytest.approx(1 / 2)
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
