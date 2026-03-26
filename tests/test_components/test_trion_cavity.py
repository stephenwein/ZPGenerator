from zpgenerator.components import Source
from zpgenerator.components.sources import TrionCavitySource
from zpgenerator.elements import Emitter


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
