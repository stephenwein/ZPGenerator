from zpgenerator.components import Source
from zpgenerator.components.sources import PhononAssistedSource, PurcellSource
from zpgenerator.elements import TwoLevelEmitter, CavityEmitter, Emitter
from zpgenerator.system import CouplingBase, MultiBodyEmitter


def test_source_init():
    emitter = TwoLevelEmitter()
    cavity = CavityEmitter()
    coupling = CouplingBase.jaynes_cummings(emitter.operators['lower'], cavity.operators['annihilation'])
    assert emitter.subdims == [2]
    assert cavity.subdims == [2]
    assert coupling.subdims == [2, 2]

    coupling2 = CouplingBase()
    coupling2.add(coupling)
    assert coupling2.subdims == [2, 2]

    purcell = MultiBodyEmitter(subsystems=[emitter, cavity], coupling=coupling)
    assert purcell.coupling.subdims == [2, 2]

    emitter = Emitter.purcell()
    assert emitter.subdims == [2, 2]

    source = Source.purcell()
    assert source.subdims == [2, 2]


def test_purcell_factory_matches_component_semantics():
    factory_source = Source.purcell()
    component_source = PurcellSource()

    assert factory_source.is_masked == component_source.is_masked
    assert [port.is_closed for port in factory_source.output.ports] == [port.is_closed for port in component_source.output.ports]
    assert [port.is_closed for port in factory_source.input.ports] == [port.is_closed for port in component_source.input.ports]


def test_phonon_assisted_factory_matches_component_semantics():
    factory_source = Source.phonon_assisted()
    component_source = PhononAssistedSource()

    assert factory_source.is_masked == component_source.is_masked
    assert [port.is_closed for port in factory_source.output.ports] == [port.is_closed for port in component_source.output.ports]
    assert [port.is_closed for port in factory_source.input.ports] == [port.is_closed for port in component_source.input.ports]



