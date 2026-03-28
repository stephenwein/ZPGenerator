from zpgenerator.components import Source
from zpgenerator.components.sources import PhononAssistedSource, PurcellSource
from zpgenerator.dynamic import Pulse
from zpgenerator.elements import TwoLevelEmitter, CavityEmitter, Emitter
from zpgenerator.system import CouplingBase, MultiBodyEmitter
from zpgenerator.time import TimeInterval


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
    assert factory_source.output.port_names == ['direct', 'cavity']
    assert factory_source.output.open_port_names == ['cavity']


def test_phonon_assisted_factory_matches_component_semantics():
    factory_source = Source.phonon_assisted()
    component_source = PhononAssistedSource()

    assert factory_source.is_masked == component_source.is_masked
    assert [port.is_closed for port in factory_source.output.ports] == [port.is_closed for port in component_source.output.ports]
    assert [port.is_closed for port in factory_source.input.ports] == [port.is_closed for port in component_source.input.ports]
    assert factory_source.output.port_names == ['direct', 'cavity']
    assert factory_source.output.open_port_names == ['cavity']


def test_phonon_assisted_does_not_mutate_passed_pulse_parameters():
    pulse = Pulse.gaussian(parameters={'detuning': 1.5, 'width': 2})
    PhononAssistedSource(pulse=pulse)

    assert 'detuning' in pulse.parameters
    assert '_detuning' not in pulse.parameters


def test_purcell_emitter_keyword_defaults_still_overwrite_explicit_dependents():
    emitter = Emitter.purcell(purcell_factor=5, regime=0.2, timescale=2)

    parameters = emitter.set_parameters({'coupling': 999}).dict

    assert parameters['coupling'] != 999
    assert parameters['coupling'] == emitter.default_parameters['coupling']


def test_purcell_gate_insert_preserves_explicit_rate():
    pulse = Pulse.dirac()
    defaults = PurcellSource().default_parameters
    gate = TimeInterval.source_gate(pulse, parameters=defaults, parameter_name='_purcell_rate')
    gate.create_insert_parameter_function(PurcellSource._purcell_rate)

    derived_rate = PurcellSource._purcell_rate(defaults)['_purcell_rate']

    assert gate.get_parameters()['_purcell_rate'] == derived_rate


def test_inserted_public_purcell_rate_can_be_overridden_explicitly():
    pulse = Pulse.dirac()
    gate = TimeInterval.source_gate(
        pulse,
        parameters={'purcell_rate': 1, 'coupling': 5},
        parameter_name='purcell_rate',
    )
    gate.create_insert_parameter_function(lambda args: {'purcell_rate': args['coupling']})

    assert gate.get_parameters()['purcell_rate'] == 5
    assert gate.get_parameters({'purcell_rate': 123})['purcell_rate'] == 123
