from ..components.sources import *
from ..time import TimeInterval, PulseBase, Lifetime, parinit
from ..dynamic.operator.phonon_bath import Material
from typing import Union
from qutip import Qobj
from ..elements import Emitter
from .sources.base_source import GatedSourceComponent, infer_source_gate
from ..system import LindbladVector


class Source(SourceComponent):
    """
    A source factory
    """

    @classmethod
    def two_level(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                  parameters: dict = None, name: str = None, emitter_name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return TwoLevelSource(pulse=pulse, gate=gate, efficiency=efficiency, parameters=parameters, name=name,
                              emitter_name=emitter_name)

    @classmethod
    def purcell(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                purcell_factor: float = None, regime: float = None, timescale: float = None,
                parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return PurcellSource(pulse=pulse, gate=gate, efficiency=efficiency,
                             purcell_factor=purcell_factor, regime=regime, timescale=timescale,
                             parameters=parameters, name=name)

    @classmethod
    def phonon_assisted(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                        purcell_factor: float = 10, regime: float = 0.1, timescale: float = 1,
                        temperature: float = 4, material: Material = Material.ingaas_quantum_dot(),
                        resolution: int = 300, max_power: float = 30,
                        parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return PhononAssistedSource(pulse=pulse, gate=gate, efficiency=efficiency,
                                    purcell_factor=purcell_factor, regime=regime, timescale=timescale,
                                    temperature=temperature, material=material,
                                    resolution=resolution, max_power=max_power,
                                    parameters=parameters, name=name)

    @classmethod
    def exciton(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return ExcitonSource(pulse=pulse, gate=gate, efficiency=efficiency,
                             parameters=parameters, name=name)

    @classmethod
    def biexciton(cls, pulse: PulseBase = None, gate: Union[TimeInterval, list] = None, efficiency: float = 1,
                  parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return BiexcitonSource(pulse=pulse, gate=gate, efficiency=efficiency,
                               parameters=parameters, name=name)

    @classmethod
    def trion(cls, charge: str = 'negative', pulse: PulseBase = None, pulse_orthogonal: PulseBase = None,
              gate: Union[TimeInterval, list] = None, efficiency: float = 1, parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return TrionSource(charge=charge, pulse=pulse, pulse_orthogonal=pulse_orthogonal,
                           gate=gate, efficiency=efficiency,
                           parameters=parameters, name=name)

    @classmethod
    def fock(cls, state: Union[int, Qobj], gate: Union[TimeInterval, list] = None,
             shape: Union[PulseBase, Lifetime] = None, shape_resolution: int = 1000, efficiency: float = 1,
             parameters: dict = None, name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return FockSource(state=state, gate=gate, shape=shape, shape_resolution=shape_resolution,
                          efficiency=efficiency,
                          parameters=parameters, name=name)

    @classmethod
    def shaped_laser(cls,
                     shape: Union[PulseBase, Lifetime, callable] = None,
                     resolution: int = 1000,
                     truncation: int = 2,
                     gate: Union[TimeInterval, list] = None,
                     efficiency: float = 1,
                     parameters: dict = None,
                     name: str = None):
        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        return ShapedLaserSource(shape=shape, resolution=resolution, truncation=truncation, gate=gate,
                                 efficiency=efficiency,
                                 parameters=parameters, name=name)

    @classmethod
    def perceval(cls,
                 emission_probability: float = 1,
                 multiphoton_component: float = 0,
                 indistinguishability: float = 1,
                 name: str = None):
        return DistinguishableNoiseSource(emission_probability=emission_probability,
                                          multiphoton_component=multiphoton_component,
                                          indistinguishability=indistinguishability,
                                          name=name)

    @classmethod
    def from_master_equation(cls,
                             hamiltonian=None,
                             monitored=None,
                             environment=None,
                             initial_state: Union[Qobj, str] = None,
                             initial_time: Union[float, int] = None,
                             gate: Union[TimeInterval, list, callable] = None,
                             efficiency: float = 1,
                             parameters: dict = None,
                             states: dict = None,
                             operators: dict = None,
                             name: str = None,
                             emitter_name: str = None,
                             close_outputs: list = None,
                             mask_outputs: bool = False):
        if monitored is None:
            monitored_inputs = []
        elif isinstance(monitored, (list, tuple, LindbladVector)):
            monitored_inputs = monitored
        else:
            monitored_inputs = [monitored]

        monitored_modes = monitored_inputs.modes if isinstance(monitored_inputs, LindbladVector) else len(monitored_inputs)
        if monitored_modes == 0:
            raise ValueError("A source requires at least one monitored collapse operator.")

        efficiency = parinit({'efficiency': efficiency}, parameters)['efficiency']
        emitter = Emitter.from_master_equation(hamiltonian=hamiltonian,
                                               monitored=monitored_inputs,
                                               environment=environment,
                                               states=states,
                                               operators=operators,
                                               initial_state=initial_state,
                                               initial_time=initial_time,
                                               parameters=parameters,
                                               name=emitter_name)

        gate = infer_source_gate(emitter, parameters) if gate is None else gate
        source = GatedSourceComponent(emitter=emitter,
                                      gate=gate,
                                      efficiency=efficiency,
                                      parameters=parameters,
                                      name=name)

        for output in ([] if close_outputs is None else close_outputs):
            source.output.ports[output].close()
        if mask_outputs:
            source.mask()

        return source
