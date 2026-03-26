from ...elements import Emitter
from ...time import TimeInterval, Operator, PulseBase
from ...time.parameters import parinit
from ...dynamic.control import Control
from ...dynamic.pulse import Pulse
from typing import Union
from .base_source import GatedSourceComponent, source_from_emitter


class ExcitonSource(GatedSourceComponent):
    """
    An exciton emitter driven by a single polarized pulse
    """

    def __init__(self,
                 pulse: PulseBase = None,
                 gate: Union[TimeInterval, list] = None,
                 efficiency: float = 1,
                 parameters: dict = None,
                 name: str = None):
        emitter = Emitter.exciton(parameters=parameters)
        pulse = Pulse.dirac(parameters=parameters) if pulse is None else pulse

        dipole = Operator.polarised(emitter.operators['lower_x'],
                                    emitter.operators['lower_y'],
                                    parinit({'theta': 0, 'phi': 0}, parameters))

        emitter.add(Control.drive(pulse=pulse, transition=dipole))
        gate = TimeInterval.source_gate(pulse, parameters=parameters) if gate is None else gate

        emitter.initial_state = emitter.states['|g>']

        source = source_from_emitter(emitter=emitter,
                                     gate=gate,
                                     efficiency=efficiency,
                                     parameters=parameters,
                                     name=name)
        self.__dict__ = source.__dict__
        self.default_name = '_Exciton'
