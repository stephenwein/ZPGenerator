from ...elements import Emitter
from .base_source import GatedSourceComponent, source_from_emitter
from ...time import TimeInterval, Operator, PulseBase
from ...time.parameters import parinit
from ...dynamic.control import Control
from ...dynamic.pulse import Pulse
from typing import Union
from numpy import sqrt, pi
from qutip import tensor


class TrionCavitySource(GatedSourceComponent):
    """
    A trion source coupled to orthogonally polarised cavity modes.
    """

    def __init__(self,
                 charge: str = 'negative',
                 pulse: PulseBase = None,
                 pulse_orthogonal: PulseBase = None,
                 gate: Union[TimeInterval, list, callable] = None,
                 efficiency: float = 1,
                 truncation: int = 2,
                 parameters: dict = None,
                 name: str = None):
        emitter = Emitter.trion_cavity(charge=charge, truncation=truncation, parameters=parameters)
        pulse = Pulse.dirac(parameters=parameters) if pulse is None else pulse

        trion = emitter.subsystems['trion']
        right = emitter.operators['lower_R']
        left = emitter.operators['lower_L']
        horizontal = (right + left) / sqrt(2)
        vertical = 1.j * (right - left) / sqrt(2)
        params = parinit({'theta': pi / 4, 'phi': -pi / 2}, parameters)
        dipole = Operator.polarised(horizontal, vertical, parameters=params)
        emitter.add(Control.drive(pulse=pulse, transition=dipole))

        if pulse_orthogonal:
            dipole_orthogonal = Operator.polarised_orthogonal(horizontal, vertical, parameters=params)
            emitter.add(Control.drive(pulse=pulse_orthogonal, transition=dipole_orthogonal))

        emitter.initial_state = (
            tensor(emitter.subsystems['cavity_h'].states['|0>'],
                   emitter.subsystems['cavity_v'].states['|0>'],
                   trion.states['|spin_down>']) *
            tensor(emitter.subsystems['cavity_h'].states['|0>'],
                   emitter.subsystems['cavity_v'].states['|0>'],
                   trion.states['|spin_down>']).dag() +
            tensor(emitter.subsystems['cavity_h'].states['|0>'],
                   emitter.subsystems['cavity_v'].states['|0>'],
                   trion.states['|spin_up>']) *
            tensor(emitter.subsystems['cavity_h'].states['|0>'],
                   emitter.subsystems['cavity_v'].states['|0>'],
                   trion.states['|spin_up>']).dag()
        ) / 2

        gate = TimeInterval.source_gate(pulse, parameters=parameters) if gate is None else gate

        source = source_from_emitter(emitter=emitter,
                                     gate=gate,
                                     efficiency=efficiency,
                                     parameters=parameters,
                                     name=name,
                                     close_outputs=[2, 3],
                                     mask_outputs=True)
        self.__dict__ = source.__dict__
        self.default_name = '_TrionCavity'
