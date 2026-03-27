from ...elements import Emitter
from .base_source import GatedSourceComponent, rate_gate_from_pulse, source_from_emitter
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
                 purcell_factor: float = None,
                 regime: float = None,
                 timescale: float = None,
                 purcell_factor_h: float = None,
                 purcell_factor_v: float = None,
                 regime_h: float = None,
                 regime_v: float = None,
                 parameters: dict = None,
                 name: str = None):
        emitter = Emitter.trion_cavity(charge=charge,
                                       truncation=truncation,
                                       purcell_factor=purcell_factor,
                                       regime=regime,
                                       timescale=timescale,
                                       purcell_factor_h=purcell_factor_h,
                                       purcell_factor_v=purcell_factor_v,
                                       regime_h=regime_h,
                                       regime_v=regime_v,
                                       parameters=parameters)
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

        if gate is None:
            gate = rate_gate_from_pulse(
                pulse,
                rate_function=self._trion_cavity_rate,
                parameter_name='_trion_cavity_rate',
                pulse_parameters=emitter.default_parameters | (parameters if parameters else {}),
            )

        source = source_from_emitter(emitter=emitter,
                                     gate=gate,
                                     efficiency=efficiency,
                                     parameters=emitter.default_parameters | (parameters if parameters else {}),
                                     name=name,
                                     close_outputs=[2, 3],
                                     mask_outputs=True)
        self.__dict__ = source.__dict__
        self.default_name = '_TrionCavity'

    @staticmethod
    def _mode_purcell_rate(args: dict, cavity: str):
        kappa = args[f'{cavity}/decay']
        gamma = args['trion/decay'] + 2 * args.get('trion/dephasing', 0)
        delta = args['trion/resonance'] - args[f'{cavity}/resonance']
        coupling = args[f'coupling_{cavity[-1]}']
        rate = 4 * coupling ** 2 * (kappa + gamma) / ((kappa + gamma) ** 2 + 4 * delta ** 2)
        return (rate * kappa) / (rate + kappa)

    @classmethod
    def _trion_cavity_rate(cls, args: dict):
        return {
            '_trion_cavity_rate':
                cls._mode_purcell_rate(args, 'cavity_h') + cls._mode_purcell_rate(args, 'cavity_v')
        }
