from ...elements import Emitter
from ...time import TimeInterval, PulseBase, parinit
from ...dynamic import Control, Pulse
from ...dynamic.operator.phonon_bath import Material, PhononBath
from .base_source import GatedSourceComponent, rate_gate_from_pulse, source_from_emitter
from typing import Union
from copy import deepcopy


class PhononAssistedSource(GatedSourceComponent):
    """
    A two-level emitter coupled to a phonon bath driven by a single pulse with emission coupled to a cavity mode
    """

    def __init__(self,
                 pulse: PulseBase = None,
                 gate: Union[TimeInterval, list] = None,
                 efficiency: float = 1,
                 purcell_factor: float = 10,
                 regime: float = 0.1,
                 timescale: float = 1,
                 temperature: float = 4,
                 material: Material = Material.ingaas_quantum_dot(),
                 resolution: int = 300,
                 max_power: float = 30,
                 parameters: dict = None,
                 name: str = None):

        emitter = Emitter.purcell(purcell_factor=purcell_factor, regime=regime, timescale=timescale,
                                  parameters=parameters)

        pulse = deepcopy(pulse) if pulse else Pulse.gaussian(parameters=parameters)
        pulse.rename_parameter('detuning', '_detuning')

        self.bath = PhononBath(material=material, temperature=temperature, resolution=resolution, max_power=max_power)
        self.bath.initialize()

        tls = emitter.subsystems['emitter']
        tls.add(Control.drive(pulse=pulse, transition=tls.operators['lower']))
        tls.add(self.bath.build_environment(pulse=pulse, transition=tls.operators['lower']))

        emitter.initial_state = emitter.states['|g>|0>']

        def purcell_timescale(args: dict):
            kappa = args['cavity/decay']
            Gamma = args['emitter/decay'] + 2 * args['emitter/dephasing']
            Delta = args['emitter/resonance'] - args['cavity/resonance']
            R = 4 * args['coupling'] ** 2 * (kappa + Gamma) / ((kappa + Gamma) ** 2 + 4 * Delta ** 2)
            return {'purcell_rate': (R * kappa) / (R + kappa)}

        gate_par = parinit({'emitter/decay': 1, 'emitter/resonance': 0, 'emitter/dephasing': 0,
                            'cavity/decay': 1, 'cavity/resonance': 0, 'coupling': 0},
                           emitter.default_parameters | (parameters if parameters else {}))
        if gate is None:
            gate_pulse = deepcopy(pulse)
            gate_pulse.name = tls.name
            gate = rate_gate_from_pulse(gate_pulse,
                                        rate_function=purcell_timescale,
                                        parameter_name='purcell_rate',
                                        gate_parameters=gate_par)

        source = source_from_emitter(emitter=emitter,
                                     gate=gate,
                                     efficiency=efficiency,
                                     name=name,
                                     close_outputs=['direct'],
                                     mask_outputs=True)
        self.__dict__ = source.__dict__
        self.default_name = '_PhononAssisted'
