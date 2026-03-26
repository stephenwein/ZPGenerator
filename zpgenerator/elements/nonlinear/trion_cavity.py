from ...system import MultiBodyEmitter, CouplingBase, CouplingTerm
from .trion import TrionEmitter
from .cavity import CavityEmitter
from qutip import qeye
from numpy import sqrt


class TrionCavityEmitter(MultiBodyEmitter):
    """
    A trion emitter coupled to two orthogonally polarised cavity modes.
    """

    def __init__(self,
                 charge: str = 'negative',
                 truncation: int = 2,
                 parameters: dict = None,
                 name: str = None):
        trion = TrionEmitter(charge=charge, name='trion')
        cavity_h = CavityEmitter(truncation=truncation, name='cavity_h')
        cavity_v = CavityEmitter(truncation=truncation, name='cavity_v')

        lower_r = trion.operators['lower_R']
        lower_l = trion.operators['lower_L']
        dipole_h = (lower_r + lower_l) / sqrt(2)
        dipole_v = 1.j * (lower_r - lower_l) / sqrt(2)

        id_h = qeye(cavity_h.dim)
        id_v = qeye(cavity_v.dim)

        coupling = CouplingBase([
            CouplingTerm(
                [lambda args: args['coupling_h'] * dipole_h, cavity_h.operators['annihilation'].dag(), id_v],
                parameters={'coupling_h': 1},
            ),
            CouplingTerm(
                [lambda args: args['coupling_h'] * dipole_h.dag(), cavity_h.operators['annihilation'], id_v],
                parameters={'coupling_h': 1},
            ),
            CouplingTerm(
                [lambda args: args['coupling_v'] * dipole_v, id_h, cavity_v.operators['annihilation'].dag()],
                parameters={'coupling_v': 1},
            ),
            CouplingTerm(
                [lambda args: args['coupling_v'] * dipole_v.dag(), id_h, cavity_v.operators['annihilation']],
                parameters={'coupling_v': 1},
            ),
        ])

        super().__init__(subsystems=[trion, cavity_h, cavity_v], coupling=coupling, parameters=parameters, name=name)

