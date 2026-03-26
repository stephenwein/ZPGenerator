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
        cavity_h = CavityEmitter(truncation=truncation, name='cavity_h')
        cavity_v = CavityEmitter(truncation=truncation, name='cavity_v')
        trion = TrionEmitter(charge=charge, name='trion')

        lower_r = trion.operators['lower_R']
        lower_l = trion.operators['lower_L']
        dipole_h = (lower_r + lower_l) / sqrt(2)
        dipole_v = 1.j * (lower_r - lower_l) / sqrt(2)

        id_h = qeye(cavity_h.dim)
        id_v = qeye(cavity_v.dim)

        coupling = CouplingBase([
            CouplingTerm(
                [cavity_h.operators['annihilation'].dag(), id_v, lambda args: args.get('coupling_h', 1) * dipole_h],
                parameters={'coupling_h': 1},
            ),
            CouplingTerm(
                [cavity_h.operators['annihilation'], id_v, lambda args: args.get('coupling_h', 1) * dipole_h.dag()],
                parameters={'coupling_h': 1},
            ),
            CouplingTerm(
                [id_h, cavity_v.operators['annihilation'].dag(), lambda args: args.get('coupling_v', 1) * dipole_v],
                parameters={'coupling_v': 1},
            ),
            CouplingTerm(
                [id_h, cavity_v.operators['annihilation'], lambda args: args.get('coupling_v', 1) * dipole_v.dag()],
                parameters={'coupling_v': 1},
            ),
        ])

        super().__init__(subsystems=[cavity_h, cavity_v, trion], coupling=coupling, parameters=parameters, name=name)
