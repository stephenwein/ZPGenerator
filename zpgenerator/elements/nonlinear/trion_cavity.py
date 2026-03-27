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
                 purcell_factor: float = None,
                 regime: float = None,
                 timescale: float = None,
                 purcell_factor_h: float = None,
                 purcell_factor_v: float = None,
                 regime_h: float = None,
                 regime_v: float = None,
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

        def keyword_defaults(args: dict):
            purcell_total = args['purcell_factor_h'] + args['purcell_factor_v']
            decay = 1 / (purcell_total * args['timescale'])
            return {
                'coupling_h': decay * args['purcell_factor_h'] / (2 * args['regime_h']),
                'coupling_v': decay * args['purcell_factor_v'] / (2 * args['regime_v']),
                'cavity_h/decay': decay * args['purcell_factor_h'] / args['regime_h'] ** 2,
                'cavity_v/decay': decay * args['purcell_factor_v'] / args['regime_v'] ** 2,
                'trion/decay': decay,
            }

        make_parameter_function = any(value is not None for value in [
            purcell_factor, regime, timescale, purcell_factor_h, purcell_factor_v, regime_h, regime_v
        ])

        purcell_factor = 10 if purcell_factor is None else purcell_factor
        regime = 0.1 if regime is None else regime
        timescale = 1 if timescale is None else timescale
        keywords = {
            'purcell_factor_h': purcell_factor if purcell_factor_h is None else purcell_factor_h,
            'purcell_factor_v': purcell_factor if purcell_factor_v is None else purcell_factor_v,
            'regime_h': regime if regime_h is None else regime_h,
            'regime_v': regime if regime_v is None else regime_v,
            'timescale': timescale,
        }

        if make_parameter_function:
            self.create_overwrite_parameter_function(keyword_defaults, parameters=keywords)

        self.update_default_parameters(keyword_defaults(keywords) | (parameters if parameters else {}))
