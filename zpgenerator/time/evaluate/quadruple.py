from .operator import EvaluatedOperator, evop_mv, evop_umv
from typing import List
from qutip import qzero, qeye, Qobj as _Qobj, liouvillian
from copy import deepcopy
from .dims import qzero_or_empty, qeye_or_empty


class EvaluatedQuadruple:
    """
    A list of EvaluatedOperators representing an evaluated component
    """

    def  __init__(self,
                 hamiltonian: EvaluatedOperator = None,
                 environment: List[EvaluatedOperator] = None,
                 transitions: List[EvaluatedOperator] = None,
                 scatterer: EvaluatedOperator = None):

        self.hamiltonian = EvaluatedOperator() if hamiltonian is None else hamiltonian
        self.environment = [] if environment is None else environment
        self.transitions = [] if transitions is None else transitions

        if hamiltonian:
            self._subdims = hamiltonian.subdims
        else:
            if self.environment:
                self._subdims = self.environment[0].subdims
            elif self.transitions and isinstance(self.transitions[0], EvaluatedOperator):
                self._subdims = self.transitions[0].subdims
            else:
                self._subdims = [0]

        if not self.transitions and scatterer is not None:
            if not (isinstance(scatterer, EvaluatedOperator) and scatterer._is_empty_operator()):
                zero_transition = qzero_or_empty(self.subdims)
                self.transitions = [EvaluatedOperator(constant=zero_transition) for _ in range(scatterer.dim)]

        if scatterer is None:
            self.scatterer = EvaluatedOperator(constant=qeye(self.modes)) if self.modes > 0 else EvaluatedOperator()
        else:
            self.scatterer = scatterer

        if self.modes > 0:
            assert self.modes == self.scatterer.dim, \
                "Scattering matrices must have a dimension matching the number of modes"

    @property
    def modes(self):
        return len(self.transitions)

    @property
    def subdims(self):
        return self._subdims

    def tensor_insert(self, i: int, dims: list):
        """
        insert into a larger space
        :param i: position in subdims of new space
        :param dims: list of subdims/dims
        return EvaluatedQuadruple with new dimensions
        """
        return EvaluatedQuadruple(hamiltonian=self.hamiltonian.tensor_insert(i, dims),
                                  environment=[env.tensor_insert(i, dims) for env in self.environment],
                                  transitions=[tra.tensor_insert(i, dims) for tra in self.transitions],
                                  scatterer=self.scatterer)

    def __add__(self, other):
        if other == 0:
            return self
        else:
            return EvaluatedQuadruple(hamiltonian=self.hamiltonian + other.hamiltonian,
                                      environment=self.environment + other.environment,
                                      transitions=self.transitions + other.transitions,
                                      scatterer=self.scatterer.concatenate(other.scatterer))

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    # cascaded quantum coupling (https://www.tandfonline.com/doi/full/10.1080/23746149.2017.1343097)
    def __mul__(self, other):
        if other == 1:
            return self
        else:
            assert self.modes == other.modes, "Components must have the same number of modes."
            subdims = [self.subdims, other.subdims]

            scatterer = other.scatterer

            self_vector = [trn.tensor_insert(0, subdims) for trn in self.transitions]
            other_vector = [trn.tensor_insert(1, subdims) for trn in other.transitions]
            for vec in self_vector + other_vector:
                vec.reshape()

            hamiltonian = self.hamiltonian.tensor_insert(0, subdims) + other.hamiltonian.tensor_insert(1, subdims)

            environment = [env.tensor_insert(0, subdims) for env in self.environment] + \
                          [env.tensor_insert(1, subdims) for env in other.environment]

            transitions = evop_mv(scatterer, self_vector)
            transitions = [transitions[i] + other_vector[i] for i in range(0, len(transitions))]

            # Quantum cascaded interaction superoperator
            if (not any(_is_trivial_evop(v) for v in other_vector)) and \
                    (not any(_is_trivial_evop(v) for v in self_vector)):
                # for v in other_vector:
                #     print(v.constant)
                #     for pair in v.variable:
                #         print(pair.op)
                LRB = [v.dag().spost() - v.dag().spre() for v in other_vector]
                RLB = [v.spre() - v.spost() for v in other_vector]
                environment += [evop_umv(LRB, scatterer, [v.spre() for v in self_vector]) +
                                evop_umv(RLB, scatterer.dag(), [v.dag().spost() for v in self_vector])]

            # reshape() takes subdims with unit dimensions: [1, 2, 1] and changes it to [2]
            hamiltonian.reshape()
            for env in environment:
                env.reshape()
            environment = [env for env in environment if not _is_zero_evop(env)]
            for trn in transitions:
                trn.reshape()

            cascaded_scatterer = other.scatterer * self.scatterer
            if not isinstance(cascaded_scatterer, EvaluatedOperator):
                cascaded_scatterer = EvaluatedOperator(constant=cascaded_scatterer.constant,
                                                       variable=cascaded_scatterer.variable)
            if not isinstance(cascaded_scatterer.constant, _Qobj):
                cascaded_scatterer = EvaluatedOperator(constant=cascaded_scatterer.constant * qeye_or_empty(self.modes),
                                                       variable=cascaded_scatterer.variable)

            return EvaluatedQuadruple(hamiltonian=hamiltonian, environment=environment,
                                      transitions=transitions, scatterer=cascaded_scatterer)

    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            assert False, "Cannot cascade backwards"


    def evaluate(self, t: float, parameters: dict = None) -> _Qobj:
        h = self.hamiltonian.evaluate(t, parameters)
        c = [env.evaluate(t, parameters) for env in self.environment]
        h_is_empty = isinstance(h, _Qobj) and h.shape == (1, 1) and h.dims == [[1], [1]]
        h_as_operator = h.isoper and not h_is_empty
        if not (h_as_operator or c):
            return _Qobj()
        base_h = h if h_as_operator else qzero_or_empty(self.environment[0].subdims if c else [0])
        return liouvillian(H=base_h, c_ops=[op for op in c if op.isoper]) + sum([op for op in c if op.issuper])

    def pad(self, number: int):
        mode_increase = number - self.modes
        if mode_increase > 0:
            self.scatterer = self.scatterer.concatenate(EvaluatedOperator.id(mode_increase))
            self.transitions += [EvaluatedOperator(qzero_or_empty(self.subdims)) for i in range(0, mode_increase)]

    def permute(self, perm: List[int]):
        if sorted(perm) != perm:
            self.scatterer = self.scatterer.permute(perm)
            self.transitions = [self.transitions[i] for i in perm]

    def match(self, perm: List[int]):
        quad = deepcopy(self)
        quad.pad(len(perm))
        quad.permute(perm)
        return quad


def _is_zero_evop(evop: EvaluatedOperator) -> bool:
    return not evop.variable and isinstance(evop.constant, _Qobj) and evop.constant == 0 * evop.constant


def _is_trivial_evop(evop: EvaluatedOperator) -> bool:
    if evop.variable or not isinstance(evop.constant, _Qobj):
        return False
    if evop.constant.shape != (1, 1) or evop.constant.dims != [[1], [1]]:
        return False
    return evop.constant == 0 * evop.constant or evop.constant.full()[0, 0] != evop.constant.full()[0, 0]


# Keep Qobj available for wildcard imports used in tests.
Qobj = _Qobj
