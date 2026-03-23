from .state import VState
from ..time import EvaluatedOperator, Func
from .solver_options import mesolve_options
from abc import ABC, abstractmethod
from qutip import Qobj, mesolve, spre, spost, liouvillian, lindblad_dissipator
from typing import Union


class AVirtualPropagator(ABC):
    """
    An object that propagates a virtual state to time t conditioned on a virtual configuration.
    """

    # Computes the jump operator given the virtual configuration
    @abstractmethod
    def jump(self, virtual_configuration):
        pass

    # Propagates the VState object forward until time t
    @abstractmethod
    def propagate(self, virtual_state: VState, t: float, tlist: list = None):
        pass


class VPropHTD(AVirtualPropagator):
    """
    A propagator that uses qutip.mesolve with time-dependent Hermitian and time-independent non-Hermitian evolution.
    """

    def __init__(self,
                 hamiltonian: list,
                 collapse_operators: list[Qobj] = None,
                 jumps: list[Qobj] = None,
                 expect_operators: Union[Qobj, callable] = None,
                 options: dict = None
                 ):
        """

        :param hamiltonian: a list of the form [Qobj, [Qobj, function], ...] describing the time-dependent HamiltonianBase.
        :param collapse_operators: a list of Qobj describing all the collapse operators.
        :param jumps: a list of Qobj superoperators describing the jump statistics (without scaling by vconfig)
        :param expect_operators: a list of Qobj to evaluate expecation values for
        :param options: a dictionary of qutip.mesolve options.
        """
        self.hamiltonian = self._normalise_hamiltonian(hamiltonian)

        self.collapse_operators = self._normalise_collapse_operators(collapse_operators)

        self.jumps = [] if jumps is None else jumps
        self.expect_operators = expect_operators
        self.options = options

    @staticmethod
    def _normalise_hamiltonian(hamiltonian: list):
        normalised = []
        for term in hamiltonian:
            if isinstance(term, list) and len(term) == 2 and isinstance(term[1], Func):
                func = term[1]
                normalised.append([term[0], lambda t, args=None, f=func: f(t, args)])
            elif isinstance(term, list) and len(term) == 2 and hasattr(term[1], "__call__"):
                normalised.append(term)
            else:
                normalised.append(term)
        return normalised

    @staticmethod
    def _normalise_collapse_operators(collapse_operators: list[Qobj]):
        normalised = []
        if collapse_operators is None:
            return normalised

        for term in collapse_operators:
            if isinstance(term, list) and len(term) == 1:
                term = term[0]
            elif isinstance(term, list) and len(term) == 2 and isinstance(term[1], Func):
                func = term[1]
                term = [term[0], lambda t, args=None, f=func: f(t, args)]

            base = _collapse_term_base(term)
            if _is_zero_qobj(base):
                continue
            normalised.append(term)
        return normalised

    def jump(self, vconfig):
        default = 0 * spre(self.hamiltonian[0])
        return sum([-vconfig[i] * list_get(self.jumps, i, default) for i in range(0, len(vconfig))], default)

    def propagate(self, virtual_state: VState, t: float, tlist: list = None):
        jump = self.jump(virtual_state.virtual_configuration)
        use_superoperator_solver = (
            jump != 0 * jump
            or any(_collapse_term_is_super(op) for op in self.collapse_operators)
        )

        if use_superoperator_solver:
            generator = _compile_superoperator_generator(
                hamiltonian=self.hamiltonian,
                collapse_operators=self.collapse_operators,
                jump=jump if jump != 0 * jump else None,
            )
            rho0 = virtual_state if virtual_state.isoper else virtual_state * virtual_state.dag()
            result = mesolve(H=generator,
                             rho0=rho0,
                             tlist=[virtual_state.time, t] if tlist is None else tlist,
                             c_ops=[],
                             e_ops=self.expect_operators,
                             options=mesolve_options(self.options, force_unnormalized=True))
        else:
            result = mesolve(H=self.hamiltonian,
                             rho0=virtual_state,
                             tlist=[virtual_state.time, t] if tlist is None else tlist,
                             c_ops=self.collapse_operators,
                             e_ops=self.expect_operators,
                             options=mesolve_options(self.options, force_unnormalized=False))
        virtual_state.update(state=result.states[-1], time=t)
        return result


class VPropNHTD(AVirtualPropagator):
    """
    A propagator that uses qutip.mesolve with a time-dependent non-Hermitian evolution.
    """

    def __init__(self,
                 generator: EvaluatedOperator,
                 jumps: list[EvaluatedOperator] = None,
                 expect_operators: Union[Qobj, callable] = None,
                 options: dict = None
                 ):
        """

        :param generator: an EvaluatedOperator object describing the time-dependent generator.
        :param jumps: a list of EvaluatedOperator objects describing possibly time-dependent jumps.
        :param expect_operators: a list of Qobj to evaluate expecation values for
        :param options: a dictionary of qutip.mesolve options.
        """
        self.generator = generator
        self.jumps = [] if jumps is None else jumps
        self.expect_operators = expect_operators
        self.options = options

    def jump(self, vconfig) -> EvaluatedOperator:
        default = 0 * self.jumps[0].constant if self.jumps else 0 * self.generator.constant
        return sum((-vconfig[i] * list_get(self.jumps, i, default) for i in range(0, len(vconfig))), default)

    def propagate(self, virtual_state: VState, t: float, tlist: list = None):
        jump = self.jump(virtual_state.virtual_configuration) if virtual_state.virtual_configuration else None
        gen = self.generator + jump if jump is not None else self.generator
        hamiltonian = []
        for term in gen.list_form():
            if isinstance(term, list) and len(term) == 2 and isinstance(term[1], Func):
                func = term[1]
                hamiltonian.append([term[0], lambda time, args=None, f=func: f(time, args)])
            else:
                hamiltonian.append(term)

        rho0 = virtual_state if virtual_state.isoper else virtual_state * virtual_state.dag()
        result = mesolve(H=hamiltonian,
                         rho0=rho0,
                         tlist=[virtual_state.time, t] if tlist is None else tlist,
                         e_ops=self.expect_operators,
                         options=mesolve_options(
                             self.options,
                             force_unnormalized=jump is not None and not _is_zero_evalop(jump),
                         ))
        virtual_state.update(state=result.states[-1], time=t)
        return result


class VPropTI(AVirtualPropagator):
    """
    A propagator that uses matrix exponentiation to propagate a state using a time-independent generator
    """

    def __init__(self,
                 generator: Qobj,
                 jumps: list[Qobj] = None,
                 ):
        assert generator.isoper or generator.issuper, "gen must be an operator or superoperator"
        self.generator = liouvillian(generator) if generator.isoper else generator
        self.jumps = [] if jumps is None else jumps

    def jump(self, vconfig):
        default = 0 * self.generator
        return sum([-vconfig[i] * list_get(self.jumps, i, default) for i in range(0, len(vconfig))], default)

    def propagate(self, virtual_state: VState, t: float, tlist: list = None):
        virtual_state.apply_generator(op=(self.generator + self.jump(virtual_state.virtual_configuration)),
                                      time=t - virtual_state.time)
        virtual_state.time = t


def list_get(lst, idx, default):
    try:
        return lst[idx]
    except IndexError:
        return default


def _is_zero_evalop(evop: EvaluatedOperator) -> bool:
    return not evop.variable and evop.constant == 0 * evop.constant


def _compile_superoperator_generator(hamiltonian: list, collapse_operators: list[Qobj], jump: Qobj = None):
    compiled = []
    for term in hamiltonian:
        if isinstance(term, list) and len(term) == 2 and hasattr(term[1], "__call__"):
            op = term[0]
            func = term[1]
            if op.issuper:
                compiled.append([op, func])
            else:
                compiled.append([-1.j * spre(op), func])
                compiled.append([1.j * spost(op.dag()), _conjugate_coefficient(func)])
        else:
            compiled.append(_as_superoperator(term))

    for term in collapse_operators:
        if isinstance(term, list) and len(term) == 2 and hasattr(term[1], "__call__"):
            op = term[0]
            func = term[1]
            if hasattr(op, "isoper") and op.isoper:
                compiled.append([lindblad_dissipator(op), lambda t, args=None, f=func: abs(f(t, args)) ** 2])
            elif hasattr(op, "issuper") and op.issuper:
                compiled.append([op, func])
        else:
            op = _collapse_term_base(term)
            if hasattr(op, "isoper") and op.isoper:
                compiled.append(lindblad_dissipator(op))
            elif hasattr(op, "issuper") and op.issuper:
                compiled.append(op)

    if jump is not None:
        compiled.append(jump)

    return _collapse_qobj_list(compiled)


def _as_superoperator(op: Qobj) -> Qobj:
    if op.issuper:
        return op
    return liouvillian(op)


def _collapse_qobj_list(terms):
    static_terms = [term for term in terms if not (isinstance(term, list) and len(term) == 2)]
    time_terms = [term for term in terms if isinstance(term, list) and len(term) == 2]
    if not static_terms:
        if time_terms:
            return time_terms
        assert False, "Cannot build a generator from an empty term list."
    static = sum(static_terms[1:], static_terms[0])
    return [static, *time_terms]


def _collapse_term_base(term):
    if isinstance(term, list) and term:
        return term[0]
    return term


def _collapse_term_is_super(term) -> bool:
    base = _collapse_term_base(term)
    return hasattr(base, "issuper") and base.issuper


def _is_zero_qobj(op) -> bool:
    if not isinstance(op, Qobj):
        return False
    return op == 0 * op


def _conjugate_coefficient(func):
    return lambda t, args=None, f=func: complex(f(t, args)).conjugate()
