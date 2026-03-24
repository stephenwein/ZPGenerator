from qutip import Qobj, liouvillian
from ..time.evaluate import EvaluatedDiracOperator, expmv
from typing import Union


class VState(Qobj):
    """
    A virtual state of a source conditioned on a history of virtual configurations,
    and that can evolve in time conditioned on a current configuration.

    :param state: a state of the source
    """

    def __init__(self, state: Qobj, time: float = 0, virtual_configuration: list = None):
        super().__init__(state)
        self.virtual_configuration = [] if virtual_configuration is None else virtual_configuration
        self.time = time

    def _set_state(self, state: Qobj):
        state = state.copy()
        self._data = state._data
        self._dims = state._dims
        self._isherm = state._isherm
        self._isunitary = state._isunitary
        return self

    def update(self, state: Qobj = None, time: float = None, virtual_configuration: list = None):
        if state is not None:
            self._set_state(state)
        if time is not None:
            self.time = time
        if virtual_configuration is not None:
            self.virtual_configuration = virtual_configuration
        return self

    # Apply an instantaneous operator or superoperator
    def apply_operator(self, op: Union[Qobj, EvaluatedDiracOperator]):
        if isinstance(op, Qobj):
            if op.isoper:
                rho = self if self.isoper else self * self.dag()
                self._set_state(op * rho * op.dag())
            else:
                self._set_state(op(self))
            return self
        elif isinstance(op, EvaluatedDiracOperator):
            if op.hamiltonian:
                self.apply_generator(op.hamiltonian)
            if op.channel:
                self.apply_operator(op.channel)

    # Apply an instantaneous HamiltonianBase or Liouvillian
    def apply_generator(self, op: Qobj, time: float = 1):
        # Could still be optimised...
        if op.isoper:
            op = liouvillian(op)

        rho = self if self.isoper else self * self.dag()
        self._set_state(expmv(time, op, rho))
        return self

    # Propagating the state forward in time given the current configuration
    def propagate(self, propagator, t: float, tlist: list = None):
        return propagator.propagate(self, t, tlist=tlist)

    # QuTiP 5 changed indexing semantics. Keep legacy matrix-like behavior expected by this package.
    def __getitem__(self, item):
        matrix = self.full()
        if isinstance(item, tuple):
            return matrix[item]
        if isinstance(item, int):
            return matrix[item:item + 1, :]
        return matrix[item]
