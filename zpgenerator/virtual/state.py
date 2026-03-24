from ..time.evaluate import EvaluatedDiracOperator, expmv
from .backends import qutip_backend as qb
from typing import Union


class VState:
    """
    A virtual runtime state with explicit metadata layered around a backend state object.

    The backend payload remains available through ``qobj`` for the current QuTiP-based
    runtime, but this wrapper no longer subclasses backend types directly.
    """

    def __init__(self, state: qb.BackendState, time: float = 0, virtual_configuration: list = None):
        self._qobj = qb.copy_state(state)
        self.virtual_configuration = [] if virtual_configuration is None else list(virtual_configuration)
        self.time = time

    @property
    def qobj(self) -> qb.BackendState:
        """Compatibility alias for the wrapped backend state."""
        return self._qobj

    @property
    def dims(self):
        return qb.state_dims(self._qobj)

    @property
    def shape(self):
        return qb.state_shape(self._qobj)

    @property
    def isoper(self):
        return qb.state_is_operator(self._qobj)

    def _set_state(self, state: qb.BackendState):
        self._qobj = qb.copy_state(state)
        return self

    def update(self, state: qb.BackendState = None, time: float = None, virtual_configuration: list = None):
        if state is not None:
            self._set_state(state)
        if time is not None:
            self.time = time
        if virtual_configuration is not None:
            self.virtual_configuration = list(virtual_configuration)
        return self

    def branched(self, configuration, pos: int = -1) -> "VState":
        branched_state = VState(
            state=qb.copy_state(self._qobj),
            time=self.time,
            virtual_configuration=list(self.virtual_configuration),
        )
        branch_num = len(branched_state.virtual_configuration)
        if pos >= branch_num:
            branched_state.virtual_configuration = (
                branched_state.virtual_configuration + [0] * (pos - branch_num + 1)
            )
        elif branch_num == 0 and pos == -1:
            branched_state.virtual_configuration = [0]
        branched_state.virtual_configuration[pos] = configuration
        return branched_state

    def density_matrix(self) -> qb.BackendState:
        return qb.density_matrix(self._qobj)

    def tr(self):
        return qb.trace(self._qobj)

    def full(self):
        return qb.full(self._qobj)

    def dag(self):
        return qb.dagger(self._qobj)

    # Apply an instantaneous operator or superoperator
    def apply_operator(self, op: Union[qb.BackendOperator, EvaluatedDiracOperator]):
        if qb.is_backend_object(op):
            self._set_state(qb.apply_operator(self._qobj, op))
            return self
        elif isinstance(op, EvaluatedDiracOperator):
            if op.hamiltonian:
                self.apply_generator(op.hamiltonian)
            if op.channel:
                self.apply_operator(op.channel)

    # Apply an instantaneous HamiltonianBase or Liouvillian
    def apply_generator(self, op: qb.BackendOperator, time: float = 1):
        self._set_state(qb.apply_generator(self._qobj, op, time, expmv))
        return self

    def __eq__(self, other):
        if isinstance(other, VState):
            return qb.equals(self._qobj, other._qobj)
        return qb.equals(self._qobj, other)

    # QuTiP 5 changed indexing semantics. Keep legacy matrix-like behavior expected by this package.
    def __getitem__(self, item):
        return qb.legacy_getitem(self._qobj, item)

    def __getattr__(self, item):
        return getattr(self._qobj, item)
