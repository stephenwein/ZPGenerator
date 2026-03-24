from qutip import Qobj, lindblad_dissipator, liouvillian, mesolve, ptrace, spost, spre


BackendOperator = Qobj
BackendState = Qobj


def copy_state(state: BackendState) -> BackendState:
    return state.copy()


def density_matrix(state: BackendState) -> BackendState:
    return state if state.isoper else state * state.dag()


def state_dims(state: BackendState):
    return state.dims


def state_shape(state: BackendState):
    return state.shape


def state_is_operator(state: BackendState) -> bool:
    return state.isoper


def trace(state: BackendState):
    return state.tr()


def full(state: BackendState):
    return state.full()


def dagger(state: BackendState) -> BackendState:
    return state.dag()


def equals(left, right) -> bool:
    return left == right


def legacy_getitem(state: BackendState, item):
    matrix = full(state)
    if isinstance(item, tuple):
        return matrix[item]
    if isinstance(item, int):
        return matrix[item:item + 1, :]
    return matrix[item]


def is_backend_object(obj) -> bool:
    return isinstance(obj, Qobj)


def is_superoperator(op) -> bool:
    return hasattr(op, "issuper") and op.issuper


def apply_operator(state: BackendState, op: BackendOperator) -> BackendState:
    if op.isoper:
        rho = density_matrix(state)
        return op * rho * op.dag()
    return op(state)


def apply_generator(state: BackendState, op: BackendOperator, time: float, expmv_callable) -> BackendState:
    generator = liouvillian(op) if op.isoper else op
    return expmv_callable(time, generator, density_matrix(state))


def to_superoperator(op: BackendOperator) -> BackendOperator:
    return op if op.issuper else liouvillian(op)


def left_super(op: BackendOperator) -> BackendOperator:
    return spre(op)


def right_super(op: BackendOperator) -> BackendOperator:
    return spost(op)


def lindblad(op: BackendOperator) -> BackendOperator:
    return lindblad_dissipator(op)


def solve_master_equation(H, rho0, tlist, c_ops=None, e_ops=None, options=None):
    return mesolve(H=H, rho0=rho0, tlist=tlist, c_ops=[] if c_ops is None else c_ops, e_ops=e_ops, options=options)


def state_from_array(array_like, dims) -> BackendState:
    return Qobj(array_like, dims=dims)


def with_square_dims(state: BackendState, dims) -> BackendState:
    updated = copy_state(state)
    updated.dims = [dims, dims]
    return updated


def partial_trace(state: BackendState, select):
    return ptrace(state, select)

