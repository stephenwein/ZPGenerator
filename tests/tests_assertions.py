from qutip import Qobj


def is_empty_qobj(op) -> bool:
    if not isinstance(op, Qobj):
        return False
    if op.shape != (1, 1) or op.dims != [[1], [1]]:
        return False
    value = op.full()[0, 0]
    return value != value


def assert_empty_qobj(op):
    assert is_empty_qobj(op), (
        f"Expected empty scalar Qobj semantics, got shape={getattr(op, 'shape', None)} "
        f"dims={getattr(op, 'dims', None)}"
    )
