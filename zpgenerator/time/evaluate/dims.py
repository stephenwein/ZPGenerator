from qutip import Qobj, qeye, qzero


def is_trivial_dim(dim) -> bool:
    return dim in (0, 1, [0], [1])


def canonical_dim_list(dim):
    if dim in (0, [0]):
        return [1]
    return dim if isinstance(dim, list) else [dim]


def qzero_or_empty(dims):
    if dims in (0, [0]):
        return Qobj()
    return qzero(dims)


def qeye_or_empty(dim):
    if dim in (0, [0]):
        return Qobj()
    return qeye(dim)
