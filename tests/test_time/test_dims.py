from zpgenerator.time.evaluate.dims import (
    canonical_dim_list,
    is_trivial_dim,
    qeye_or_empty,
    qzero_or_empty,
)
from qutip import qeye, qzero


def _is_empty_qobj(op):
    if op.shape != (1, 1) or op.dims != [[1], [1]]:
        return False
    value = op.full()[0, 0]
    return value != value


def test_is_trivial_dim_variants():
    assert is_trivial_dim(0)
    assert is_trivial_dim(1)
    assert is_trivial_dim([0])
    assert is_trivial_dim([1])
    assert not is_trivial_dim(2)
    assert not is_trivial_dim([2])


def test_canonical_dim_list_normalization():
    assert canonical_dim_list(0) == [1]
    assert canonical_dim_list([0]) == [1]
    assert canonical_dim_list(2) == [2]
    assert canonical_dim_list([2, 3]) == [2, 3]


def test_qzero_or_empty():
    assert _is_empty_qobj(qzero_or_empty(0))
    assert _is_empty_qobj(qzero_or_empty([0]))
    assert qzero_or_empty(2) == qzero(2)


def test_qeye_or_empty():
    assert _is_empty_qobj(qeye_or_empty(0))
    assert _is_empty_qobj(qeye_or_empty([0]))
    assert qeye_or_empty(2) == qeye(2)
