from zpgenerator.time.evaluate.tensor import tensor_insert
from qutip import destroy, qeye, spre


def test_tensor_insert_operator_skips_trivial_dims():
    op = destroy(2)
    inserted = tensor_insert(op, 1, [0, 2, 1])
    assert inserted == op


def test_tensor_insert_superoperator_skips_trivial_dims():
    op = spre(destroy(2))
    inserted = tensor_insert(op, 1, [1, [2], 0])
    assert inserted == op


def test_tensor_insert_accepts_equivalent_scalar_and_list_dims():
    inserted = tensor_insert(qeye(1), 0, [0, 2])
    assert inserted == qeye([1, 2])
