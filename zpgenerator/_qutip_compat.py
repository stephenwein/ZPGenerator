import qutip as _qt
from qutip import Qobj
from numpy import isnan


def qobj_compat(*args, **kwargs):
    # Backward-compatibility aliases used across this codebase and tests.
    if "inpt" in kwargs and "arg" not in kwargs and not args:
        kwargs["arg"] = kwargs.pop("inpt")
    if "type" in kwargs:
        qtype = kwargs.pop("type")
        if qtype == "super" and "superrep" not in kwargs:
            kwargs["superrep"] = "super"
    return Qobj(*args, **kwargs)


def patch_qutip_qobj_kwargs():
    # Backward-compatible runtime behavior used across this codebase and tests.
    if getattr(Qobj, "_zpg_qobj_legacy_patched", False):
        return

    original_eq = Qobj.__eq__
    original_qeye = _qt.qeye
    original_qzero = _qt.qzero
    original_liouvillian = _qt.liouvillian

    def patched_eq(self, other):
        if isinstance(other, Qobj):
            if self.shape == (1, 1) and other.shape == (1, 1) and self.dims == [[1], [1]] and other.dims == [[1], [1]]:
                left = self.full()[0, 0]
                right = other.full()[0, 0]
                if (isnan(left.real) or isnan(left.imag)) and (isnan(right.real) or isnan(right.imag)):
                    return True
        return original_eq(self, other)

    Qobj.__eq__ = patched_eq

    def patched_qeye(dimensions):
        if dimensions == 0 or dimensions == [0]:
            return Qobj()
        return original_qeye(dimensions)

    def patched_qzero(dimensions):
        if dimensions == 0 or dimensions == [0]:
            return Qobj()
        return original_qzero(dimensions)

    _qt.qeye = patched_qeye
    _qt.qzero = patched_qzero

    def patched_liouvillian(H=None, c_ops=None, data_only=False, chi=None):
        c_ops = [] if c_ops is None else list(c_ops)
        op_c_ops = [op for op in c_ops if hasattr(op, "isoper") and op.isoper]
        super_c_ops = [op for op in c_ops if hasattr(op, "issuper") and op.issuper]
        base = original_liouvillian(H=H, c_ops=op_c_ops, data_only=data_only, chi=chi)
        if super_c_ops:
            base = base + sum(super_c_ops)
        return base

    _qt.liouvillian = patched_liouvillian

    Qobj._zpg_qobj_legacy_patched = True
