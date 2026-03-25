# classes and functions for handling conversion to QuTiP time-dependent operator formats
# perhaps a lot of this could be improved subclassing QobjEvo from qutip

from .tensor import tensor_insert, concat_diag, permutation_qobj
from qutip import Qobj, qeye, qzero, spre, spost, lindblad_dissipator, liouvillian
from typing import Union, List
from numpy import conj, isnan
from math import prod
from .cache import DefaultCache


class Func:
    """A class to evaluate a function using a set of arguments and define some operations between functions"""
    def __init__(self, func: Union[callable, float, int, complex], args: dict = None, cache=False):
        self.func = func
        self.args = {} if args is None else args
        self.cache = cache

    def __call__(self, t: float, args: dict = None):
        return self.cached_call(t, args) if self.cache else self.call(t, args)

    @DefaultCache(time_arg=True)
    def cached_call(self, t: float, args: dict = None):
        return self.call(t, args)

    def call(self, t: float, args: dict = None):
        return self.func(t, self.args | args if args else self.args) if callable(self.func) else self.func

    def __add__(self, other):
        if callable(other):
            return Func(lambda t, args: self(t, args) + other(t, args))
        else:
            return Func(lambda t, args: self(t, args) + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Func):
            return Func(lambda t, args: self(t, args) * other(t, args))
        elif not callable(other):
            return Func(lambda t, args: self(t, args) * other)
        elif isinstance(other, Qobj):
            return OpFuncPair(op=other, func=self)
        else:
            return other.__mul__(self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def conj(self):
        return Func(lambda t, args: conj(self(t, args)))

    def compose_with(self, function: callable, parameters: dict):
        return Func(lambda t, args: function(self(t, args), args), args=parameters)


class EvaluatedFunction:
    """An object that contains a constant value and/or a list of functions"""

    def __init__(self, constant=None, variable=None):
        self.variable = [] if variable is None else variable if isinstance(variable, list) else [variable]
        self.constant = 0 if constant is None else constant
        self._vartype = Func

    def __call__(self, t: float, parameters: dict = None):
        return self.constant + sum(v(t, parameters) for v in self.variable)

    def __add__(self, other):
        """
        :param other: another EvaluatedFunction or possibly a constant
        :return: another EvaluatedFunction
        """
        if isinstance(other, self.__class__):
            return self.__class__(constant=self.constant + other.constant,  # adds constants directly
                                  variable=self.variable + other.variable)  # concatenates lists of functions
        elif isinstance(other, self._vartype):
            return self.__class__(constant=self.constant,
                                  variable=self.variable + [other])
        elif not callable(other) or isinstance(other, Qobj):
            return self.__class__(constant=self.constant + other,  # adds other to constant part
                                  variable=self.variable)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1. * other)

    def __mul__(self, other):
        if isinstance(other, EvaluatedOperator) or isinstance(other, EvaluatedFunction):
            const = self.constant * other.constant
            var1 = [self.constant * v for v in other.variable]
            var2 = [v * other.constant for v in self.variable]
            var3 = [v1 * v2 for v1 in self.variable for v2 in other.variable]
            if isinstance(const, Qobj):
                evop = EvaluatedOperator(constant=const, variable=var1 + var2 + var3)
                evop._clean()
                return evop
            else:
                evfu = EvaluatedFunction(constant=const, variable=var1 + var2 + var3)
                evfu._clean()
                return evfu
        elif isinstance(other, Func):
            if isinstance(self.constant, Qobj):
                return EvaluatedOperator(constant=0 * self.constant,
                                         variable=[other * v for v in self.variable] + [other * self.constant])
            else:
                return EvaluatedFunction(constant=0,
                                         variable=[other * v for v in self.variable] + [other * self.constant])
        elif not callable(other) or isinstance(other, Qobj):
            return self.__class__(constant=self.constant * other,
                                  variable=[v * other for v in self.variable])
        else:
            return other.__mul__(self)

    def __rmul__(self, other):
        if isinstance(other, EvaluatedOperator) or isinstance(other, EvaluatedFunction):
            const = other.constant * self.constant
            var1 = [v * self.constant for v in other.variable]
            var2 = [other.constant * v for v in self.variable]
            var3 = [v2 * v1 for v1 in self.variable for v2 in other.variable]
            if isinstance(const, Qobj):
                evop = EvaluatedOperator(constant=const, variable=var1 + var2 + var3)
                evop._clean()
                return evop
            else:
                evfu = EvaluatedFunction(constant=const, variable=var1 + var2 + var3)
                evfu._clean()
                return evfu
        elif not callable(other) or isinstance(other, Qobj):
            if other == 0 * other:
                return self.__class__()
            else:
                return self.__class__(constant=other * self.constant,
                                      variable=[other * v for v in self.variable])

    def _clean(self):
        pass


class OpFuncPair:
    """An object that contains a constant Qobj and an optional modulating function"""

    def __init__(self, op: Qobj, func: Union[callable, Func], parameters: dict = None):
        self.op = op
        self.func = func if isinstance(func, Func) else Func(func, parameters)

    def __call__(self, t: float, parameters: dict = None):
        return self.op * self.func(t, parameters)

    def __mul__(self, other):
        if isinstance(other, OpFuncPair):
            return OpFuncPair(op=self.op * other.op, func=self.func * other.func)
        elif isinstance(other, Func) or isinstance(other, EvaluatedFunction):
            return OpFuncPair(op=self.op, func=self.func * other)
        else:
            return OpFuncPair(op=self.op * other, func=self.func)

    def __rmul__(self, other):
        if isinstance(other, OpFuncPair):
            return OpFuncPair(op=other.op * self.op, func=self.func * other.func)
        elif isinstance(other, Func):
            return OpFuncPair(op=self.op, func=self.func * other.func)
        else:
            return OpFuncPair(op=other * self.op, func=self.func)

    def tensor_insert(self, i: int, dims: list):
        return OpFuncPair(op=tensor_insert(self.op, i, dims), func=self.func)

    def pad_left(self, subdims: Union[int, List[int]]):
        return OpFuncPair(op=concat_diag(qzero(subdims), self.op), func=self.func)

    def pad_right(self, subdims: Union[int, List[int]]):
        return OpFuncPair(op=concat_diag(self.op, qzero(subdims)), func=self.func)

    def spre(self):
        return OpFuncPair(op=spre(self.op), func=self.func)

    def spost(self):
        return OpFuncPair(op=spost(self.op), func=self.func)

    def lind(self):
        return OpFuncPair(op=lindblad_dissipator(self.op), func=self.func * self.func.conj())

    def liou(self):
        return OpFuncPair(op=liouvillian(self.op), func=self.func)

    def dag(self):
        return OpFuncPair(op=self.op.dag(), func=self.func.conj())

    @property
    def is_super(self):
        return self.op.issuper

    @property
    def subdims(self):
        mat = self.op
        return mat.dims[0][0] if mat.issuper else mat.dims[0]

    def list_form(self):
        return [self.op, self.func]

    def permute_left(self, order: list):
        return OpFuncPair(op=permutation_qobj(order) * self.op, func=self.func)

    def permute_right(self, order: list):
        return OpFuncPair(op=self.op * (permutation_qobj(order).dag()), func=self.func)

    def permute(self, order: list):
        mat = permutation_qobj(order)
        return OpFuncPair(op=mat * (self.op * mat.dag()), func=self.func)

    def evaluate(self, t: float, parameters: dict = None):
        return self.op * self.func(t, parameters)

    def element(self, i: int, j: int) -> Func:
        return self.op[i, j] * self.func


class EvaluatedOperator(EvaluatedFunction):
    """An object that contains a constant operator and/or a list of OpFuncPair objects"""

    def __init__(self, constant: Qobj = None, variable: Union[OpFuncPair, List[OpFuncPair]] = None):
        super().__init__(constant, variable)
        if self.constant == 0:
            self.constant = 0 * self.variable[0].op if self.variable else Qobj()
        self._vartype = OpFuncPair

    @staticmethod
    def _is_scalar_empty(op):
        if not isinstance(op, Qobj) or op.shape != (1, 1) or op.dims != [[1], [1]]:
            return False
        value = op.full()[0, 0]
        return isnan(value.real) or isnan(value.imag)

    @staticmethod
    def _is_numeric_zero(op):
        return not isinstance(op, Qobj) and op == 0

    def _is_empty_operator(self):
        return not self.variable and (self._is_scalar_empty(self.constant) or self._is_numeric_zero(self.constant))

    def __add__(self, other):
        if isinstance(other, EvaluatedOperator):
            if self._is_empty_operator():
                return other
            if other._is_empty_operator():
                return self
            try:
                return super().__add__(other)
            except ValueError:
                if isinstance(self.constant, Qobj) and isinstance(other.constant, Qobj):
                    left = self.constant.copy()
                    right = other.constant.copy()
                    left.dims = _clean_dim_tree(left.dims)
                    right.dims = _clean_dim_tree(right.dims)
                    if left.dims == right.dims:
                        return EvaluatedOperator(constant=left + right,
                                                 variable=self.variable + other.variable)
                raise
        return super().__add__(other)

    def __rmul__(self, other):
        if not callable(other) and not isinstance(other, (EvaluatedOperator, EvaluatedFunction, Func, Qobj)):
            if other == 0 * other:
                return EvaluatedOperator(constant=0 * self.constant,
                                         variable=[other * v for v in self.variable])
        return super().__rmul__(other)

    def _clean(self):
        zero = 0 * self.constant
        self.variable = [v for v in self.variable if v.op != zero]

    # Not sure if these should change self in place or copy... below might be slow but perhaps more predictable
    def tensor_insert(self, i: int, dims: list):
        return EvaluatedOperator(
            constant=tensor_insert(self.constant, i, dims)
            if isinstance(self.constant, Qobj) and not self._is_scalar_empty(self.constant) and
            (self.constant.isoper or self.constant.issuper)
            else self.constant,
            variable=[v.tensor_insert(i, dims) for v in self.variable])

    def concatenate(self, other):
        if self._is_empty_operator():
            return other
        if other._is_empty_operator():
            return self
        if isinstance(self.constant, Qobj):
            left_constant = self.constant
        elif isinstance(other.constant, Qobj):
            left_constant = 0 * other.constant
        else:
            left_constant = Qobj()

        if isinstance(other.constant, Qobj):
            right_constant = other.constant
        elif isinstance(self.constant, Qobj):
            right_constant = 0 * self.constant
        else:
            right_constant = Qobj()

        return EvaluatedOperator(constant=concat_diag(left_constant, right_constant),
                                 variable=[v.pad_right(other.subdims) for v in self.variable] +
                                          [v.pad_left(self.subdims) for v in other.variable])

    def spre(self):
        constant = spre(self.constant) if isinstance(self.constant, Qobj) else self.constant
        return EvaluatedOperator(constant=constant, variable=[v.spre() for v in self.variable])

    def spost(self):
        constant = spost(self.constant) if isinstance(self.constant, Qobj) else self.constant
        return EvaluatedOperator(constant=constant, variable=[v.spost() for v in self.variable])

    def jump(self):
        return self.spre() * self.dag().spost()

    def num(self):
        return self.dag() * self

    def lind(self):
        constant = lindblad_dissipator(self.constant) if isinstance(self.constant, Qobj) else self.constant
        return EvaluatedOperator(constant=constant,
                                 variable=[v.lind() for v in self.variable])

    def liou(self):
        constant = liouvillian(self.constant) if isinstance(self.constant, Qobj) else self.constant
        return EvaluatedOperator(constant=constant,
                                 variable=[v.liou() for v in self.variable])

    def dag(self):
        constant = self.constant.dag() if isinstance(self.constant, Qobj) else self.constant
        return EvaluatedOperator(constant=constant, variable=[v.dag() for v in self.variable])

    @property
    def is_super(self):
        return self.constant.issuper if isinstance(self.constant, Qobj) else False

    def list_form(self):
        variable = [v.list_form() for v in self.variable]
        return [self.constant, *variable]

    @property
    def subdims(self):
        if isinstance(self.constant, Qobj):
            mat = self.constant
            return mat.dims[0][0] if mat.issuper else mat.dims[0]
        if self.variable:
            return self.variable[0].subdims
        return [0]

    @property
    def dim(self):
        return prod(self.subdims)

    def permute(self, perm: List[int]):
        mat = permutation_qobj(perm)
        return mat * (self * mat.dag())

    def evaluate(self, t: float, parameters: dict = None):
        return self.constant + sum(v.op * v.func(t, parameters) for v in self.variable)

    def element(self, i: int, j: int):
        constant = self.constant[i, j] if isinstance(self.constant, Qobj) else 0
        return EvaluatedFunction(constant=constant, variable=[v.element(i, j) for v in self.variable])

    def reshape(self, subdims: List[int] = None):
        subdims = _clean_dims(self.subdims) if subdims is None else subdims
        if prod(subdims) != self.dim:
            raise ValueError("Subdimension product must match the total dimension.")
        if not isinstance(self.constant, Qobj):
            self.constant = qzero(subdims)
        self.constant.dims = [[subdims, subdims], [subdims, subdims]] if self.constant.issuper else [subdims, subdims]
        for v in self.variable:
            v.op.dims = [[subdims, subdims], [subdims, subdims]] if v.op.issuper else [subdims, subdims]

    @classmethod
    def id(cls, dim: int):
        return cls(constant=qeye(dim))


def evop_mv(m: EvaluatedOperator, v: List[EvaluatedOperator]):
    if m.dim != len(v):
        raise ValueError("Matrix dimensions must match vector length")
    return [sum(m.element(i, j) * n for j, n in enumerate(v)) for i in range(0, m.dim)]


def evop_umv(s: List[EvaluatedOperator], m: EvaluatedOperator, v: List[EvaluatedOperator]):
    if m.dim != len(v):
        raise ValueError("Matrix dimensions must match vector length")
    if m.dim != len(s):
        raise ValueError("Matrix dimensions must match vector length")
    return sum(o * (m.element(i, j) * n) for j, n in enumerate(v) for i, o in enumerate(s))


def _clean_dims(subdims: List[int]):
    new_dims = [i for i in subdims if i != 0 and i != 1]
    return new_dims if new_dims else [1]


def _clean_dim_tree(dims):
    if isinstance(dims, list):
        if dims and all(isinstance(v, int) for v in dims):
            new_dims = [v for v in dims if v != 1]
            return new_dims if new_dims else [1]
        return [_clean_dim_tree(v) for v in dims]
    return dims
