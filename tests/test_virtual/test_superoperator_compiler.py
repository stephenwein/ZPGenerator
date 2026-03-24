from cmath import exp
from time import perf_counter
from zpgenerator.virtual.propagator import _compile_superoperator_generator
from zpgenerator.virtual.propagator import VPropHTD
from zpgenerator.virtual.state import VState
from qutip import (
    create,
    destroy,
    fock,
    lindblad_dissipator,
    liouvillian,
    mesolve,
    num,
    spre,
    spost,
    sprepost,
)


def _dense_close(a, b, tol=1e-10):
    return (a - b).norm() < tol


def test_compiler_time_dependent_complex_hamiltonian_splits_conjugate_terms():
    op = create(2) + destroy(2)
    coeff = lambda t, args: exp(1j * args["w"] * t)
    compiled = _compile_superoperator_generator(hamiltonian=[[op, coeff]], collapse_operators=[])

    assert isinstance(compiled, list)
    assert len(compiled) == 2
    assert _dense_close(compiled[0][0], -1j * spre(op))
    assert _dense_close(compiled[1][0], 1j * spost(op.dag()))

    t = 0.37
    args = {"w": 2.1}
    assert compiled[0][1](t, args) == coeff(t, args)
    assert compiled[1][1](t, args) == coeff(t, args).conjugate()


def test_compiler_time_dependent_collapse_operator_uses_abs_squared_scaling():
    op = destroy(2)
    coeff = lambda t, args: 1 + 1j * t
    compiled = _compile_superoperator_generator(hamiltonian=[], collapse_operators=[[op, coeff]])

    assert isinstance(compiled, list)
    assert len(compiled) == 1
    assert _dense_close(compiled[0][0], lindblad_dissipator(op))

    t = 0.4
    args = {}
    assert compiled[0][1](t, args) == abs(coeff(t, args)) ** 2


def test_compiler_mixed_operator_and_superoperator_collapse_terms():
    op_c = destroy(2)
    super_c = sprepost(destroy(2), create(2))
    compiled = _compile_superoperator_generator(hamiltonian=[], collapse_operators=[op_c, super_c])

    expected = lindblad_dissipator(op_c) + super_c
    assert isinstance(compiled, list)
    assert len(compiled) == 1
    assert _dense_close(compiled[0], expected)


def test_compiler_jump_inclusion_and_exclusion():
    op_c = destroy(2)
    jump = 0.3 * sprepost(destroy(2), create(2))

    compiled_no_jump = _compile_superoperator_generator(hamiltonian=[], collapse_operators=[op_c], jump=None)
    compiled_jump = _compile_superoperator_generator(hamiltonian=[], collapse_operators=[op_c], jump=jump)

    expected_no_jump = lindblad_dissipator(op_c)
    expected_jump = expected_no_jump + jump
    assert isinstance(compiled_no_jump, list)
    assert isinstance(compiled_jump, list)
    assert len(compiled_no_jump) == 1
    assert len(compiled_jump) == 1
    assert _dense_close(compiled_no_jump[0], expected_no_jump)
    assert _dense_close(compiled_jump[0], expected_jump)


def test_compiled_generator_matches_classic_mesolve_for_resonant_and_detuned_cases():
    a = destroy(2)
    sx = create(2) + a

    c_ops = [a]
    rho0 = fock(2, 1) * fock(2, 1).dag()
    tlist = [0, 0.5]
    options = {"normalize_output": False, "atol": 1e-10, "rtol": 1e-10, "nsteps": 200000}

    for delta in (0.0, 1.3):
        hamiltonian = [
            delta * num(2),
            [sx / 2, lambda t, args=None: 1.2 * exp(-0.3j * t).real],
        ]
        classic = mesolve(H=hamiltonian, rho0=rho0, tlist=tlist, c_ops=c_ops, options=options)

        compiled = _compile_superoperator_generator(hamiltonian=hamiltonian, collapse_operators=c_ops)
        compiled_result = mesolve(H=compiled, rho0=rho0, tlist=tlist, c_ops=[], options=options)

        assert _dense_close(classic.states[-1], compiled_result.states[-1], tol=5e-9)


def test_vprophtd_keeps_unnormalized_trace_with_superoperator_jumps():
    a = destroy(2)
    ad = create(2)
    sx = a + ad
    rho0 = fock(2, 1) * fock(2, 1).dag()

    hamiltonian = [0.2 * num(2), [0.3 * sx, lambda t, args=None: exp(0.4j * t)]]
    collapse_ops = [a, 0.15 * sprepost(a, ad)]
    raw_jump = 0.25 * sprepost(a, ad)
    vconfig = [0.6]

    vprop = VPropHTD(
        hamiltonian=hamiltonian,
        collapse_operators=collapse_ops,
        jumps=[raw_jump],
        options={"normalize_output": True, "atol": 1e-10, "rtol": 1e-10, "nsteps": 200000},
    )
    vstate = VState(state=rho0, time=0, virtual_configuration=vconfig)
    vprop.propagate(vstate, t=0.7, tlist=[0, 0.35, 0.7])

    compiled = _compile_superoperator_generator(
        hamiltonian=hamiltonian,
        collapse_operators=collapse_ops,
        jump=-vconfig[0] * raw_jump,
    )
    unnormalized = mesolve(
        H=compiled,
        rho0=rho0,
        tlist=[0, 0.35, 0.7],
        c_ops=[],
        options={"normalize_output": False, "atol": 1e-10, "rtol": 1e-10, "nsteps": 200000},
    )
    normalized = mesolve(
        H=compiled,
        rho0=rho0,
        tlist=[0, 0.35, 0.7],
        c_ops=[],
        options={"normalize_output": True, "atol": 1e-10, "rtol": 1e-10, "nsteps": 200000},
    )

    assert _dense_close(vstate, unnormalized.states[-1], tol=5e-9)
    assert abs(complex(vstate.tr()) - 1.0) > 1e-6
    assert not _dense_close(normalized.states[-1], unnormalized.states[-1], tol=1e-6)


def test_compiled_superoperator_runtime_sanity():
    a = destroy(2)
    ad = create(2)
    sx = a + ad
    rho0 = fock(2, 1) * fock(2, 1).dag()

    hamiltonian = [0.2 * num(2), [0.3 * sx, lambda t, args=None: exp(0.4j * t)]]
    collapse_ops = [a, 0.15 * sprepost(a, ad)]
    compiled = _compile_superoperator_generator(
        hamiltonian=hamiltonian,
        collapse_operators=collapse_ops,
        jump=-0.15 * sprepost(a, ad),
    )

    tlist = [0.02 * i for i in range(101)]
    start = perf_counter()
    mesolve(
        H=compiled,
        rho0=rho0,
        tlist=tlist,
        c_ops=[],
        options={"normalize_output": False, "nsteps": 200000},
    )
    elapsed = perf_counter() - start
    assert elapsed < 1.0
