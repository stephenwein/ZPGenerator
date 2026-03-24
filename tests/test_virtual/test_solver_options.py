from zpgenerator.virtual.solver_options import (
    copy_solver_options,
    default_virtual_solver_options,
    generator_solver_options,
    mesolve_options,
)


class _DummyOptions:
    def __init__(self, normalize_output=True):
        self.normalize_output = normalize_output
        self.keep = 7


class _DummyOptionsWithAsDict:
    def __init__(self):
        self.normalize_output = True
        self.store_states = False

    def as_dict(self):
        return {"normalize_output": self.normalize_output, "store_states": self.store_states}


def test_default_virtual_solver_options_precision():
    options = default_virtual_solver_options(precision=6)
    assert options["nsteps"] == 500000
    assert options["atol"] == 1e-6
    assert options["rtol"] == 1e-6
    assert options["normalize_output"] is False


def test_mesolve_options_none_respects_force_flag():
    assert mesolve_options(None, force_unnormalized=False) is None
    assert mesolve_options(None, force_unnormalized=True) == {"normalize_output": False}


def test_mesolve_options_dict_merges_without_mutating_input():
    original = {"atol": 1e-7, "normalize_output": True}

    unforced = mesolve_options(original, force_unnormalized=False)
    forced = mesolve_options(original, force_unnormalized=True)

    assert unforced == {"atol": 1e-7, "normalize_output": True}
    assert forced == {"atol": 1e-7, "normalize_output": False}
    assert original == {"atol": 1e-7, "normalize_output": True}


def test_mesolve_options_object_is_copied_and_never_mutated():
    opt = _DummyOptions(normalize_output=True)

    unforced = mesolve_options(opt, force_unnormalized=False)
    assert unforced == {"normalize_output": True, "keep": 7}
    assert opt.normalize_output is True
    assert opt.keep == 7

    forced = mesolve_options(opt, force_unnormalized=True)
    assert forced == {"normalize_output": False, "keep": 7}
    assert opt.normalize_output is True
    assert opt.keep == 7


def test_generator_solver_options_applies_defaults_and_forced_unnormalized():
    defaulted = generator_solver_options(None, precision=5)
    assert defaulted["atol"] == 1e-5
    assert defaulted["rtol"] == 1e-5
    assert defaulted["normalize_output"] is False

    custom = generator_solver_options({"rtol": 1e-4, "normalize_output": True}, precision=8)
    assert custom["rtol"] == 1e-4
    assert custom["normalize_output"] is False


def test_copy_solver_options_accepts_as_dict_provider():
    opt = _DummyOptionsWithAsDict()
    copied = copy_solver_options(opt)
    assert copied == {"normalize_output": True, "store_states": False}
