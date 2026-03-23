from qutip import Qobj


def qobj_compat(*args, **kwargs):
    # Backward-compatibility aliases used across this codebase and tests.
    if "inpt" in kwargs and "arg" not in kwargs and not args:
        kwargs["arg"] = kwargs.pop("inpt")
    if "type" in kwargs:
        qtype = kwargs.pop("type")
        if qtype == "super" and "superrep" not in kwargs:
            kwargs["superrep"] = "super"
    return Qobj(*args, **kwargs)
