def copy_solver_options(options):
    if options is None:
        return None

    if isinstance(options, dict):
        return dict(options)

    for method in ("as_dict", "to_dict"):
        if hasattr(options, method):
            data = getattr(options, method)()
            if isinstance(data, dict):
                return dict(data)

    try:
        return dict(options)
    except (TypeError, ValueError):
        pass

    try:
        data = {key: value for key, value in vars(options).items() if not key.startswith("_")}
    except TypeError:
        data = {}

    if not data:
        for name in dir(options):
            if name.startswith("_"):
                continue
            value = getattr(options, name)
            if callable(value):
                continue
            data[name] = value

    return data


def default_virtual_solver_options(precision: int) -> dict:
    return {
        "nsteps": 500000,
        "atol": 10 ** -precision,
        "rtol": 10 ** -precision,
        "normalize_output": False,
    }


def mesolve_options(options, force_unnormalized: bool):
    options_dict = copy_solver_options(options)
    if options_dict is None:
        return {"normalize_output": False} if force_unnormalized else None

    if force_unnormalized:
        options_dict["normalize_output"] = False
    return options_dict


def generator_solver_options(options, precision: int):
    options = default_virtual_solver_options(precision) if options is None else options
    return mesolve_options(options, force_unnormalized=True)
