from typing import List

import numpy as np
from qutip import Qobj, operator_to_vector, ptrace
from ..time.parameters import Parameters


def normalise_context_value(value):
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(value.shape), tuple(value.reshape(-1).tolist()))
    if isinstance(value, dict):
        return tuple(sorted((k, normalise_context_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(normalise_context_value(v) for v in value)
    return value


def build_simulation_context(parameters: dict = None, bin_list: list = None, basis: List[Qobj] = None):
    parameter_context = None
    if parameters is not None:
        if isinstance(parameters, Parameters):
            parameters = parameters.dict
        parameter_context = tuple(sorted((k, normalise_context_value(v)) for k, v in parameters.items()))

    bin_context = None if bin_list is None else tuple(bin_list)
    basis_context = None
    if basis is not None:
        basis_context = tuple((tuple(state.shape), tuple(state.dims[0]), tuple(state.dims[1])) for state in basis)

    return {
        "parameters": parameter_context,
        "bin_list": bin_context,
        "basis": basis_context,
    }


def prepare_channel_basis(basis: List[Qobj]):
    if not basis:
        raise ValueError("Please provide a state basis for a subspace to construct the effective channel")
    # Take outer product of orthonormal set to build unit elements of the density matrix.
    return [psi1 * psi2.dag() for psi2 in basis for psi1 in basis]


def invert_tensors(tensors) -> bool:
    contains_unnormalised_detector = False
    for tensor in tensors:
        contains_unnormalised_detector = contains_unnormalised_detector or tensor.invert()
    return contains_unnormalised_detector


def build_channels_from_results(results: list, basis: List[Qobj], dims: List[int] = None, select: List[int] = None):
    channels = {}

    target_dims = basis[0].dims if dims is None else [dims, dims]

    for key in results[0].keys():  # looping over all measurement outcomes
        channel_input = []
        new_dims = target_dims
        for states in results:  # for each set of states in conditional states simulated
            state = states[key].copy()  # copy so channel extraction cannot mutate cached source states
            state.dims = target_dims  # apply desired sub-dimensions
            if select:
                state = ptrace(state, select)  # trace out desired subspaces
            new_dims = state.dims
            channel_input.append(operator_to_vector(state).full())
        channel_input = np.hstack(channel_input)  # rearrange into matrix
        channels[key] = Qobj(channel_input, dims=[new_dims, new_dims], superrep='super')

    return channels
