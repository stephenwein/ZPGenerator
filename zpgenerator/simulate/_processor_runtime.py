from typing import List

from qutip import Qobj

from ..virtual import VGrove


def assert_continue_allowed(grove, current_time, existing_context: dict = None, simulation_context: dict = None):
    if grove is None:
        raise RuntimeError("No simulation to continue.")
    if current_time is None:
        raise RuntimeError("No current time.")
    if existing_context is not None and simulation_context is not None and existing_context != simulation_context:
        raise ValueError("Cannot continue simulation after changing parameters, bins, or basis.")


def initialise_or_resume_grove(current_time: float,
                               grove,
                               branch_order: list,
                               initial_time: float,
                               basis_states: List[Qobj],
                               branches: list):
    if current_time is None:
        grove = VGrove(initial_time=initial_time, states=basis_states)
        branch_order = grove.initialize(time=initial_time, branches=branches)
        current_time = initial_time
    return current_time, grove, branch_order


def build_propagation_times(current_time: float, component_times: list, final_time: float):
    return [current_time] + [t for t in component_times if current_time < t < final_time] + [final_time]
