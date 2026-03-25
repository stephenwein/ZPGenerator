from typing import List
from dataclasses import dataclass

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


@dataclass(frozen=True)
class PropagationStep:
    time: float
    has_instantaneous_operation: bool
    opens_measurement_branch: bool


def classify_propagation_step(component, t: float, parameters: dict = None, branch_times: list = None) -> PropagationStep:
    branch_times = [] if branch_times is None else branch_times
    return PropagationStep(
        time=t,
        has_instantaneous_operation=component.is_dirac(t, parameters),
        opens_measurement_branch=t in branch_times,
    )
