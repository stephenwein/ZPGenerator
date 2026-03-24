from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectionAction:
    component_port_index: int
    permutation_target: int
    kind: str
    element_port_index: int | None = None


@dataclass(frozen=True)
class ConnectionPlan:
    new_mode_number: int
    existing_actions: tuple[ConnectionAction, ...]
    padding_targets: tuple[int, ...]
    remaining_element_port_indices: tuple[int, ...]
    permutation: tuple[int, ...]
