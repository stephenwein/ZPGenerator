from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class CompositionRequest:
    position: Union[int, str]
    element: object
    parameters: dict | None = None
    name: str | None = None
    bin_name: str | None = None
