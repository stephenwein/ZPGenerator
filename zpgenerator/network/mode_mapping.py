from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Iterator, Union

ModePosition = Union[int, str]


def is_mode_position_token(value) -> bool:
    return isinstance(value, str) or (isinstance(value, Integral) and not isinstance(value, bool))


def normalize_mode_position(value) -> ModePosition:
    if isinstance(value, bool):
        raise TypeError("Position must be an integer index or named port string.")
    if isinstance(value, Integral):
        value = int(value)
        if value < 0:
            raise ValueError("Position must be non-negative.")
        return value
    if isinstance(value, str):
        if value == "":
            raise ValueError("Position name must be non-empty.")
        return value
    raise TypeError("Position must be an integer index or named port string.")


@dataclass(frozen=True)
class ModeMapping:
    positions: tuple[ModePosition, ...]

    def __iter__(self) -> Iterator[ModePosition]:
        return iter(self.positions)

    @property
    def single(self) -> ModePosition:
        if len(self.positions) != 1:
            raise ValueError("Position mapping must contain exactly one position here.")
        return self.positions[0]

    @classmethod
    def from_input(cls, mapping: Union[ModePosition, Iterable[ModePosition]]) -> "ModeMapping":
        if isinstance(mapping, (list, tuple)):
            if not mapping:
                raise ValueError("Position mapping cannot be empty.")
            return cls(tuple(normalize_mode_position(position) for position in mapping))
        return cls((normalize_mode_position(mapping),))
