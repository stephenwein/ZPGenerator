from zpgenerator.dynamic.control import Control
from zpgenerator.dynamic.pulse import Pulse
from qutip import destroy
import pytest


def test_control_operator_rejects_interval_pulses():
    with pytest.raises(ValueError, match="only Dirac delta"):
        Control.operator(Pulse.square(), destroy(2))


def test_control_operator_requires_dirac_content():
    with pytest.raises(ValueError, match="at least one Dirac delta"):
        Control.operator(Pulse(), destroy(2))
