from ..time import OperatorInputList, CompositeTimeOperator, sum_flatten
from ..time.evaluate.quadruple import EvaluatedQuadruple
from ..time.evaluate.dirac import EvaluatedDiracOperator
from .quantum import SystemCollection
from .natural import NaturalSystem, HamiltonianBase, EnvironmentBase
from typing import Union, List
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluatedControl:
    continuous: EvaluatedQuadruple
    instantaneous: EvaluatedDiracOperator


def combine_evaluated_controls(controls: List["EvaluatedControl"]) -> "EvaluatedControl":
    return EvaluatedControl(
        continuous=sum((control.continuous for control in controls), EvaluatedQuadruple()),
        instantaneous=sum((control.instantaneous for control in controls), EvaluatedDiracOperator()),
    )


class ChannelBase(CompositeTimeOperator):
    """
    One or more instant operators or superoperators that apply gates or channels to a source state directly.
    """

    def __init__(self,
                 operators: OperatorInputList = None,
                 parameters: dict = None,
                 name: str = None):
        super().__init__(operators=operators, parameters=parameters, name=name)

    def _check_objects(self):
        super()._check_objects()
        if not all(not op.has_interval for op in self._objects):
            raise ValueError("Operators must be instant.")

    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None):
        return self.is_super and self.is_time_dependent(t, self.set_parameters(parameters))


class ControlBase(NaturalSystem):
    """A quantum system that controls the behaviour of another system's evolution in time"""

    def __init__(self,
                 hamiltonian: Union[HamiltonianBase, OperatorInputList] = None,
                 environment: Union[EnvironmentBase, OperatorInputList] = None,
                 channel: Union[ChannelBase, OperatorInputList] = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):
        """
        :param hamiltonian: the control potential defining how the control impacts a system HamiltonianBase.
        :param environment: the response defining how the control impacts the system environment.
        :param channel: the time operator defining how the control impacts the system state directly.
        """

        self.channel = channel if isinstance(channel, ChannelBase) else ChannelBase(channel)
        self.channel.default_name = '_channel'

        super().__init__(hamiltonian=hamiltonian, environment=environment, parameters=parameters, name=name,
                         types=[HamiltonianBase, EnvironmentBase, ChannelBase] if types is None else types)

        self._sync_objects()
        self._check_objects()

    def _check_objects(self):
        super()._check_objects()
        if self.channel.subdims and self.hamiltonian.subdims:
            if self.channel.subdims != self.hamiltonian.subdims:
                raise ValueError("Channel and HamiltonianBase must share the same dimensions")
        if self.channel.subdims and self.environment.subdims:
            if self.channel.subdims != self.environment.subdims:
                raise ValueError("Channel and EnvironmentBase must share the same dimensions")
    @property
    def subdims(self):
        return self.hamiltonian.subdims if self.hamiltonian.operators \
            else self.environment.subdims if self.environment.subdims \
            else self.channel.subdims

    @property
    def objects(self):
        return [self.hamiltonian, self.environment, self.channel]

    def _sync_objects(self):
        self._objects = self.objects
        self.set_children(self._objects)

    def _add(self, control, parameters: dict = None, name: str = None):
        if isinstance(control, HamiltonianBase):
            self.hamiltonian.add(control, parameters, name)
        elif isinstance(control, EnvironmentBase):
            self.environment.add(control, parameters, name)
        elif isinstance(control, ChannelBase):
            self.channel.add(control, parameters, name)

    def evaluate_control(self, t: float, parameters: dict = None) -> EvaluatedControl:
        parameters = self.set_parameters(parameters)
        continuous, instantaneous = self.evaluate_natural_dynamics(t, parameters)
        instantaneous = instantaneous + self.channel.evaluate_dirac(t, parameters)
        return EvaluatedControl(continuous=continuous, instantaneous=instantaneous)

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return self.evaluate_control(t, parameters).continuous

    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        return self.evaluate_control(t, parameters).instantaneous


class CompositeControl(SystemCollection):
    """
    A collection of quantum control subsystems that are summed together when evaluated
    """

    def __init__(self,
                 systems: Union[ControlBase, List[ControlBase]] = None,
                 parameters: dict = None,
                 name: dict = None,
                 types: list = None):
        super().__init__(systems=systems, parameters=parameters, name=name, rule=sum_flatten,
                         types=[ControlBase, CompositeControl] if types is None else types)

    def evaluate_control(self, t: float, parameters: dict = None) -> EvaluatedControl:
        parameters = self.set_parameters(parameters)
        return combine_evaluated_controls([system.evaluate_control(t, parameters) for system in self._objects])

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return self.evaluate_control(t, parameters).continuous

    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        return self.evaluate_control(t, parameters).instantaneous


class ControlledSystem(NaturalSystem):
    """
    A natural system that is controlled by additional control subsystems
    """

    def __init__(self,
                 hamiltonian: Union[HamiltonianBase, OperatorInputList] = None,
                 environment: Union[EnvironmentBase, OperatorInputList] = None,
                 control: Union[CompositeControl, ControlBase, HamiltonianBase, EnvironmentBase, ChannelBase] = None,
                 states: dict = None,
                 operators: dict = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):

        if isinstance(control, HamiltonianBase):
            self.control = CompositeControl(ControlBase(hamiltonian=control))
        elif isinstance(control, EnvironmentBase):
            self.control = CompositeControl(ControlBase(environment=control))
        elif isinstance(control, ChannelBase):
            self.control = CompositeControl(ControlBase(channel=control))
        elif isinstance(control, ControlBase):
            self.control = CompositeControl(control)
        elif isinstance(control, CompositeControl):
            self.control = control
        else:
            self.control = CompositeControl()
        self.control.default_name = '_control'

        super().__init__(hamiltonian=hamiltonian, environment=environment, states=states, operators=operators,
                         parameters=parameters, name=name,
                         types=[HamiltonianBase, EnvironmentBase, ChannelBase, ControlBase] if types is None else types)

        self._sync_objects()
        self._check_objects()

    def _check_objects(self):
        super()._check_objects()
        if self.control.operator_list:
            if self.subdims != self.control.subdims:
                raise ValueError("Controls must share the same dimensions as the HamiltonianBase")

    @property
    def objects(self):
        return [self.hamiltonian, self.environment, self.control]

    def _sync_objects(self):
        self._objects = self.objects
        self.set_children(self._objects)

    def _add(self, operator: Union[HamiltonianBase, EnvironmentBase, ControlBase, CompositeControl],
             parameters: dict = None, name: str = None):
        super()._add(operator, parameters, name)
        if isinstance(operator, ControlBase):
            self.control.add(operator, parameters, name)

    def evaluate_control(self, t: float, parameters: dict = None) -> EvaluatedControl:
        return self.control.evaluate_control(t, self.set_parameters(parameters))

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        parameters = self.set_parameters(parameters)
        natural, _ = self.evaluate_natural_dynamics(t, parameters)
        return natural + self.evaluate_control(t, parameters).continuous

    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        parameters = self.set_parameters(parameters)
        _, natural = self.evaluate_natural_dynamics(t, parameters)
        return natural + self.evaluate_control(t, parameters).instantaneous
