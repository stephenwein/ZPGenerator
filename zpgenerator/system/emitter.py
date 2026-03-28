from ..time import ATimeOperator, Operator, TimeVectorOperator
from .scatterer import AElement
from .natural import AQuantumSystem, NaturalSystem, HamiltonianBase, EnvironmentBase
from .control import ChannelBase, ControlBase, CompositeControl, ControlledSystem
from ..time.evaluate.quadruple import EvaluatedQuadruple
from ..time.evaluate.dirac import EvaluatedDiracOperator
from typing import Union, List
from qutip import Qobj
from abc import abstractmethod
from copy import deepcopy


def _normalise_operator_inputs(operators):
    if operators is None:
        return []
    if isinstance(operators, tuple):
        return list(operators)
    return operators


def _operators_overlap(left, right):
    for left_op in left:
        for right_op in right:
            try:
                if left_op == right_op:
                    return True
            except Exception:
                if left_op is right_op:
                    return True
            try:
                if not left_op.is_time_dependent(0) and not right_op.is_time_dependent(0):
                    if left_op.evaluate(0) == right_op.evaluate(0):
                        return True
            except Exception:
                pass
    return False


# maybe could be made a subclass of EnvironmentBase?
class LindbladVector(AQuantumSystem, TimeVectorOperator):
    """
    A vector of mode-coupling Lindblad collapse operators that may have time-dependent amplitudes.
    """

    def __init__(self,
                 operators: Union[List[ATimeOperator], List[Qobj], List[Operator]] = None,
                 parameters: dict = None,
                 name: str = None):
        super().__init__(operators=operators, parameters=parameters, name=name)

    # The number of collapse operators
    @property
    def modes(self):
        modes = 0
        for op in self._objects:
            modes += (op.modes if isinstance(op, LindbladVector) else 1)  # allows for nested LindbladVectors
        return modes

    def _check_objects(self):
        super()._check_objects()
        if not all(not op.is_super for op in self._objects):
            raise ValueError("Cannot add superoperators.")
        if not all(not op.has_instant for op in self._objects):
            raise ValueError("Cannot add instant operators.")

    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None):
        return self.is_time_dependent(t, parameters)

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        return EvaluatedQuadruple(transitions=super().partial_evaluate(t, parameters))


class AQuantumEmitter(AQuantumSystem, AElement):

    @property
    @abstractmethod
    def initial_state(self):
        pass

    @property
    @abstractmethod
    def initial_time(self):
        pass

    @property
    @abstractmethod
    def modes(self) -> int:
        pass

    @property
    def is_emitter(self) -> bool:
        return True


class EmitterBase(AQuantumEmitter, ControlledSystem):
    """
    A controlled quantum system with a list of transition operators specifying coupling modes to emit light
    """

    def __init__(self,
                 hamiltonian: Union[HamiltonianBase, Qobj, Operator] = None,
                 environment: Union[EnvironmentBase, List[Qobj], List[Operator]] = None,
                 control: Union[CompositeControl, ControlBase, HamiltonianBase, EnvironmentBase, ChannelBase] = None,
                 transitions: Union[LindbladVector, List[Qobj], List[ATimeOperator]] = None,
                 states: dict = None,
                 operators: dict = None,
                 parameters: dict = None,
                 name: str = None,
                 types: list = None):

        self.transitions = LindbladVector(transitions) if not isinstance(transitions, LindbladVector) else \
            LindbladVector() if transitions is None else transitions
        self.transitions.default_name = '_transitions'

        super().__init__(hamiltonian=hamiltonian, environment=environment, control=control,
                         states=states, operators=operators, parameters=parameters, name=name,
                         types=[HamiltonianBase, EnvironmentBase, ChannelBase, ControlBase, LindbladVector]
                         if types is None else types)
        self._sync_objects()
        self._check_objects()

        self._initial_time = None
        self._initial_state = None
        self._transition_names = None

        self.system = None

    def _check_objects(self):
        super()._check_objects()
        if self.transitions.operator_list:
            if not self.environment.operator_list:
                raise ValueError("Transitions cannot occur without an environment")
            if self.subdims != self.transitions.subdims:
                raise ValueError("Transition operator dimensions must match the dimensions of the system.")

    @property
    def modes(self):
        return self.transitions.modes

    @property
    def transition_names(self):
        if self._transition_names is not None:
            return self._transition_names
        if self.modes == 1 and self.name:
            return [self.name]
        return [f'mode_{i}' for i in range(self.modes)]

    @transition_names.setter
    def transition_names(self, names):
        if names is None:
            self._transition_names = None
            return
        if len(names) != self.modes:
            raise ValueError("Transition names must match the number of emitter modes.")
        if len(set(names)) != len(names):
            raise ValueError("Transition names must be unique.")
        self._transition_names = list(names)

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Qobj):
        self._initial_state = state

    @property
    def initial_time(self):
        return self._initial_time

    @initial_time.setter
    def initial_time(self, time: Union[float, int]):
        self._initial_time = time

    @property
    def objects(self):
        return [self.hamiltonian, self.environment, self.control, self.transitions]

    def _sync_objects(self):
        self._objects = self.objects
        self.set_children(self._objects)

    def set_system(self,
                   system: AQuantumSystem,
                   transitions: Union[LindbladVector, List[Qobj], List[ATimeOperator]] = None):
        self.hamiltonian = system.hamiltonian if hasattr(system, 'hamiltonian') else HamiltonianBase()
        self.environment = system.environment if hasattr(system, 'environment') else EnvironmentBase()
        self.control = system.control if hasattr(system, 'control') else CompositeControl()
        self.transitions = LindbladVector(transitions) if not isinstance(transitions, LindbladVector) else \
            LindbladVector() if transitions is None else transitions
        self.transitions.default_name = '_transitions'

        self.states = system.states if hasattr(system, 'states') else {}
        self.operators = system.operators if hasattr(system, 'operators') else {}

        self._default_parameters = system.local_default_parameters if hasattr(system, 'local_default_parameters') else {}
        self.name = system.name
        self._sync_objects()
        self._check_objects()
        self.system = system
        self._transition_names = None

    @classmethod
    def from_master_equation(cls,
                             hamiltonian: Union[HamiltonianBase, Qobj, Operator] = None,
                             monitored: Union[dict, LindbladVector, List[Qobj], List[ATimeOperator]] = None,
                             environment: Union[EnvironmentBase, List[Qobj], List[Operator]] = None,
                             states: dict = None,
                             operators: dict = None,
                             initial_state: Union[Qobj, str] = None,
                             initial_time: Union[float, int] = None,
                             parameters: dict = None,
                             name: str = None):
        """
        Build an emitter directly from a master-equation style specification.

        The monitored collapse operators are included both in the dissipative environment and in the
        emitter transitions, so they contribute to the Liouvillian and define the collected output modes.
        """
        monitored_names = list(monitored.keys()) if isinstance(monitored, dict) else None
        monitored = _normalise_operator_inputs(list(monitored.values()) if isinstance(monitored, dict) else monitored)
        environment = _normalise_operator_inputs(environment)

        transitions = monitored if isinstance(monitored, LindbladVector) else \
            LindbladVector(monitored, parameters=parameters)

        base_environment = environment if isinstance(environment, EnvironmentBase) else \
            EnvironmentBase(environment, parameters=parameters)
        if _operators_overlap(transitions.operator_list, base_environment.operator_list):
            raise ValueError(
                "Monitored channels are added to the dissipative environment automatically. "
                "Do not pass the same collapse operator in both 'monitored' and 'environment'."
            )
        combined_environment = deepcopy(base_environment)
        if transitions.operator_list:
            combined_environment.add(deepcopy(transitions.operator_list))

        system = NaturalSystem(hamiltonian=hamiltonian,
                               environment=combined_environment,
                               states=states,
                               operators=operators,
                               parameters=parameters,
                               name=name)

        emitter = cls()
        emitter.set_system(system=system, transitions=deepcopy(transitions))
        if monitored_names is not None:
            emitter.transition_names = monitored_names

        if isinstance(initial_state, str):
            if initial_state not in emitter.states:
                raise ValueError(f"Unknown initial state '{initial_state}'.")
            emitter.initial_state = emitter.states[initial_state]
        elif initial_state is not None:
            emitter.initial_state = initial_state

        if initial_time is not None:
            emitter.initial_time = initial_time

        return emitter

    def _add(self, system, parameters: dict = None, name: str = None):
        super()._add(system, parameters, name)
        if isinstance(system, LindbladVector):
            self.transitions.add(system, parameters, name)

    def evaluate_quadruple(self, t: float, parameters: dict = None) -> EvaluatedQuadruple:
        parameters = self.set_parameters(parameters)
        return super().evaluate_quadruple(t, parameters) + self.transitions.evaluate_quadruple(t, parameters)

    def evaluate_dirac(self, t: float, parameters: dict = None) -> EvaluatedDiracOperator:
        return super().evaluate_dirac(t, parameters)

    def gather_quadruples(self, t: float, parameters: dict = None) -> List[EvaluatedQuadruple]:
        return [self.evaluate_quadruple(t, parameters)]
