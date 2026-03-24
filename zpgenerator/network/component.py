from ..time import TimeFunctionCollection, merge_times, EvaluatedQuadruple
from ..time.evaluate.cache import DefaultCache
from ..system import AElement, ScattererBase, AScatteringMatrix
from .composition_request import CompositionRequest
from .connection_plan import ConnectionAction, ConnectionPlan
from .element import ElementCollection
from .detector import ADetectorGate
from .mode_mapping import normalize_mode_position, is_mode_position_token
from .port import InputLayer, OutputLayer, OutputPort, InputPort
from typing import Union, List
from abc import abstractmethod
from copy import deepcopy
from qutip import Qobj
from frozendict import frozendict
from itertools import chain


class AComponent(ElementCollection):
    """A collection of one or more elements with a layer of input and output ports"""

    @property
    @abstractmethod
    def input(self) -> InputLayer:
        pass

    @property
    @abstractmethod
    def output(self) -> OutputLayer:
        pass

    def bin_all_detectors(self, bin_name: str):
        self.output.bin_all_detectors(bin_name)

    @property
    def input_modes(self) -> int:
        return self.input.open_modes

    @property
    def output_modes(self) -> int:
        return self.output.open_modes

    @property
    def type(self) -> str:
        if self.input.is_closed and self.output.is_closed:
            return 'Processor'
        elif not self.output.is_open and self.output.is_monitored:
            return 'Detector'
        elif self.input.is_closed:
            return 'Source'
        elif self.output.is_closed:
            return 'Trace'
        else:
            return 'Element'

    @abstractmethod
    def mask(self):
        pass

    @abstractmethod
    def unmask(self):
        pass

    @property
    @abstractmethod
    def is_masked(self) -> bool:
        pass


class Component(AComponent):
    """
    A collection of elements with some inputs and outputs having different statuses
    """

    ComponentInputTypes = Union[AElement, List[AElement], ADetectorGate, List[ADetectorGate]]

    def __init__(self,
                 elements: ComponentInputTypes = None,
                 parameters: dict = None,
                 name: str = None,
                 masked: bool = True,
                 types: list = None):
        self._is_masked = masked
        self._input = InputLayer.make(0)
        self._output = OutputLayer.make(0)
        self.permutations = []
        self._next_pos = 0

        super().__init__(parameters=parameters, name=name, types=[AElement, ADetectorGate] if not types else types)
        self.set_children([self._objects, self._output.ports])
        if isinstance(elements, list):
            for elm in elements:
                self.add(elm)
        else:
            self.add(elements)

    def _check_objects(self):
        super(TimeFunctionCollection, self)._check_objects()

    @property
    def input(self) -> InputLayer:
        return self._input

    @property
    def output(self) -> OutputLayer:
        return self._output

    @property
    def modes(self):
        return len(self.permutations[0]) if self.permutations else 0

    @property
    def is_masked(self) -> bool:
        return self._is_masked

    @is_masked.setter
    def is_masked(self, value):
        self._is_masked = value

    def mask(self):
        self._is_masked = True

    def unmask(self):
        self._is_masked = False

    def unmasked_position(self, position: int):
        """
        :param position: a masked (not closed) or unmasked position
        :return: the corresponding unmasked position
        """
        if self._elements and self.is_masked:
            try:
                return [i for i, port in enumerate(self.output.ports) if not port.is_closed][position]
            except IndexError:
                return self.modes
        else:
            return position

    def masked_position(self, position: int):
        """
        :param position: a masked (not closed) or unmasked position
        :return: the corresponding unmasked position
        """
        if self._elements and self.is_masked:
            return position
        else:
            return [port.is_closed for i, port in enumerate(self.output.ports)
                    if i < self.unmasked_position(position)].count(False)

    def get_port_number(self, position: Union[str, int]) -> int:
        position = normalize_mode_position(position)

        if isinstance(position, str):
            if not self.elements:
                raise ValueError("Processor has no ports.")
            number = self.output.get_port_number(position)
            if number is None:
                raise ValueError("No port named " + position + ".")
            position = number
        unmasked_position = self.unmasked_position(position)
        if self.is_masked and self._elements and unmasked_position < self.modes:
            if self.output.ports[unmasked_position].is_closed:
                raise ValueError("Selected port to connect must not be closed.")
            return self.masked_position(position)
        else:
            return position

    def _position_to_add(self, position: int):
        """
        :param position: a masked or unmasked position
        :return: the corresponding masked position, excluding monitored ports
        """
        if not self._elements or position >= self.modes:
            return position
        else:
            return [port.is_open for i, port in enumerate(self.output.ports)
                    if i < self.unmasked_position(position)].count(True)

    # annoying workaround to make signature of networks similar to Perceval while keeping other collections consistent
    # it is silly that Perceval doesn't consider 'position' a keyword argument in the second argument position
    @staticmethod
    def _resolve_add_call(position, element):
        if element is None:
            if is_mode_position_token(position):
                raise ValueError("Please specify an element to add")
            return 0, position
        return normalize_mode_position(position), element

    @staticmethod
    def _coerce_element(element):
        if hasattr(element, 'compute_unitary'):
            return Component(ScattererBase(Qobj(element.compute_unitary())))
        return element

    @staticmethod
    def _prepare_component_for_binning(element: AComponent, parameters: dict = None,
                                       name: str = None, bin_name: str = None):
        if not bin_name:
            return element, parameters, name
        element = deepcopy(element)
        element.bin_all_detectors(bin_name)
        if parameters or name:
            element.update_default_parameters(parameters)
            parameters = None
            element.name = name
            name = None
        return element, parameters, name

    @classmethod
    def _build_composition_request(cls,
                                   position,
                                   element,
                                   parameters: dict = None,
                                   name: str = None,
                                   bin_name: str = None) -> CompositionRequest | None:
        position, element = cls._resolve_add_call(position, element)
        if element is None:
            return None
        return CompositionRequest(
            position=position,
            element=element,
            parameters=parameters,
            name=name,
            bin_name=bin_name,
        )

    def _normalize_composition_request(self, request: CompositionRequest) -> CompositionRequest:
        element = self._coerce_element(request.element)
        position = self.get_port_number(request.position)
        parameters = request.parameters
        name = request.name

        if isinstance(element, AComponent):
            element, parameters, name = self._prepare_component_for_binning(
                element,
                parameters,
                name,
                request.bin_name,
            )

        return CompositionRequest(
            position=position,
            element=element,
            parameters=parameters,
            name=name,
            bin_name=request.bin_name,
        )

    def add(self,
            position: Union[int, str, ComponentInputTypes],
            element: ComponentInputTypes = None,
            parameters: dict = None, name: str = None, bin_name: str = None):
        # Component composition stays single-target for now. Processor.add expands
        # multi-position mappings before delegating here.
        request = self._build_composition_request(position, element, parameters=parameters, name=name, bin_name=bin_name)
        if request is None:
            return
        if isinstance(request.element, list):
            if not request.element:
                return
            for elm in request.element:
                self.add(request.position, elm,
                         parameters=request.parameters,
                         name=request.name,
                         bin_name=request.bin_name)
            return

        request = self._normalize_composition_request(request)

        if isinstance(request.element, AElement):
            self._add_element(request.element, request.position, request.parameters, request.name)
        elif isinstance(request.element, ADetectorGate):
            self._add_detector(request.element, request.position, request.parameters, request.name, request.bin_name)
        elif request.element is not None:
            raise TypeError("Can only add AElement or ADetectorGate to a Component")

    def _add_element(self, element: AElement, position: int = None, parameters: dict = None, name: str = None):
        if self._elements:
            unmasked_position = self.unmasked_position(position)
            if unmasked_position < self.modes:
                if not self.output.ports[unmasked_position].is_open:
                    raise ValueError("Selected port to connect must be open.")
        position = self._position_to_add(position)  # maps the input position (masked or unmasked) to an open position
        self._next_pos = position
        super().add(element, parameters, name)

    def _add_detector(self, element: ADetectorGate, position: int = None,
                      parameters: dict = None, name: str = None, bin_name: str = None):
        position = self.unmasked_position(position)
        if not self.elements:
            raise ValueError("Cannot add a detector before adding at least one element.")
        if position >= self.modes:
            raise ValueError("Selected port to connect does not exist.")
        if self.output.ports[position].is_closed:
            raise ValueError("Selected port to connect must not be closed.")
        self._output.ports[position].add(element, parameters=parameters, name=name, bin_name=bin_name)

    def __floordiv__(self, other):
        if isinstance(other, tuple):
            if len(other) == 2:
                position = other[0]
                element = other[1]
            else:
                position = 0
                element = other[0]
        else:
            position = 0
            element = other
        self.add(position, element)
        return self

    def _check_add(self, element, parameters: dict = None, name: str = None):
        if isinstance(element, AElement):
            self._connect(element)
        return super()._check_add(element, parameters, name)

    def _compute_new_mode_number(self, element: AElement) -> int:
        connected_modes = min(self.output_modes, element.input_modes) - self._next_pos
        return self.modes + element.modes - connected_modes

    @staticmethod
    def _partition_element_ports(element: AElement):
        if isinstance(element, AComponent):
            element_active_ports = []
            element_inactive_ports = []
            for i, input_port in enumerate(element.input.ports):
                port_to_add = element.output.ports[i]
                if input_port.is_closed:
                    element_inactive_ports.append([i, input_port, port_to_add])
                else:
                    element_active_ports.append([i, input_port, port_to_add])
            return element_active_ports, element_inactive_ports
        return [[i, InputPort(), OutputPort()] for i in range(element.modes)], []

    def _plan_connection(self, element: AElement) -> ConnectionPlan:
        new_mode_number = self._compute_new_mode_number(element)
        element_active_ports, element_inactive_ports = self._partition_element_ports(element)
        active_port_indices = iter(port[0] for port in element_active_ports)
        inactive_port_indices = [port[0] for port in element_inactive_ports]
        element_extra_mode_number = iter(range(element.modes, new_mode_number))
        position_counter = self._next_pos
        existing_actions = []

        for i, output_port in enumerate(self.output.ports):
            if output_port.is_open:
                if position_counter:
                    existing_actions.append(ConnectionAction(
                        component_port_index=i,
                        permutation_target=next(element_extra_mode_number),
                        kind='preserve',
                    ))
                    position_counter -= 1
                else:
                    try:
                        port_index = next(active_port_indices)
                        existing_actions.append(ConnectionAction(
                            component_port_index=i,
                            permutation_target=port_index,
                            kind='attach',
                            element_port_index=port_index,
                        ))
                    except StopIteration:
                        existing_actions.append(ConnectionAction(
                            component_port_index=i,
                            permutation_target=next(element_extra_mode_number),
                            kind='preserve',
                        ))
            else:
                existing_actions.append(ConnectionAction(
                    component_port_index=i,
                    permutation_target=next(element_extra_mode_number),
                    kind='preserve',
                ))

        padding_targets = tuple(next(element_extra_mode_number) for _ in range(position_counter))
        remaining_element_port_indices = tuple(sorted(list(active_port_indices) + inactive_port_indices))
        permutation = tuple(action.permutation_target for action in existing_actions) + \
            padding_targets + remaining_element_port_indices

        return ConnectionPlan(
            new_mode_number=new_mode_number,
            existing_actions=tuple(existing_actions),
            padding_targets=padding_targets,
            remaining_element_port_indices=remaining_element_port_indices,
            permutation=permutation,
        )

    def _build_connection_layers(self, plan: ConnectionPlan, element_port_lookup: dict[int, list]):
        new_output = OutputLayer()
        new_input = InputLayer()

        for action in plan.existing_actions:
            new_input.add(deepcopy(self.input.ports[action.component_port_index]))
            output_port = deepcopy(self.output.ports[action.component_port_index])
            if action.kind == 'attach':
                port = element_port_lookup[action.element_port_index]
                new_output.add(deepcopy(port[2]))
            else:
                new_output.add(output_port)

        for _ in plan.padding_targets:
            new_output.add(OutputPort())
            new_input.add(InputPort())

        for port_index in plan.remaining_element_port_indices:
            port = element_port_lookup[port_index]
            new_input.add(deepcopy(port[1]))
            new_output.add(deepcopy(port[2]))

        return new_input, new_output

    def _connect(self, element):
        """
        Connects open input ports of an element to the open outputs ports of a component
        :param element: an element to connect
        """
        element_active_ports, element_inactive_ports = self._partition_element_ports(element)
        element_port_lookup = {port[0]: port for port in element_active_ports + element_inactive_ports}
        plan = self._plan_connection(element)
        new_input, new_output = self._build_connection_layers(plan=plan, element_port_lookup=element_port_lookup)

        self._adjust_orderings(list(plan.permutation))
        self._output = new_output
        self.set_children([self._objects, self._output.ports])
        self._input = new_input

    def _adjust_orderings(self, perm: List[int]):
        pad = list(range(self.modes, len(perm)))
        self.permutations = [perm + pad for perm in self.permutations]  # append modes to all components
        self.permutations.append(perm)  # add permutation for most recent component

    @DefaultCache(time_arg=False)
    def times(self, parameters: dict = None):
        return merge_times([merge_times([function.times(parameters) for function in self._objects]),
                            self.output.times(self.set_parameters(parameters))])

    @DefaultCache(time_arg=True)
    def evaluate_quadruple(self, t: float, parameters: Union[dict, frozendict] = None):
        if self._elements:
            return self._rule(self.gather_quadruples(t, parameters))
        else:
            return EvaluatedQuadruple()

    def gather_quadruples(self, t: float, parameters: dict = None) -> List[EvaluatedQuadruple]:
        parameters = self.set_parameters(parameters)
        quad_list = [[quad.match(self.permutations[i]) for quad in component.gather_quadruples(t, parameters)]
                     for i, component in enumerate(self._elements)]
        return list(chain(*quad_list))

    @DefaultCache(time_arg=True)
    def is_time_dependent(self, t: float, parameters: dict = None) -> bool:
        return any(function.is_time_dependent(t, parameters) for function in self._objects) \
            or self.output.is_time_dependent(t, parameters)

    @DefaultCache(time_arg=True)
    def is_nonhermitian_time_dependent(self, t: float, parameters: dict = None) -> bool:
        return any(element.is_nonhermitian_time_dependent(t, parameters)
                   if not isinstance(element, AScatteringMatrix) else
                   element.is_time_dependent(t, parameters) for element in self.elements.values()) \
            or self.output.is_time_dependent(t, parameters)

    def _cache_clear(self):
        super()._cache_clear()
        self.times.cache_clear()
        self.evaluate_quadruple.cache_clear()
        self.is_time_dependent.cache_clear()
        self.is_nonhermitian_time_dependent.cache_clear()


def make_masked_source(source: AComponent, port: int):
    port = source.unmasked_position(port) if isinstance(source, Component) else port

    masked_source = Component(source, masked=False)
    for i in range(masked_source.modes):
        if i != port:
            masked_source.output.ports[i].close()
    masked_source.mask()

    return masked_source
