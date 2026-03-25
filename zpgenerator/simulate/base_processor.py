from ..network import AComponent, Component, ADetectorGate, TimeBin, DetectorGate
from ..system import AElement
from ..virtual import Generator, MeasurementBranch
from typing import Union, List
from qutip import Qobj
from ._processor_helpers import (
    build_simulation_context,
    prepare_channel_basis,
)
from ._processor_runtime import (
    assert_continue_allowed,
    build_propagation_times,
    classify_propagation_step,
    initialise_or_resume_grove,
)
from ._processor_results import ProcessorResultMaps


class ProcessorBase:
    """
    A photonic processor composed of one or more sources of light, a linear-optical circuit, and an array of detectors.
    """

    def __init__(self, component: Union[AElement, AComponent] = None):
        """
        :param component: a component to simulate
        """
        self.component = Component(component, masked=True)  # placing a mask
        # for port in self.component.input.ports:
        #     port.close()

        self._precision = 6

        self._grove = None
        self._current_time = None  # Note that current_time = None -> simulation will restart from initial conditions
        self._branch_order = []
        self._binned_detectors = {}
        self._simulation_context = None

        self._result_maps = ProcessorResultMaps()

        self._initial_state = None
        self._initial_time = None
        self._final_time = None

    @property
    def _probabilities(self):
        return self._result_maps.probabilities

    @_probabilities.setter
    def _probabilities(self, value: dict):
        self._result_maps.probabilities = value

    @property
    def _states(self):
        return self._result_maps.states

    @_states.setter
    def _states(self, value: dict):
        self._result_maps.states = value

    @property
    def _channels(self):
        return self._result_maps.channels

    @_channels.setter
    def _channels(self, value: dict):
        self._result_maps.channels = value

    @property
    def _contains_unnormalised_detector(self):
        return self._result_maps.contains_unnormalised_detector

    @_contains_unnormalised_detector.setter
    def _contains_unnormalised_detector(self, value: bool):
        self._result_maps.contains_unnormalised_detector = value

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

    def _check_simulate(self):
        if self.component.dim <= 1:
            raise ValueError("Processor must contain at least one quantum system.")
        if not any(port.is_monitored for port in self.component.output.ports):
            raise ValueError("Processor must contain at least one detector.")

    @property
    def modes(self):
        return self.component.modes - self.component.output.closed_modes

    def add(self, position: Union[int, str], element: Union[AElement, ADetectorGate],
            parameters: dict = None, name: str = None, bin_name: str = None):
        self._reset_grove()
        self.component.add(position, element, parameters, name, bin_name)

    @property
    def initial_state(self):
        return self._initial_state if self._initial_state else self.component.initial_state

    @initial_state.setter
    def initial_state(self, state: Union[Qobj, None]):
        self._reset_grove()
        if state:
            if state.dims[0] != self.component.subdims:
                raise ValueError(
                    "Input state dimension must match the dimension of all quantum systems contained by the processor"
                )
        self._initial_state = state

    @property
    def initial_time(self):
        return self._initial_time if self._initial_time is not None else self.component.initial_time

    @initial_time.setter
    def initial_time(self, t0: Union[float, int, None]):
        self._reset_grove()
        self._initial_time = t0

    @property
    def final_time(self):
        return self._final_time

    @final_time.setter
    def final_time(self, t: Union[float, int]):
        self._reset_grove()
        self._final_time = t

    def copy_conditions(self, processor: 'ProcessorBase'):
        self.initial_state = processor.initial_state
        self.initial_time = processor.initial_time
        self.final_time = processor.final_time
        self.precision = processor.precision

    @property
    def parameters(self) -> list[str]:
        return self.component.parameters

    @property
    def default_parameters(self) -> dict:
        return self.component.default_parameters

    def update_default_parameters(self, parameters: dict):
        self.component.update_default_parameters(parameters)

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._reset_grove()
        self._precision = precision

    def _measurement_branches(self, parameters: dict = None, bin_list: list = None):
        binned_detectors = self.component.output.binned_detectors
        bin_keys = list(binned_detectors.keys())
        if bin_list is not None:
            if not all(k < len(bin_keys) if isinstance(k, int) else k in bin_keys for k in bin_list):
                raise ValueError("One or more bins does not exist.")
            binned_detectors = {k: binned_detectors.get(bin_keys[k] if isinstance(k, int) else k) for k in bin_list}
        self.binned_detectors = binned_detectors
        branches = [MeasurementBranch(time_bin, parameters, name)
                    for name, time_bin in self.binned_detectors.items()]
        if not branches:  # we simulate the natural evolution (without any measurement)
            branches = [MeasurementBranch(time_bins=[TimeBin(detector=DetectorGate(resolution=None))])]
        return branches, self.binned_detectors

    def _get_initial_time(self, times: list):
        if self.initial_time is not None:
            initial_time = self.initial_time
        elif times:
            initial_time = times[0]
        else:
            initial_time = 0
        return initial_time

    def _get_final_time(self, times: list):
        if self.final_time is not None:
            final_time = self.final_time
        elif times:
            final_time = times[-1]
        else:
            raise ValueError("Must specify a final time.")  # could be replaced with a default convergence to steady state
        return final_time

    def _check_if_continue(self, continue_simulation: bool, simulation_context: dict):
        if continue_simulation:
            assert_continue_allowed(
                grove=self._grove,
                current_time=self._current_time,
                existing_context=self._simulation_context,
                simulation_context=simulation_context,
            )
            return True
        else:
            self._reset_grove()
            self._simulation_context = simulation_context
            return False

    def _reset_grove(self):
        self._current_time = None
        self._grove = None
        self._branch_order = []
        self.binned_detectors = {}
        self._result_maps.reset()
        self._simulation_context = None

    def _get_states(self, basis: List[Qobj]):
        return [self.initial_state] if basis is None else basis  # a set of one or more initial states to propagate

    def _initialize_grove(self, initial_time: float, parameters: dict = None,
                          bin_list: list = None, basis: List[Qobj] = None):
        branches, binned_detectors = self._measurement_branches(parameters, bin_list)
        branch_times = sorted([branch.start_time for branch in branches])

        self._current_time, grove, branch_order = initialise_or_resume_grove(
            current_time=self._current_time,
            grove=self._grove,
            branch_order=self._branch_order,
            initial_time=initial_time,
            basis_states=self._get_states(basis),
            branches=branches,
        )
        return branch_times, branches, branch_order, binned_detectors, grove

    def _simulate_grove(self,
                        parameters: dict = None,
                        bin_list: list = None,
                        basis: List[Qobj] = None,
                        options: dict = None,
                        continue_simulation: bool = False):
        parameters = self.component.set_parameters(parameters)
        simulation_context = build_simulation_context(parameters=parameters, bin_list=bin_list, basis=basis)

        times = self.component.times(parameters)  # determine simulation stop times
        initial_time = self._get_initial_time(times)
        final_time = self._get_final_time(times)

        self._check_if_continue(continue_simulation, simulation_context=simulation_context)
        branch_times, branches, branch_order, binned_detectors, grove = \
            self._initialize_grove(initial_time, parameters, bin_list, basis)
        times = build_propagation_times(current_time=self._current_time, component_times=times, final_time=final_time)

        propagator_factory = Generator(self.component, binned_detectors=binned_detectors, precision=self.precision)

        # Main propagation algorithm
        for i in range(1, len(times)):  # Propagate from initial time to final time
            t0 = times[i - 1]  # current time
            t1 = times[i]  # next stop time
            step = classify_propagation_step(self.component, t0, parameters=parameters, branch_times=branch_times)

            if step.has_instantaneous_operation:
                grove.apply_operator(self.component.evaluate_dirac(t0, parameters))

            if step.opens_measurement_branch:
                branch_order += grove.add_branches(t0, branches)

            # build propagator
            propagator = propagator_factory.build_propagator(t0, parameters=parameters, options=options)

            # apply propagator to all trees in the grove
            grove.propagate(propagator, t1)  # propagate to next stop time

        self._current_time = final_time
        self._grove = grove
        self._branch_order = branch_order

    def generating_points(self, parameters: dict = None,
                          basis: List[Qobj] = None, options: dict = None):
        self._simulate_grove(parameters=parameters, basis=basis, options=options)
        return [tree.get_points() for tree in self._grove]

    def generating_states(self, parameters: dict = None,
                          basis: List[Qobj] = None, options: dict = None):
        self._simulate_grove(parameters=parameters, basis=basis, options=options)
        return [tree.get_states() for tree in self._grove]

    def generating_channels(self, parameters: dict = None,
                            basis: List[Qobj] = None, options: dict = None):
        self._simulate_grove(parameters=parameters, basis=basis, options=options)
        return list(map(list, zip(*[tree.get_states() for tree in self._grove])))

    @staticmethod
    def _normalise_channel_basis_if_needed(point_rank: int, basis: List[Qobj] = None):
        return prepare_channel_basis(basis) if point_rank == 2 else basis

    def _validate_simulation_request(self, point_rank: int):
        if not self.component.is_emitter:
            raise ValueError("At least one component must be a quantum emitter.")
        if not any(port.is_monitored for port in self.component.output.ports):
            raise ValueError("Processor must contain at least one detector.")

    def _extract_tensor_results(self, point_rank: int, dims: List[int] = None, select: List[int] = None,
                                basis: List[Qobj] = None):
        tensors = self._grove.build_tensors(point_rank, self.precision)
        return self._result_maps.ingest(point_rank=point_rank, tensors=tensors, dims=dims, select=select, basis=basis)

    def simulate(self,
                 parameters: dict = None,
                 point_rank: int = 0,
                 bin_list: list = None,
                 dims: List[int] = None,
                 select: List[int] = None,
                 basis: list[Qobj] = None,
                 options: dict = None,
                 reset: bool = True):
        """
        :param parameters: optional parameters to modify the default parameters.
        :param point_rank: simulation rank (0 = probabilities, 1 = states, 2 = channels).
        :param bin_list: a list of integers or strings specifying which measurement bins to simulate.
        :param dims: a list of integers specifying the desired subspace dimensions of the channel.
        :param select: a list of integers specifying which subspace dimensions to trace out.
        :param basis: the orthonormal basis of initial states for channels (if rank = 2).
        :param options: options for qutip mesolve.
        :param reset: whether to continue simulation from the current time or reset from the beginning
        """
        self._validate_simulation_request(point_rank=point_rank)
        basis = self._normalise_channel_basis_if_needed(point_rank=point_rank, basis=basis)

        # simulate the virtual tree
        self._simulate_grove(parameters=parameters, bin_list=bin_list, basis=basis, options=options,
                             continue_simulation=not reset)

        self._extract_tensor_results(point_rank=point_rank, dims=dims, select=select, basis=basis)

    def _order_bins(self, distribution: dict):
        return {tuple(k[i] for i in self._branch_order): v for k, v in distribution.items()}

    def probs(self, parameters: dict = None, bin_list: list = None, options: dict = None, reset: bool = True):
        self.simulate(parameters=parameters, point_rank=0, bin_list=bin_list, options=options, reset=reset)
        return self._order_bins(self._probabilities)

    def conditional_states(self, parameters: dict = None, bin_list: list = None, dims: List[int] = None,
                           select: List[int] = None, options: dict = None, reset: bool = True):
        self.simulate(parameters=parameters, point_rank=1, bin_list=bin_list,
                      dims=dims, select=select, options=options, reset=reset)
        return self._order_bins(self._states)

    def conditional_channels(self, parameters: dict = None, bin_list: list = None,
                             dims: List[int] = None, select: List[int] = None, basis: List[Qobj] = None,
                             options: dict = None, reset: bool = True):
        self.simulate(parameters=parameters, point_rank=2, bin_list=bin_list,
                      dims=dims, select=select, basis=basis, options=options, reset=reset)
        return self._order_bins(self._channels)
