from qutip import Qobj, basis

from zpgenerator.simulate._processor_helpers import prepare_channel_basis
from zpgenerator.simulate._processor_results import ProcessorResultMaps


class _DummyTensor:
    def __init__(self, invert_value, results):
        self._invert_value = invert_value
        self._results = results

    def invert(self):
        return self._invert_value

    def extract_results(self, dims=None, select=None):
        return self._results


def test_result_maps_ingest_rank0_updates_probabilities_and_flag():
    maps = ProcessorResultMaps()
    tensors = [
        _DummyTensor(False, {(0,): 0.6, (1,): 0.4}),
        _DummyTensor(True, {(0,): 0.6, (1,): 0.4}),
    ]
    maps.ingest(point_rank=0, tensors=tensors)

    assert maps.contains_unnormalised_detector is True
    assert maps.probabilities[(0,)] == 0.6
    assert maps.probabilities[(1,)] == 0.4


def test_result_maps_ingest_rank1_updates_states_and_probabilities():
    maps = ProcessorResultMaps()
    rho = Qobj([[0.75, 0], [0, 0.25]])
    maps.ingest(point_rank=1, tensors=[_DummyTensor(False, {(0,): rho})])

    assert (0,) in maps.states
    assert maps.probabilities[(0,)] == rho.tr()


def test_result_maps_ingest_rank2_updates_channels_without_mutating_source():
    maps = ProcessorResultMaps()
    source_state = Qobj([[1, 0], [0, 0]], dims=[[2], [2]])
    basis_states = prepare_channel_basis([basis(2, 0), basis(2, 1)])
    tensors = [_DummyTensor(False, {(0,): source_state}) for _ in range(4)]

    maps.ingest(point_rank=2, tensors=tensors, basis=basis_states, dims=[1, 2])

    assert source_state.dims == [[2], [2]]
    assert (0,) in maps.channels
    assert maps.channels[(0,)].shape == (4, 4)
