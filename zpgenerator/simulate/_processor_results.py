from dataclasses import dataclass, field

from ._processor_helpers import build_channels_from_results, invert_tensors


@dataclass
class ProcessorResultMaps:
    probabilities: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)
    channels: dict = field(default_factory=dict)
    contains_unnormalised_detector: bool = False

    def reset(self):
        self.probabilities.clear()
        self.states.clear()
        self.channels.clear()
        self.contains_unnormalised_detector = False

    def ingest(self, point_rank: int, tensors, dims=None, select=None, basis=None):
        self.contains_unnormalised_detector = invert_tensors(tensors)
        results = [tensor.extract_results(dims=dims, select=select) for tensor in tensors]

        if point_rank == 0:
            self.probabilities.update(results[0])
            return results

        if point_rank == 1:
            self.states.update(results[0])
            self.probabilities.update({k: v.tr() for k, v in results[0].items()})
            return results

        if point_rank == 2:
            self.channels.update(build_channels_from_results(results=results, basis=basis, dims=dims, select=select))
            return results

        raise NotImplementedError(f"Unsupported point rank: {point_rank}")
