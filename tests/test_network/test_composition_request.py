import pytest

from zpgenerator.network.component import Component
from zpgenerator.network.detector import DetectorGate
from zpgenerator.elements.linear import BeamSplitter


def test_build_composition_request_returns_none_for_missing_element():
    with pytest.raises(ValueError, match="Please specify an element to add"):
        Component._build_composition_request(0, None)


def test_normalize_composition_request_resolves_named_port_and_preserves_detector():
    comp = Component()
    comp.add(BeamSplitter())
    comp.output.ports[0].port_name = 'left'

    request = Component._build_composition_request('left', DetectorGate(), bin_name='D')
    normalized = comp._normalize_composition_request(request)

    assert normalized.position == 0
    assert isinstance(normalized.element, DetectorGate)
    assert normalized.bin_name == 'D'


def test_normalize_composition_request_bins_nested_component_without_mutating_original():
    parent = Component()
    parent.add(BeamSplitter())

    child = Component(masked=False)
    child.add(BeamSplitter())
    child.add(0, DetectorGate())
    assert child.output.bins == 1
    assert list(child.output.binned_detectors.keys()) == ['_bin']

    request = Component._build_composition_request(0, child, name='child-copy', bin_name='B')
    normalized = parent._normalize_composition_request(request)

    assert normalized.position == 0
    assert normalized.element is not child
    assert normalized.element.output.bins == 1
    assert list(normalized.element.output.binned_detectors.keys()) == ['B']
    assert child.output.bins == 1
    assert list(child.output.binned_detectors.keys()) == ['_bin']
    assert normalized.parameters is None
    assert normalized.name is None
    assert normalized.element.name == 'child-copy'
