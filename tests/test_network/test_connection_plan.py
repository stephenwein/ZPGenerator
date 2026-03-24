from zpgenerator.network.component import Component
from zpgenerator.system import ScattererBase
from zpgenerator.time import Operator


def test_connection_plan_records_attach_and_preserve_actions():
    comp = Component(masked=False)
    comp.add(ScattererBase(Operator([[1, 2], [3, 4]])))

    element = ScattererBase(Operator([[5, 6], [7, 8]]))
    comp._next_pos = 1
    plan = comp._plan_connection(element)

    assert plan.new_mode_number == 3
    assert plan.permutation == (2, 0, 1)
    assert tuple(action.kind for action in plan.existing_actions) == ('preserve', 'attach')
    assert tuple(action.component_port_index for action in plan.existing_actions) == (0, 1)
    assert tuple(action.element_port_index for action in plan.existing_actions) == (None, 0)
    assert plan.padding_targets == ()
    assert plan.remaining_element_port_indices == (1,)


def test_connection_plan_records_padding_and_remaining_ports():
    comp = Component(masked=False)
    comp.add(ScattererBase(Operator([[1, 2], [3, 4]])))
    comp.output.ports[1].close()

    element = ScattererBase(Operator([[5, 6], [7, 8]]))
    comp._next_pos = 2
    plan = comp._plan_connection(element)

    assert plan.new_mode_number == 5
    assert plan.permutation == (2, 3, 4, 0, 1)
    assert tuple(action.kind for action in plan.existing_actions) == ('preserve', 'preserve')
    assert plan.padding_targets == (4,)
    assert plan.remaining_element_port_indices == (0, 1)
