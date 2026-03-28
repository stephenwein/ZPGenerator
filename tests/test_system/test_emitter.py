from zpgenerator.system.emitter import *
from zpgenerator.time import TimeOperator
from zpgenerator.dynamic import Pulse
from test_control import _make_controlled_system
from qutip import destroy, num, create, Qobj
from zpgenerator.time.parameters import Parameters
from tests_assertions import assert_empty_qobj
import pytest


d = Parameters.DELIMITER

def test_lindblad_vector():
    vec = LindbladVector()
    assert vec.modes == 0
    vec.add(destroy(2))
    assert vec.modes == 1
    vec.add(destroy(2))
    assert vec.modes == 2


def test_emitter_base_init_empty():
    emitter = EmitterBase()
    assert emitter.dim is None
    assert emitter.subdims is None
    assert emitter.parameters == []
    assert emitter.states == {}
    assert emitter.operators == {}
    assert_empty_qobj(emitter.evaluate(0))
    assert_empty_qobj(emitter.transitions.evaluate(0))
    assert emitter.modes == 0


def test_emitter_base_init():
    system = _make_controlled_system()
    emitter = EmitterBase(hamiltonian=system.hamiltonian,
                          environment=system.environment,
                          control=system.control,
                          transitions=[destroy(2), destroy(2)],
                          states=system.states,
                          operators=system.operators)

    assert emitter.modes == 2
    assert emitter.dim == 2
    assert emitter.subdims == [2]
    assert emitter.parameters == ['decay' + d + 'rate',
                                  'dephasing' + d + 'rate',
                                  'detuning',
                                  'pulse' + d + 'area',
                                  'pulse' + d + 'delay',
                                  'pulse' + d + 'flip' + d + 'time',
                                  'pulse' + d + 'phonon_coefficient',
                                  'pulse' + d + 'width']
    assert emitter.states == {}
    assert emitter.operators == {}
    assert emitter.evaluate(0) == system.evaluate(0)
    assert_empty_qobj(emitter.transitions.evaluate(0)) # transitions do not have a Liouvillian
    assert emitter.evaluate_quadruple(0).transitions[0].constant == destroy(2)


def test_emitter_add():
    emitter = EmitterBase()
    env = EnvironmentBase()
    env.add(lambda args: args['a'] * destroy(2), parameters={'a': 1})
    emitter.add(env)

    # adding an EnvironmentBase directly to emitter nests it within the current environment
    assert emitter.parameter_tree() == {'_environment': {'_EnvironmentBase': {'_TimeOperator': {'_operator': {'a': 1}}}}}

    emitter = EmitterBase()
    emitter.environment.add(lambda args: args['a'] * destroy(2), parameters={'a': 1})

    # adding to the current emitter environment modifies the current environment
    assert emitter.parameter_tree() == {'_environment': {'_TimeOperator': {'_operator': {'a': 1}}}}

    # the same behaviour holds for HamiltonianBase, LindbladVector, and ControlBase
    emitter = EmitterBase()
    ham = HamiltonianBase()
    env = EnvironmentBase()
    trn = LindbladVector()
    con = ControlBase()
    ham.add(lambda args: args['a'] * num(2), parameters={'a': 0})
    env.add(lambda args: args['b'] * destroy(2), parameters={'b': 1})
    trn.add(lambda args: args['c'] * destroy(2), parameters={'c': 1})
    con.hamiltonian.add(lambda args: args['d'] * (destroy(2) + create(2)), parameters={'d': 2})

    emitter.add([ham, env, trn, con])
    assert emitter.parameter_tree() == {
        '_control': {'_ControlBase': {'_hamiltonian': {'_TimeOperator': {'_operator': {'d': 2}}}}},
        '_environment': {'_EnvironmentBase': {'_TimeOperator': {'_operator': {'b': 1}}}},
        '_hamiltonian': {'_HamiltonianBase': {'_TimeOperator': {'_operator': {'a': 0}}}},
        '_transitions': {'_LindbladVector': {'_TimeOperator': {'_operator': {'c': 1}}}}}


def test_emitter_set_system_preserves_behaviour():
    system = _make_controlled_system()
    transitions = LindbladVector([destroy(2), destroy(2)])
    emitter = EmitterBase()
    emitter.set_system(system=system, transitions=transitions)

    assert emitter.evaluate(0) == system.evaluate(0)
    assert emitter.evaluate_quadruple(0).transitions[0].constant == destroy(2)


def test_emitter_set_system_preserves_subclass_state():
    class TaggedEmitter(EmitterBase):
        def __init__(self):
            super().__init__()
            self.tag = "keep-me"

    system = _make_controlled_system()
    emitter = TaggedEmitter()
    emitter.set_system(system=system, transitions=LindbladVector([destroy(2)]))

    assert emitter.tag == "keep-me"
    assert emitter.system is system


def test_emitter_rejects_transitions_without_environment():
    with pytest.raises(ValueError, match="Transitions cannot occur without an environment"):
        EmitterBase(transitions=[destroy(2)])


def test_emitter_from_master_equation_duplicates_monitored_channels_into_environment():
    monitored = destroy(2)
    background = create(2)
    states = {'|g>': Qobj([[1], [0]]), '|e>': Qobj([[0], [1]])}
    operators = {'number': num(2)}

    emitter = EmitterBase.from_master_equation(
        hamiltonian=num(2),
        monitored=[monitored],
        environment=[background],
        states=states,
        operators=operators,
        initial_state='|e>',
        initial_time=3,
        name='custom',
    )

    assert emitter.modes == 1
    assert emitter.initial_time == 3
    assert emitter.initial_state == states['|e>']
    assert emitter.states == states
    assert emitter.operators == operators
    assert emitter.evaluate_quadruple(0).hamiltonian.constant == num(2)
    assert emitter.evaluate_quadruple(0).transitions[0].constant == monitored
    assert [env.constant for env in emitter.evaluate_quadruple(0).environment] == [background, monitored]


def test_emitter_from_master_equation_rejects_duplicate_monitored_environment_channels():
    channel = destroy(2)

    with pytest.raises(ValueError, match="same collapse operator"):
        EmitterBase.from_master_equation(
            hamiltonian=num(2),
            monitored=[channel],
            environment=[channel],
        )


def test_emitter_from_master_equation_accepts_time_operator_hamiltonians():
    pulse = Pulse.gaussian()
    hamiltonian = TimeOperator(operator=num(2), functions=pulse)

    emitter = EmitterBase.from_master_equation(
        hamiltonian=HamiltonianBase(hamiltonian),
        monitored=[destroy(2)],
    )

    assert emitter.times()
