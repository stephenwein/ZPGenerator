from zpgenerator.system.emitter import *
from test_control import _make_controlled_system
from qutip import destroy, num, create
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
