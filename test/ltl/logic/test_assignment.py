import pytest
import sympy

from ltl.logic.assignment import Assignment


@pytest.fixture
def assignment():
    return Assignment(
        a=True,
        b=False,
        c=True
    )


def test_assignment(assignment):
    assert assignment['a']
    assert not assignment['b']
    assert assignment['c']


def test_all_possible_assignments():
    propositions = ('a', 'b')
    assignments = Assignment.all_possible_assignments(propositions)
    assert assignments == [
        Assignment(a=False, b=False),
        Assignment(a=False, b=True),
        Assignment(a=True, b=False),
        Assignment(a=True, b=True)
    ]
    propositions = ('a', 'b', 'c', 'd', 'e')
    assignments = Assignment.all_possible_assignments(propositions)
    assert len(assignments) == 2 ** len(propositions)
    for assignment in assignments:
        assert len(assignment) == len(propositions)
        for p in propositions:
            assert assignment[p] in (True, False)


def test_assignment_satisfies(assignment):
    a, b, c = sympy.symbols('a b c')
    assert assignment.satisfies(a)
    assert not assignment.satisfies(b)
    assert assignment.satisfies(c)
    assert assignment.satisfies(a & c)
    assert not assignment.satisfies(a & b)
    assert not assignment.satisfies(b & c)
    assert assignment.satisfies(a | b)


def test_to_frozen(assignment):
    frozen = assignment.to_frozen()
    assert frozen.assignment == frozenset([('a', True), ('b', False), ('c', True)])


def test_to_label(assignment):
    assert assignment.to_frozen().to_label() == 'a & !b & c'
