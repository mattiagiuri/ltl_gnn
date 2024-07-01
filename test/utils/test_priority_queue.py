import pytest
from utils import PriorityQueue


def test_push_and_pop():
    pq = PriorityQueue()
    pq.push('task1', 3)
    pq.push('task2', 1)
    pq.push('task3', 2)

    assert pq.pop() == ('task2', 1)
    assert pq.pop() == ('task3', 2)
    assert pq.pop() == ('task1', 3)
    with pytest.raises(IndexError):
        pq.pop()


def test_change_priority():
    pq = PriorityQueue()
    pq.push('task1', 3)
    pq.push('task2', 1)
    pq.push('task3', 2)

    pq.change_priority('task1', 0)

    assert pq.pop() == ('task1', 0)
    assert pq.pop() == ('task2', 1)
    assert pq.pop() == ('task3', 2)
    with pytest.raises(IndexError):
        pq.pop()


def test_is_empty():
    pq = PriorityQueue()
    assert pq.is_empty() is True
    pq.push('task1', 3)
    assert pq.is_empty() is False
    pq.pop()
    assert pq.is_empty() is True


def test_len():
    pq = PriorityQueue()
    assert len(pq) == 0
    pq.push('task1', 3)
    pq.push('task2', 1)
    assert len(pq) == 2
    pq.pop()
    assert len(pq) == 1
    pq.pop()
    assert len(pq) == 0


def test_contains():
    pq = PriorityQueue()
    pq.push('task1', 3)
    pq.push('task2', 1)
    assert 'task1' in pq
    assert 'task2' in pq
    pq.pop()
    assert 'task2' not in pq
    pq.pop()
    assert 'task1' not in pq


def test_change_priority_non_existent():
    pq = PriorityQueue()
    pq.push('task1', 3)
    with pytest.raises(ValueError):
        pq.change_priority('task2', 1)


def test_getitem():
    pq = PriorityQueue()
    pq.push('task1', 3)
    pq.push('task2', 1)
    pq.push('task3', 2)

    assert pq['task1'] == 3
    assert pq['task2'] == 1
    assert pq['task3'] == 2

    pq.change_priority('task1', 0)
    assert pq['task1'] == 0

    with pytest.raises(KeyError):
        pq['task4']  # this item does not exist in the priority queue


if __name__ == "__main__":
    pytest.main()
