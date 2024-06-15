import random
from pprint import pprint
from typing import Callable


class FixedSequenceSampler:  # TODO: rewrite using assignments instead of simply propositions

    @classmethod
    def partial(cls, seq: list[tuple[str, str]]) -> Callable[[list[str]], 'FixedSequenceSampler']:
        return lambda propositions: cls(propositions, seq)

    def __init__(self, propositions: list[str], seq: list[tuple[str, str]]):
        self.seq = seq

    def __call__(self) -> list[tuple[str, str]]:
        return self.seq
