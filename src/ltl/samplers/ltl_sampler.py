from abc import ABC, abstractmethod


class LTLSampler(ABC):

    def __init__(self, propositions: list[str]):
        self.propositions = propositions

    @abstractmethod
    def sample(self) -> str:
        pass
