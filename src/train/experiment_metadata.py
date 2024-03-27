from dataclasses import dataclass


@dataclass
class ExperimentMetadata:
    algorithm: str
    env: str
    name: str
    seed: int
    config: any
