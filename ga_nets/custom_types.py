from enum import Enum, IntEnum, auto


class BaseType(IntEnum):
    """Valori auto-generati a partire da 0."""

    def _generate_next_value_(name, start, count, last_values):
        return count  # assegna 0, 1, 2...


class CategoryType(Enum):
    NEURON = "neuron"
    LAYER = "layer"
    NETWORK = "network"
    SYNAPSE = "synapse"
    GATE = "gate"


class NeuronType(BaseType):
    """Tipo di neurone"""

    INPUT = auto()
    OUTPUT = auto()
    HIDDEN = auto()


class ConnDirection(BaseType):
    IN = auto()
    OUT = auto()


SynapseDirection = ConnDirection
GateDirection = ConnDirection
