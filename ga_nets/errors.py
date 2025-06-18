from ga_nets.custom_types import NeuronType


class InvalidCategory(Exception):
    """Raised when a category doesn't exist."""


class InvalidNeuronType(TypeError):
    """Utilizzato quando il tipo del neurone non è valido"""


class NeuronAlreadyConn(Exception):
    def __init__(self, conn_type=None, from_neuron=None, to_neuron=None):
        if conn_type not in (CategoryType.SYNAPSE, CategoryType.GATE):
            raise ValueError(f"Invalid connection type: {conn_type}.")

        msg = (
            f"{conn_type.value.title()} already present "
            f"from {from_neuron!r} to {to_neuron!r}."
        )
        super().__init__(msg)


class NeuronAlreadyExistsError(ValueError):
    def __init__(self, key):
        msg = f"Il neurone '{key!r}' è già presente."
        super().__init__(msg)


class NeuronNotYetExistsError(ValueError):
    def __init__(self, key):
        msg = f"Il neurone '{key!r}' non è ancora presente."
        super().__init__(msg)


class FeatureMismatchError(ValueError):
    def __init__(self, req_features: int, curr_features: int):
        msg = (
            f"Network requires {req_features} features, "
            f"{curr_features} provided."
        )
        super().__init__(msg)


def validate_neuron_type(neuron_type):
    if neuron_type not in NeuronType:
        raise InvalidNeuronType(f"Invalid neuron type: {neuron_type}")
