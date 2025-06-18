from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ga_nets.custom_types import CategoryType, SynapseDirection, GateDirection
from ga_nets import errors

if TYPE_CHECKING:
    from neuron import Neuron


class ConnectionFactoryBase:
    @classmethod
    def connect(cls, from_neuron, to_neuron, weight):
        if cls.is_already_connected(from_neuron, to_neuron):
            raise errors.NeuronAlreadyConn(
                cls.call_conn_type(),
                from_neuron,
                to_neuron
            )
        connection = cls.create_connection(from_neuron, to_neuron, weight)
        cls.register_connection(from_neuron, to_neuron, connection)
        return connection

    @staticmethod
    def is_already_connected(from_neuron, to_neuron):
        raise NotImplementedError

    @staticmethod
    def create_connection(from_neuron, to_neuron, weight):
        raise NotImplementedError

    @staticmethod
    def register_connection(from_neuron, to_neuron, connection):
        raise NotImplementedError

    @staticmethod
    def call_conn_type():
        raise NotImplementedError


class SynapseFactory(ConnectionFactoryBase):
    @staticmethod
    def is_already_connected(from_neuron, to_neuron):
        return from_neuron.is_projecting_to(to_neuron)

    @staticmethod
    def create_connection(from_neuron, to_neuron, weight):
        return Synapse(from_neuron, to_neuron, weight)

    @staticmethod
    def register_connection(from_neuron, to_neuron, connection):
        from_neuron.add_out_synapse(connection)
        to_neuron.add_in_synapse(connection)

    @staticmethod
    def call_conn_type():
        return CategoryType.SYNAPSE

class GateFactory(ConnectionFactoryBase):
    @staticmethod
    def is_already_connected(from_neuron, to_neuron):
        return from_neuron.is_projecting_to(to_neuron, gate=True)

    @staticmethod
    def create_connection(from_neuron, to_neuron, weight):
        return Gate(from_neuron, to_neuron, weight)

    @staticmethod
    def register_connection(from_neuron, to_neuron, connection):
        from_neuron.add_out_gate(connection)
        to_neuron.add_in_gate(connection)

    @staticmethod
    def call_conn_type():
        return CategoryType.GATE


class Connection(ABC):
    """Classe astratta per sinapsi e gates."""

    def __init__(
        self, from_neuron: "Neuron", to_neuron: "Neuron", weight: float
    ):
        self._from_neuron = from_neuron
        self._to_neuron = to_neuron
        self._weight = weight

    @staticmethod
    def is_connected(
        conns: list["Connection"], from_neuron: "Neuron", to_neuron: "Neuron"
    ) -> bool:
        """Verifica se c'è già una connessione (Sinapsi o Gates)."""
        return any(
            conn.from_neuron == from_neuron and conn.to_neuron == to_neuron
            for conn in conns
        )

    @property
    def from_neuron(self):
        return self._from_neuron

    @property
    def to_neuron(self):
        return self._to_neuron

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @abstractmethod
    def remove(self):
        raise NotImplementedError("Needs to be implemented.")

    def __str__(self):
        from_key = self.from_neuron.key
        to_key = self.to_neuron.key
        weight = self.weight
        return f"From {from_key} to {to_key} (w: {weight})"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"from_neuron={self.from_neuron.key!r}, "
            f"to_neuron={self.to_neuron.key!r}, "
            f"weight={self.weight!r})"
        )


class Synapse(Connection):
    """Sinapsi fra i neuroni del network."""

    def remove(self):
        self.from_neuron.synapses[SynapseDirection.OUT.value].remove(self)
        self.to_neuron.synapses[SynapseDirection.IN.value].remove(self)


class Gate(Connection):
    """Gates fra i neuroni del network."""

    def remove(self):
        self.from_neuron.gates[GateDirection.OUT.value].remove(self)
        self.to_neuron.gates[GateDirection.IN.value].remove(self)

