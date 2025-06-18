from abc import ABC, abstractmethod
from typing import Self

from ga_nets.activations import linear
from ga_nets.custom_types import CategoryType
from ga_nets.connection import Synapse
from ga_nets.custom_types import NeuronType, SynapseDirection
from ga_nets import errors
from ga_nets.indexer import Indexer
from ga_nets import utils


class Neuron(ABC):
    """Neurone utilizzato nella rete neurale"""

    def __init__(self, **kwargs):
        self._type = kwargs.get("neuron_type", NeuronType.HIDDEN)

        errors.validate_neuron_type(self._type)

        self._key = Indexer.get_id(CategoryType.NEURON)
        self._bias = kwargs.get("bias", 0)
        self._activation = kwargs.get("activation", linear)
        self._aggregation = kwargs.get("aggregation", sum)

        # È un array in quanto utilizzato anche da RNN.
        # Nel caso venga utilizzato da un feedforward
        # network avrà un solo elemento.
        self._state = []

        self._synapses = [[] for _ in SynapseDirection]

    @property
    def key(self: int) -> int:
        """Getter per l'indice del neurone.

        Returns:
          L'intero con l'indice."""
        return self._key

    @key.setter
    def key(self, key):
        self._key = key

    @property
    def type(self):
        """Ha solo il metodo getter in quanto non può variare il tipo.

        Returns:
          Un'istanza dell'oggetto Enum che definisce il tipo:
            - INPUT
            - HIDDEN
            - OUTPUT"""
        return self._type

    @property
    def bias(self):
        """Getter per il bias del neurone.

        Returns:
          Un float con il bias del neurone."""
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def activation(self):
        """Getter per la funzione d'attivazione.

        Returns:
          L'istanza della funzione utilizzata per l'attivazione.
        """
        return self._activation

    @activation.setter
    def activation(self, activation):
        self._activation = activation

    @property
    def aggregation(self):
        """Getter per la funzione d'aggregazione.

        Returns:
          L'istanza della funzione utilizzata per l'aggregazione."""
        return self._aggregation

    @aggregation.setter
    def aggregation(self, aggregation):
        self._aggregation = aggregation

    @property
    def state(self):
        """Getter che restituisce lo stato (o gli stati) del neurone.

        Returns:
          Un'array vuota o con gli interi/reali degli o dello stato
          d'attivazione."""
        return self._state

    def clear_state(self):
        self._state = []

    @property
    def synapses(self):
        """Getter che restituisce le connessioni fra i neuroni.

        Returns:
          Un'array vuota o con le istanze delle connessioni"""
        return self._synapses

    @synapses.setter
    def synapses(self, synapses):
        self._synapses = synapses

    @abstractmethod
    def activate(self):
        """Calcola lo stato del neurone"""
        raise NotImplementedError("Needs to be implemented.")

    def add_out_synapse(self, synapse) -> Synapse:
        """
        Crea una sinapsi da questo neurone verso un altro e aggiorna
        entrambe le liste.

        Args:
            neuron: Il neurone di destinazione.
            weight: Il peso della connessione.

        Returns:
            L'istanza della sinapsi creata.
        """
        self._synapses[SynapseDirection.OUT.value].append(synapse)

    def add_in_synapse(self, synapse) -> Synapse:
        """
        Crea una sinapsi da questo neurone verso un altro e aggiorna
        entrambe le liste.

        Args:
            neuron: Il neurone di destinazione.
            weight: Il peso della connessione.

        Returns:
            L'istanza della sinapsi creata.
        """
        self._synapses[SynapseDirection.IN.value].append(synapse)

    def sub_synapse(self, neuron: Self) -> None:
        """Rimuove la connessione tramite sinapsi da questo neurone verso un
        altro.

        Args:
          neuron: L'istanza del neurone al quale disconnetterlo.
        """
        outgoing = self._synapses[SynapseDirection.OUT.value]
        synapse = next((s for s in outgoing if s.to_neuron is neuron), None)
        if synapse:
            synapse.remove()

    def is_projecting_to(self, neuron: Self) -> bool:
        """Verifica se il neurone ha una connessione in uscita verso un altro.

        Args:
          neuron: L'istanza della classe del neurone in entrata.

        Returns:
          True se la connessione è presente, False altrimenti.
        """
        return any(
            conn.to_neuron == neuron
            for conn in self._synapses[SynapseDirection.OUT.value]
        )

    def copy(self):
        """Restituisce un oggetto identico a questa imitando la funzione per
        copiare, ma in maniera ottimizzata.

        Returns:
          Un'istanza dell'oggetto identico a questo stesso."""
        return self.__class__(
            key=self._key,
            bias=self._bias,
            activation=self._activation,
            aggregation=self._aggregation,
            neuron_type=self._type,
        )

    def __str__(self):

        in_syn = "\n    ".join(
            str(s) for s in self._synapses[SynapseDirection.IN.value]
        )
        out_syn = "\n    ".join(
            str(s) for s in self._synapses[SynapseDirection.OUT.value]
        )

        return (
            f"# Neuron {self._key}\n"
            f"  Type           : {self._type.name}\n"
            f"  Bias           : {self._bias:.4f}\n"
            f"  Activation Fn  : {utils.fn_name_or_repr(self._activation)}\n"
            f"  Aggregation Fn : {utils.fn_name_or_repr(self._aggregation)}\n"
            f"  In Synapses    :\n    {in_syn}\n"
            f"  Out Synapses   :\n    {out_syn}"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"key={self._key!r}, "
            f"type={self._type.name!r}, "
            f"bias={self._bias:.4f}, "
            f"activation={utils.fn_name_or_repr(self._activation)}, "
            f"aggregation={utils.fn_name_or_repr(self._aggregation)}, "
            f"in={len(self._synapses[SynapseDirection.IN.value])}, "
            f"out={len(self._synapses[SynapseDirection.OUT.value])})"
        )
