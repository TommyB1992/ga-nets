"""Architettura portante della rete neurale"""

from abc import ABC

from ga_nets.connection import SynapseFactory
from ga_nets.custom_types import CategoryType, NeuronType, SynapseDirection
from ga_nets import errors
from ga_nets.indexer import Indexer
import ga_nets.layer as Layer
from ga_nets.neuron import Neuron


class Network(ABC):
    """La classe che si occupa di gestire la struttura generale della rete
    neurale"""

    def __init__(self) -> None:
        """Inizializza la rete neurale che conterrà neuroni
        (nodi), sinpasi (connessioni) e layers neuronali."""
        self._key = Indexer.get_id(CategoryType.NETWORK)

        self._neurons = {}
        self._synapses = []
        self._layers = []

    @property
    def neurons(self):
        """Getter per la lista di neuroni.

        Returns:
          L'array con i neuroni.
        """
        return self._neurons

    @property
    def synapses(self):
        """Getter per la prorietà delle connessioni.

        Returns:
          L'array con le connessioni.
        """
        return self._synapses

    @property
    def layers(self):
        """Getter per la prorietà dei layers.

        Returns:
          La lista tridimensionale con i neuroni in ogni layer.
        """
        if not self._layers:
            self._layers = self.get_layers()

        return self._layers

    @property
    def num_inputs(self) -> int:
        """Numero di neuroni di input nel network."""
        return self._count_neurons(NeuronType.INPUT)

    @property
    def num_outputs(self) -> int:
        """Numero di neuroni di output nel network."""
        return self._count_neurons(NeuronType.OUTPUT)

    @property
    def num_hiddens(self) -> int:
        """Numero di neuroni nascosti nel network."""
        return self._count_neurons(NeuronType.HIDDEN)

    def _count_neurons(self, neuron_type) -> int:
        """Conta i neuroni di un certo tipo."""
        return sum(1 for _ in self._neurons_by_type(neuron_type))

    def _neurons_by_type(self, neuron_type):
        """Generatore dei neuroni filtrati per tipo."""
        errors.validate_neuron_type(neuron_type)
        return (n for n in self._neurons.values() if n.type is neuron_type)

    def add_neuron(self, neuron: Neuron):
        key = neuron.key
        if key in self._neurons:
            raise NeuronAlreadyExistsError(key)

        self._neurons[key] = neuron

    def sub_neuron(self, neuron: Neuron):
        """Rimuove il neurone dal network.

        Args:
          neuron: Istanza del neurone da rimuovere.
        """
        key = neuron.key
        if key not in self._neurons:
            raise NeuronNotYetExistsError(key)
        del self._neurons[key]

        # Rimuove le connessioni
        for direction in SynapseDirection:
            for synapse in neuron.synapses[direction.value]:
                self._synapses.remove(synapse)

    def add_synapse(
        self, from_neuron: Neuron, to_neuron: Neuron, weight: float
    ):
        """Connette due neuroni tramite sinapsi.

        Args:
          from_neuron: Istanza del neurone di partenza.
          to_neuron: Istanza del neurone d'arrivo.
          weight: Peso della connessione.
        Returns:
          L'istanza della connessione fra i due neuroni.

        Raises:
          AlreadyConn: Se la connessione è già presente.
        """
        synapse = SynapseFactory.connect(from_neuron, to_neuron, weight)
        self._synapses.append(synapse)

    def sub_synapse(self, synapse):
        """Rimuove una sinapsi fra i due neuroni.

        Args:
          synapse: Istanza della connessione.
        """
        self._synapses.remove(synapse)
        synapse.remove()

    def get_layers(self):
        """
        Costruisce e restituisce i layer (livelli) con l'ordine
        corretto di attivazione.

        Returns:
            Una lista di liste, dove ogni sottolista contiene i key dei
            neuroni da attivare in parallelo, nell'ordine corretto. Es:

            Se la rete è composta da 3 neuroni tutti connessi:
            #0 Input
            #1 Output
            #2 Hidden

            Il risultato potrebbe essere: [[0], [2], [1]]
        """
        inputs = self._get_keys_by_type(NeuronType.INPUT)
        hiddens = self._get_keys_by_type(NeuronType.HIDDEN)
        outputs = self._get_keys_by_type(NeuronType.OUTPUT)

        synapses = self._get_synapses()
        links = Layer.get_links(inputs, hiddens, outputs, synapses)

        return Layer.get_layers(inputs, hiddens, outputs, synapses, links)

    def _get_keys_by_type(self, neuron_type):
        """Restituisce i key dei neuroni di un certo tipo."""
        return [n.key for n in self._neurons_by_type(neuron_type)]

    def _get_synapses(self):
        """Connessioni della rete neurale.

        Args:
          network: Istanza della rete neurale.

        Returns:
          Una lista con le tuple contenente come primo indice il neurone
          d'uscita e il secondo indice quello di entrata, es: '[(0, 3),
                                                                (1, 3),
                                                                (3, 2)]'.
        """
        return [
            (s.from_neuron.key, s.to_neuron.key, s.weight)
            for s in self._synapses
        ]

    def activate(self, features):
        """Attiva la rete neurale.

        Args:
          features: Una matrice con gli input: [1, 1, ...]

        Returns:
          Una matrice tridimensionali con i valori di output (es. '[[1]]' o
          '[[1], [1]]').

        Raises:
          ValueError: Se la dimensione dell'array degli input è errata.
        """
        num_features = len(features)
        if num_features != self.num_inputs:
            raise errors.FeatureMismatchError(self.num_inputs, num_features)

        # Resetta gli stati
        self._clear_states()

        # Costruisce i layer di neuroni
        neurons = [self._neurons[n] for layer in self.layers for n in layer]

        # Attiva i neuroni e restituisce gli output in ordine
        for i, neuron in enumerate(neurons):
            if neuron.type is NeuronType.INPUT:
                neuron.state.append(features[i])
            else:
                neuron.activate()

        outputs = [self._neurons[o].state[-1] for o in self.layers[-1]]

        return outputs

    def _clear_states(self):
        """Resetta gli stati dei neuroni"""
        for neuron in self._neurons.values():
            neuron.clear_state()

    def __str__(self):
        def indent(text, prefix="    "):
            return "\n".join(prefix + line for line in text.splitlines())

        neurons = "\n".join(indent(str(n)) for n in self._neurons.values())
        return f"# Network {self._key}\n" f"  Neurons:\n{neurons}"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"key={self._key!r}, "
            f"input={self.num_inputs}, "
            f"hidden={self.num_hiddens}, "
            f"output={self.num_outputs}, "
            f"synapses={len(self._synapses)}, "
            f"layers={self.layers})"
        )
