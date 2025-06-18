# ga_nets/nets/rnn.py
#
# @Author(s):
#   - Tomas Bartoli <tomasbartoli1992@gmail.com>
# @Date: 10/04/2025
# @Since: 1.0.0
#
# Copyright (C) 2025 Tomas Bartoli
# This file is part of the ga_nets project and is distributed under the
# MIT License.
# See the LICENSE file for details.

"""Recurrent neural network (RNN) implementation.

This module defines the Recurrent class, which extends the base Network
to support recurrent connections through the use of gates. These
connections allow the network to maintain internal memory by feeding
back information from previous time steps, enabling temporal sequence
processing.

This implementation assumes that gates act as recurrent links between
neurons, modulating the influence of past activations on current state
updates.
"""
from typing import Self

from ga_nets.connection import GateFactory
from ga_nets.custom_types import GateDirection, SynapseDirection
from ga_nets.neuron import Neuron
from ga_nets.network import Network


class RecurrentNet(Network):
    """Wrapper per il recurrent network"""

    def __init__(self):
        super().__init__()

        self._gates = []

    @property
    def gates(self):
        """
        
        Returns:
          
        """
        return self._gates

    @gates.setter
    def gates(self, gates):
        self._gates = gates

    def add_gate(self, from_neuron, to_neuron, weight):
        gate = GateFactory.connect(from_neuron, to_neuron, weight)
        self._gates.append(gate)

    def sub_gate(self, gate):
        self._gates.remove(gate)
        gate.remove()

    def sub_neuron(self, neuron):
        """Rimuove il neurone dal network e in aggiunta anche i gates.

        Args:
          neuron: Istanza del neurone da rimuovere.
        """
        super().sub_neuron(neuron)

        # Rimuove i gates
        for direction in GateDirection:
            for gate in neuron.gates[direction.value]:
                self._gates.remove(gate)

    def __repr__(self):
        """Aggiunge le informazioni sui gates"""
        base_repr = super().__repr__().rstrip(")")
        return f"{base_repr}, gates={len(self._gates)})"


class RNNNeuron(Neuron):
    """Wrapper per i neuroni nei recurrent networks"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._gates = [[] for _ in GateDirection]

    @property
    def gates(self):
        """Getter dei gates della connessione.

        Returns:
          La lista con i gates.
        """
        return self._gates

    @gates.setter
    def gates(self, gates):
        self._gates = gates

    def activate(self):
        step = len(self.state) - 1

        # Input esterni (sinapsi in ingresso)
        inputs = [
            s.from_neuron.state[-1] * s.weight
            for s in self.synapses[SynapseDirection.IN.value]
            if s.from_neuron.state
        ]

        # Input ricorrenti dai gate (stato precedente dei neuroni
        # collegati tramite gate)
        recurrent = (
            sum(
                g.to_neuron.state[step] * g.weight
                for g in self.gates[GateDirection.OUT.value]
                if g.to_neuron.state
            )
            if step > -1
            else 0
        )

        if not inputs:
            inputs = [0]

        self._state.append(
            self.activation(self.aggregation(inputs) + recurrent + self.bias)
        )

    def add_out_gate(self, gate):
        self._gates[GateDirection.OUT.value].append(gate)

    def add_in_gate(self, gate):
        self._gates[GateDirection.IN.value].append(gate)

    def sub_gate(self, neuron: Self) -> None:
        outgoing = self._gates[GateDirection.OUT.value]
        gate = next((s for s in outgoing if s.to_neuron is neuron), None)
        if gate:
            gate.remove()
