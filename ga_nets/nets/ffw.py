# ffw.py
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

"""Feedforward neural network implementation.

This module defines the FeedForward class which is a wrapper around a
basic Network for standard feedforward (acyclic) topologies. It
provides standard behavior suitable for evolutionary or manually
defined networks.
"""
from ga_nets.custom_types import SynapseDirection
from ga_nets.neuron import Neuron
from ga_nets.network import Network


class FeedForwardNet(Network):
    """Classico wrapper per il feedforward network"""


class FFWNeuron(Neuron):
    """Wrapper per i neuroni nei feedforward networks"""

    def activate(self):
        states = [
            s.from_neuron.state[0] * s.weight
            for s in self.synapses[SynapseDirection.IN.value]
            if s.from_neuron.state
        ]

        # Per evitare errori dati dal min() e max()
        if not states:
            states.append(0)

        self._state.append(
            self.activation(self.aggregation(states) + self.bias)
        )
