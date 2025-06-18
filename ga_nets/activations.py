# ga_nets/activations.py
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

"""
Activation functions for neural networks.

This module defines a collection of commonly used activation functions
such as sigmoid, tanh, ReLU, GELU, and others. These functions are
used to introduce non-linearity into neural network models, enabling
them to learn complex patterns.

A dictionary named `activations` is also provided to access the
functions by name as strings.

Functions:
    - linear(x)
    - sigmoid(x)
    - tanh(x)
    - arctan(x)
    - softsign(x)
    - hard_sigmoid(x)
    - gompertz(x)
    - relu(x)
    - leaky_relu(x, alpha=0.01)
    - elu(x, alpha=1.0)
    - swish(x, beta=1.0)
    - gelu(x)
"""
import math


def linear(x: float) -> float:
    """Linear activation."""
    return x


def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-x))


def tanh(x: float) -> float:
    """Hyperbolic tangent activation function."""
    return math.tanh(x)


def arctan(x: float) -> float:
    """Arctangent activation function."""
    return math.atan(x)


def softsign(x: float) -> float:
    """Softsign activation function."""
    return x / (1 + abs(x))


def hard_sigmoid(x: float) -> float:
    """Hard sigmoid activation: linear approx clipped to [0,1]."""
    return max(0.0, min(1.0, 0.2 * x + 0.5))


def gompertz(x: float) -> float:
    """Gompertz function."""
    return math.exp(-math.exp(-x))


def relu(x: float) -> float:
    """ReLU activation function."""
    return max(0.0, x)


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Leaky ReLU activation function."""
    return x if x > 0 else alpha * x


def elu(x: float, alpha: float = 1.0) -> float:
    """Exponential Linear Unit (ELU) activation."""
    return x if x > 0 else alpha * (math.exp(x) - 1)


def swish(x: float, beta: float = 1.0) -> float:
    """Swish activation function."""
    return x * sigmoid(beta * x)


def gelu(x: float) -> float:
    """Gaussian Error Linear Unit (approximation)."""
    return (
        0.5
        * x
        * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    )


# dictionary to access by name
activations = {
    "linear": linear,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "arctan": arctan,
    "softsign": softsign,
    "hard_sigmoid": hard_sigmoid,
    "gompertz": gompertz,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "swish": swish,
    "gelu": gelu,
}
