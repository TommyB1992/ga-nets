# ga_nets/aggregations.py
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
Aggregation functions for neural network layers and competitive
selection.

This module provides a wide range of aggregation operations, including
additive, multiplicative, selective
(e.g., max, median), normalization-based
(e.g., softmax, sparsemax), vector norms, logical operations, and
sparse  or competitive functions such as maxout and k-winner-take-all.

An `aggregations` dictionary is included to allow convenient lookup by
name as strings.

Functions:
    - abs_sum(neuron_list)
    - abs_prod(neuron_list)
    - abs_mean(neuron_list)
    - softmax(x)
    - sparsemax(x)
    - log_sum_exp(x)
    - norm_l1(x)
    - norm_l2(x)
    - maxout(x, group_size)
    - threshold(x, thresh=0.0)
    - k_winner_take_all(x, k)
    - logical_and(x)
    - logical_or(x)
    - logical_xor(x)
"""
import math
import statistics
from typing import List
from functools import reduce


def abs_sum(neuron_list):
    return sum(map(abs, neuron_list))


def abs_prod(neuron_list):
    return math.prod(map(abs, neuron_list))


def abs_mean(neuron_list):
    return statistics.mean(map(abs, neuron_list))


def softmax(x: List[float]) -> List[float]:
    max_x = max(x)
    exps = [math.exp(v - max_x) for v in x]
    total = sum(exps)
    return [e / total for e in exps]


def sparsemax(x: List[float]) -> List[float]:
    # Reference: From Martins & Astudillo (2016)
    z = sorted(x, reverse=True)
    cumsum = 0.0
    k = 0
    for i, z_i in enumerate(z):
        cumsum += z_i
        t = (cumsum - 1) / (i + 1)
        if z_i > t:
            k = i + 1
    tau = (sum(z[:k]) - 1) / k
    return [max(v - tau, 0.0) for v in x]


def log_sum_exp(x: List[float]) -> float:
    max_x = max(x)
    return max_x + math.log(sum(math.exp(v - max_x) for v in x))


def norm_l1(x: List[float]) -> float:
    return sum(map(abs, x))


def norm_l2(x: List[float]) -> float:
    return math.sqrt(sum(v * v for v in x))


def maxout(x: List[float], group_size: int) -> List[float]:
    """Applica maxout dividendo l'input in gruppi di dimensione
    group_size."""
    if len(x) % group_size != 0:
        raise ValueError(
            "La lunghezza di x deve essere multiplo di group_size"
        )

    return [max(x[i: i + group_size]) for i in range(0, len(x), group_size)]


def threshold(x: List[float], thresh: float = 0.0) -> List[float]:
    return [v if v >= thresh else 0.0 for v in x]


def k_winner_take_all(x: List[float], k: int) -> List[float]:
    if k >= len(x):
        return x
    threshold_value = sorted(x, reverse=True)[k - 1]
    return [v if v >= threshold_value else 0.0 for v in x]


def logical_and(x: List[bool]) -> bool:
    return all(x)


def logical_or(x: List[bool]) -> bool:
    return any(x)


def logical_xor(x: List[bool]) -> bool:
    return reduce(lambda a, b: a ^ b, x)


# dictionary to access by name
aggregations = {
    # Additive
    "sum": sum,
    "mean": statistics.mean,
    "abs_mean": abs_mean,
    # Multiplicative
    "prod": math.prod,
    "abs_prod": abs_prod,
    "geometric_mean": statistics.geometric_mean,
    # Selettive
    "max": max,
    "min": min,
    "median": statistics.median,
    "mode": statistics.mode,
    # Reciprocal-based
    "harmonic_mean": statistics.harmonic_mean,
    # Normalizzanti
    "softmax": softmax,
    "sparsemax": sparsemax,
    "log_sum_exp": log_sum_exp,
    # Norme vettoriali
    "norm_l1": norm_l1,
    "norm_l2": norm_l2,
    # Logiche
    "and": logical_and,
    "or": logical_or,
    "xor": logical_xor,
    # Sparse / Competitive
    "maxout": maxout,
    "k-winner": k_winner_take_all,
    "threshold": threshold,
}
