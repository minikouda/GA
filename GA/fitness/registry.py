"""Registry and typing helpers for GA fitness strategies."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
FitnessScores = FloatArray
FitnessStrategy = Callable[[FloatArray, FloatArray, int, int], FitnessScores]

_STRATEGY_REGISTRY: Dict[str, FitnessStrategy] = {}


def register_fitness_strategy(name: str, strategy: FitnessStrategy) -> None:
    """Register a new fitness strategy under the provided name."""
    key = name.lower()
    _STRATEGY_REGISTRY[key] = strategy


def get_fitness_strategy(name: str) -> FitnessStrategy:
    """Retrieve a registered fitness strategy by name."""
    key = name.lower()
    try:
        return _STRATEGY_REGISTRY[key]
    except KeyError as exc:  # pragma: no cover - error path
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys())) or "<none>"
        raise ValueError(
            f"Unknown fitness strategy '{name}'. Available strategies: {available}"
        ) from exc


def list_available_strategies() -> Tuple[str, ...]:
    """Return the tuple of available fitness strategy names."""
    return tuple(sorted(_STRATEGY_REGISTRY.keys()))


__all__ = [
    "FitnessScores",
    "FitnessStrategy",
    "get_fitness_strategy",
    "list_available_strategies",
    "register_fitness_strategy",
]
