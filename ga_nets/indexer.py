"""Indexing di networks e neuroni"""

from itertools import count

from ga_nets.errors import InvalidCategory
from ga_nets.custom_types import CategoryType


class Indexer:
    """Gestisce le chiavi di indicizzazione per categorie come neuroni
    o layer."""

    indexes = {}

    @classmethod
    def _validate_category(cls, key):
        if not isinstance(key, CategoryType):
            raise InvalidCategory(f"Invalid category: {key}")

    @classmethod
    def get_id(cls, key: CategoryType) -> int:
        """Restituisce un indice incrementale associato a una chiave.

        Args:
            key: Il nome della categoria (es. 'neuron', 'layer').

        Returns:
            Intero rappresentante un ID univoco nella categoria.

        Exceptions:
          InvalidCategory:
        """
        cls._validate_category(key)

        if key not in cls.indexes:
            cls.reset(key)
        return next(cls.indexes[key])

    @classmethod
    def reset(cls, key: CategoryType) -> None:
        """Resetta il contatore della chiave specificata.

        Args:
            key: Il nome della categoria da resettare.
        """
        cls._validate_category(key)
        cls.indexes[key] = count(0)
