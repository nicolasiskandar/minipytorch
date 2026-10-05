"""Model serialization.

``save_model`` has two formats and ``load_model`` sniffs the first non-blank line
to pick between them. See :mod:`core.io.format` for the tag and name tables,
:mod:`core.io.legacy` for the byte-for-byte Linear-only text,
:mod:`core.io.tagged` for the v2 format, and :mod:`core.io.registry` for the
stateless-layer table.

This package consumes ``core.nn`` and must never be imported *by* it, or the
two would form a cycle.
"""

from .api import load_model, save_model

from .format import (
    _ACTIVATION_NAMES,
    _FORMAT_TAG,
    _REVERSE_ACTIVATION_NAMES,
    _get_activation_name,
)
from .registry import _STATELESS_LAYERS

__all__ = ["save_model", "load_model"]