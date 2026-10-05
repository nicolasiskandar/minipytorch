"""Public entry points: :func:`save_model` and :func:`load_model`.

The only decision made here is which format to use: Linear-only networks keep
the byte-for-byte legacy layout, everything else gets the tagged one.
"""

from ..nn import Sequential
from .format import _FORMAT_TAG, _next_nonblank
from .legacy import _is_legacy_linear_only, _load_legacy, _save_legacy
from .tagged import _load_tagged, _save_tagged


def save_model(model, filename: str) -> None:
    if isinstance(model, Sequential):
        layers = list(model.layers)
    else:
        layers = [model]

    with open(filename, "w", encoding="utf-8") as out:
        if _is_legacy_linear_only(layers):
            _save_legacy(out, layers)
        else:
            _save_tagged(out, layers)


def load_model(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = _next_nonblank(lines, 0)
    if idx >= len(lines):
        raise ValueError(f"{filename!r} is empty")
    first = lines[idx].strip()
    if first == _FORMAT_TAG:
        return _load_tagged(lines, idx + 1, filename)
    return _load_legacy(lines, idx, filename)