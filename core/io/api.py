"""Public entry points: :func:`save_model` and :func:`load_model`."""

from ..nn import Sequential
from .format import _FORMAT_TAG, _next_nonblank
from .tagged import _load_tagged, _save_tagged


def save_model(model, filename: str) -> None:
    if isinstance(model, Sequential):
        layers = list(model.layers)
    else:
        layers = [model]

    with open(filename, "w", encoding="utf-8") as out:
        _save_tagged(out, layers)


def load_model(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = _next_nonblank(lines, 0)
    if idx >= len(lines):
        raise ValueError(f"{filename!r} is empty")
    if lines[idx].strip() != _FORMAT_TAG:
        raise ValueError(
            f"{filename!r} is not a {_FORMAT_TAG} model: expected "
            f"{_FORMAT_TAG!r} on the first non-blank line, found "
            f"{lines[idx].strip()!r}"
        )
    return _load_tagged(lines, idx + 1, filename)
