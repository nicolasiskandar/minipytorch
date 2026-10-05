"""Argument-shape helpers shared by the conv and pooling layers."""


def _pair(v):
    if isinstance(v, (tuple, list)):
        if len(v) != 2:
            raise ValueError(f"expected a pair of ints, got {v}")
        return (int(v[0]), int(v[1]))
    return (int(v), int(v))