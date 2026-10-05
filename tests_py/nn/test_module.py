"""Module/Parameter plumbing, the Sequential container, and Flatten."""

import numpy as np


def test_sequential():
    from core import nn

    l1 = nn.Linear(2, 2, activation='tanh')
    l2 = nn.Linear(2, 1, activation='sigmoid')
    seq = nn.Sequential(l1, l2)
    out = seq(np.array([0.5, -0.5]))
    assert out.shape == (1,)


def test_module_parameters():
    from core import nn

    l1 = nn.Linear(2, 3)
    l2 = nn.Linear(3, 1)
    seq = nn.Sequential(l1, l2)
    params = list(seq.parameters())
    assert len(params) == 4  # 2 weight+bias pairs
    named = list(seq.named_parameters())
    assert len(named) == 4
    assert named[0][0].startswith('0.weight') or named[0][0] == '0.weight'


def test_sequential_named_params():
    from core import nn

    l1 = nn.Linear(2, 2, activation='tanh')
    l2 = nn.Linear(2, 1, activation='sigmoid')
    seq = nn.Sequential(l1, l2)
    named = list(seq.named_parameters())
    assert len(named) == 4
    names = [n for n, _ in named]
    assert all('.' in n for n in names)


def test_flatten_flattens_to_one_dimension():
    from core import nn

    out = nn.Flatten()(np.arange(24, dtype=float).reshape(2, 3, 4))
    assert out.shape == (24,)
    assert np.allclose(out, np.arange(24, dtype=float))