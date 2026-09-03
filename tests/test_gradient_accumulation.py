"""Gradient accumulation helpers and MSE equivalence with a larger batch."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.train import is_optimizer_step, resolve_accumulation_steps


def test_resolve_defaults_to_no_accumulation():
    assert resolve_accumulation_steps({"batch_size": 2}) == 1
    assert resolve_accumulation_steps({"batch_size": 2, "effective_batch_size": 2}) == 1


def test_resolve_effective_8_and_16_with_microbatch_2():
    assert resolve_accumulation_steps({"batch_size": 2, "effective_batch_size": 8}) == 4
    assert resolve_accumulation_steps({"batch_size": 2, "effective_batch_size": 16}) == 8


def test_resolve_rejects_non_multiple():
    with pytest.raises(ValueError, match="multiple"):
        resolve_accumulation_steps({"batch_size": 2, "effective_batch_size": 7})


def test_resolve_rejects_non_positive():
    with pytest.raises(ValueError, match="batch_size"):
        resolve_accumulation_steps({"batch_size": 0})
    with pytest.raises(ValueError, match="effective_batch_size"):
        resolve_accumulation_steps({"batch_size": 2, "effective_batch_size": 0})


def test_optimizer_steps_include_epoch_leftover():
    # 10 micro-batches, accum 4 → step at 4, 8, and leftover 10
    stepped = [
        i for i in range(1, 11) if is_optimizer_step(i, n_batches=10, accum_steps=4)
    ]
    assert stepped == [4, 8, 10]


def test_optimizer_steps_when_epoch_divides_evenly():
    stepped = [
        i for i in range(1, 9) if is_optimizer_step(i, n_batches=8, accum_steps=4)
    ]
    assert stepped == [4, 8]


def _train_one_epoch(batch_size: int, effective: int, xs: torch.Tensor, ys: torch.Tensor):
    torch.manual_seed(0)
    model = nn.Linear(4, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    accum = effective // batch_size
    n_batches = xs.shape[0] // batch_size
    opt.zero_grad()
    for step_i, start in enumerate(range(0, xs.shape[0], batch_size), start=1):
        pred = model(xs[start : start + batch_size])
        loss = F.mse_loss(pred, ys[start : start + batch_size])
        (loss / accum).backward()
        if is_optimizer_step(step_i, n_batches, accum):
            opt.step()
            opt.zero_grad()
    return model.weight.detach().clone()


def test_accumulation_matches_true_batch_for_mean_mse():
    torch.manual_seed(1)
    xs = torch.randn(8, 4)
    ys = torch.randn(8, 1)
    true_batch = _train_one_epoch(batch_size=8, effective=8, xs=xs, ys=ys)
    accumulated = _train_one_epoch(batch_size=2, effective=8, xs=xs, ys=ys)
    torch.testing.assert_close(true_batch, accumulated)
