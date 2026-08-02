from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from realistic_niah_v4.prompt_counter_dynamics import (
    _bootstrap_seed_correlation,
    _linear_slope,
    _residualized_correlation,
)


def test_occurrence_adjustment_recovers_seed_level_association() -> None:
    rows = []
    for seed in range(10):
        seed_effect = float(seed - 4.5)
        for occurrence in range(1, 11):
            rows.append(
                {
                    "seed": seed,
                    "query_occurrence": occurrence,
                    "dispersion": 10.0 * occurrence + seed_effect,
                    "counter_noise": -3.0 * occurrence + 2.0 * seed_effect,
                }
            )
    frame = pd.DataFrame(rows)
    estimate = _residualized_correlation(frame, "dispersion")
    low, high = _bootstrap_seed_correlation(
        frame,
        "dispersion",
        seed=17,
        replicates=500,
    )
    assert estimate == pytest.approx(1.0)
    assert low > 0.99 and high <= 1.0 + 1e-12


def test_linear_slope_uses_the_full_normalized_occurrence_range() -> None:
    x = np.linspace(0.0, 1.0, 10)
    assert _linear_slope(x, 2.5 * x - 1.0) == pytest.approx(2.5)
