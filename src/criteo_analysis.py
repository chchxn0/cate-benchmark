"""Diagnostics for comparing Criteo uplift models across training sizes (STEP 5-4).

tie_summary      : how many predictions share a value (exact zeros, mode, tie blocks)
segment_effect   : observed treated - control conversion difference in a score segment
                   (e.g. the lowest 3% of scores) with a normal-approximation 95% CI
"""

from __future__ import annotations

import numpy as np

Z95 = 1.959964


def tie_summary(score) -> dict:
    """Exact-zero share, mode share and tie-block sizes of a prediction vector."""
    s = np.asarray(score, dtype=float)
    _, counts = np.unique(s, return_counts=True)
    tied = counts[counts > 1]
    mode_i = int(counts.argmax())
    return {
        "n": int(len(s)),
        "share_zero": float((s == 0).mean()),
        "n_unique": int(len(counts)),
        "mode_value": float(np.unique(s)[mode_i]),
        "mode_share": float(counts[mode_i] / len(s)),
        "largest_tie_block": int(counts.max()),
        "n_tie_blocks": int(len(tied)),
        "share_rows_in_tie_blocks": float(tied.sum() / len(s)),
    }


def segment_effect(score, treatment, y, fraction: float, lowest: bool = True) -> dict:
    """Observed rate difference among the `fraction` lowest (or highest) scores.

    The segment is the first round(fraction * N) rows of a stable sort; when the boundary falls
    inside a tie block, the rows taken from that block follow input order (reported via
    `boundary_tie_rows`). SE = sqrt(p_t(1-p_t)/n_t + p_c(1-p_c)/n_c).
    """
    s = np.asarray(score, dtype=float)
    t = np.asarray(treatment).astype(int)
    yy = np.asarray(y).astype(int)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = int(round(fraction * len(s)))
    order = np.argsort(s if lowest else -s, kind="stable")
    seg = order[:k]
    ts, ys = t[seg], yy[seg]
    n_t, n_c = int(ts.sum()), int(k - ts.sum())
    y_t, y_c = int((ts * ys).sum()), int(((1 - ts) * ys).sum())
    if n_t == 0 or n_c == 0:
        raise ValueError("segment has no treated or no control rows")
    p_t, p_c = y_t / n_t, y_c / n_c
    se = float(np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))
    boundary = s[order[k - 1]]
    return {
        "segment": f"{'lowest' if lowest else 'highest'} {fraction:.0%}", "k": k,
        "score_threshold": float(boundary),
        "boundary_tie_rows": int((s == boundary).sum()) if k < len(s) and s[order[k]] == boundary else 0,
        "n_treated": n_t, "n_control": n_c, "conv_treated": y_t, "conv_control": y_c,
        "rate_treated": p_t, "rate_control": p_c, "rate_diff": p_t - p_c, "rate_diff_se": se,
        "rate_diff_ci_low": p_t - p_c - Z95 * se, "rate_diff_ci_high": p_t - p_c + Z95 * se,
        "mean_score": float(s[seg].mean()),
    }
