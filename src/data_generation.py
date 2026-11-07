"""Synthetic data generation for the CATE benchmark.

Data generating process (DGP)
-----------------------------
    X_1, ..., X_10 ~ iid N(0, 1)
    T ~ Bernoulli(treat_prob)            (randomised: independent of X)
    mu0(X) = 0.2 X1 - 0.1 X2 + 0.3 X3
    tau(X) = TAU_MEAN + TAU_SCALE * g(X)
    Y = mu0(X) + T * tau(X) + eps,       eps ~ N(0, noise_sd^2)

Every effect function shares the same form tau = 0.5 + 0.3 * g(X), where g has
theoretical mean 0 and variance 1 under X ~ N(0, I). As a result, E[tau] = 0.5
and SD[tau] = 0.3 for every complexity level, so moving from "linear" to
"nonlinear"/"interaction" changes the *shape* of the effect, not its average
size or heterogeneity.

Random draws (single numpy Generator, PCG64, in this fixed order):
    1. X     : standard_normal((n, n_covariates))
    2. T     : uniform(n) < treat_prob
    3. eps   : normal(0, noise_sd, n)
Because the order is fixed, datasets with the same seed and n share X, U and
eps across treatment ratios and effect types (common random numbers).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

N_COVARIATES = 10
TAU_MEAN = 0.5
TAU_SCALE = 0.3

# SD of sin(pi * Z) for Z ~ N(0, 1): Var = (1 - E[cos(2 pi Z)]) / 2 = (1 - exp(-2 pi^2)) / 2
_SIN_PI_SD = float(np.sqrt((1.0 - np.exp(-2.0 * np.pi**2)) / 2.0))


# ---------------------------------------------------------------------------
# Outcome and effect functions
# ---------------------------------------------------------------------------
def mu0(X: np.ndarray) -> np.ndarray:
    """Expected untreated outcome: 0.2 X1 - 0.1 X2 + 0.3 X3."""
    return 0.2 * X[:, 0] - 0.1 * X[:, 1] + 0.3 * X[:, 2]


def tau_linear(X: np.ndarray) -> np.ndarray:
    """Linear effect: 0.5 + 0.3 X1."""
    return TAU_MEAN + TAU_SCALE * X[:, 0]


def tau_nonlinear(X: np.ndarray) -> np.ndarray:
    """Nonlinear (additive) effect: 0.5 + 0.3 * g(X).

    g(X) = [ (X1^2 - 1) / sqrt(2) + sin(pi X2) / s ] / sqrt(2),
    s = sqrt((1 - exp(-2 pi^2)) / 2).
    Each bracketed term has mean 0 and variance 1 and they are independent,
    so g has mean 0 and variance 1.
    """
    quad = (X[:, 0] ** 2 - 1.0) / np.sqrt(2.0)
    wave = np.sin(np.pi * X[:, 1]) / _SIN_PI_SD
    return TAU_MEAN + TAU_SCALE * (quad + wave) / np.sqrt(2.0)


def tau_interaction(X: np.ndarray) -> np.ndarray:
    """Pure interaction effect: 0.5 + 0.3 X1 X2 (X1 X2 has mean 0, variance 1)."""
    return TAU_MEAN + TAU_SCALE * X[:, 0] * X[:, 1]


EFFECT_FUNCTIONS = {
    "linear": tau_linear,
    "nonlinear": tau_nonlinear,
    "interaction": tau_interaction,
}

EFFECT_FORMULAS = {
    "linear": "0.5 + 0.3*X1",
    "nonlinear": "0.5 + 0.3*[(X1^2-1)/sqrt(2) + sin(pi*X2)/s]/sqrt(2), s=sqrt((1-exp(-2*pi^2))/2)",
    "interaction": "0.5 + 0.3*X1*X2",
}
MU0_FORMULA = "0.2*X1 - 0.1*X2 + 0.3*X3"


# ---------------------------------------------------------------------------
# Configuration and generation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationConfig:
    n: int = 1000
    treat_prob: float = 0.5
    effect: str = "linear"
    seed: int = 42
    noise_sd: float = 1.0
    n_covariates: int = N_COVARIATES

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if not 0.0 < self.treat_prob < 1.0:
            raise ValueError(f"treat_prob must be in (0, 1), got {self.treat_prob}")
        if self.effect not in EFFECT_FUNCTIONS:
            raise ValueError(f"effect must be one of {list(EFFECT_FUNCTIONS)}, got {self.effect!r}")
        if self.noise_sd < 0:
            raise ValueError(f"noise_sd must be non-negative, got {self.noise_sd}")
        if self.n_covariates < 3:
            raise ValueError("n_covariates must be at least 3 (mu0 uses X1..X3)")


def covariate_columns(n_covariates: int = N_COVARIATES) -> list[str]:
    return [f"X{j}" for j in range(1, n_covariates + 1)]


def generate_data(config: SimulationConfig) -> pd.DataFrame:
    """Generate one synthetic dataset according to the module-level DGP."""
    rng = np.random.default_rng(config.seed)

    X = rng.standard_normal((config.n, config.n_covariates))
    T = (rng.random(config.n) < config.treat_prob).astype(np.int64)
    eps = rng.normal(0.0, config.noise_sd, config.n)

    base = mu0(X)
    tau = EFFECT_FUNCTIONS[config.effect](X)
    Y = base + T * tau + eps

    df = pd.DataFrame(X, columns=covariate_columns(config.n_covariates))
    df["T"] = T
    df["mu0"] = base
    df["true_cate"] = tau
    df["Y"] = Y
    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _linear_r2(X: np.ndarray, target: np.ndarray) -> float:
    """Share of the variance of `target` explained by an OLS fit on [1, X]."""
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    ss_tot = np.sum((target - target.mean()) ** 2)
    return float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan")


def effect_diagnostics(X: np.ndarray, noise_sd: float = 1.0) -> pd.DataFrame:
    """Compare the scale and shape of every effect function on the same X.

    Columns
    -------
    mean, sd, min, q05, median, q95, max : distribution of tau(X)
    skew                                  : sample skewness of tau(X)
    sd_over_noise                         : heterogeneity relative to the noise SD
    linear_r2                             : variance of tau explained by a linear fit in X
                                            (1 = fully linear, ~0 = invisible to linear models)
    theory_mean, theory_sd                : analytical values under X ~ N(0, I)
    """
    rows = {}
    for name, fn in EFFECT_FUNCTIONS.items():
        tau = fn(X)
        sd = tau.std(ddof=1)
        centred = tau - tau.mean()
        rows[name] = {
            "mean": tau.mean(),
            "sd": sd,
            "min": tau.min(),
            "q05": np.quantile(tau, 0.05),
            "median": np.median(tau),
            "q95": np.quantile(tau, 0.95),
            "max": tau.max(),
            "skew": float(np.mean(centred**3) / np.std(tau) ** 3),
            "sd_over_noise": sd / noise_sd if noise_sd > 0 else float("inf"),
            "linear_r2": _linear_r2(X, tau),
            "theory_mean": TAU_MEAN,
            "theory_sd": TAU_SCALE,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def dataset_stem(config: SimulationConfig) -> str:
    return (
        f"synthetic_n{config.n}_p{config.treat_prob:g}_{config.effect}"
        f"_noise{config.noise_sd:g}_seed{config.seed}"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(config: SimulationConfig, df: pd.DataFrame, csv_path: Path) -> dict:
    n_treated = int(df["T"].sum())
    return {
        "dataset_file": csv_path.name,
        "sha256": _sha256(csv_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "dgp": {
            "covariates": f"X1..X{config.n_covariates} iid N(0, 1)",
            "treatment": f"T ~ Bernoulli({config.treat_prob}), independent of X",
            "mu0": MU0_FORMULA,
            "tau": EFFECT_FORMULAS[config.effect],
            "outcome": f"Y = mu0 + T*tau + eps, eps ~ N(0, {config.noise_sd}^2)",
            "rng": "numpy.random.default_rng(seed) [PCG64]; draw order: X, U(T), eps",
        },
        "summary": {
            "n_rows": int(len(df)),
            "n_treated": n_treated,
            "n_control": int(len(df) - n_treated),
            "treated_share": n_treated / len(df),
            "true_cate_mean": float(df["true_cate"].mean()),
            "true_cate_sd": float(df["true_cate"].std(ddof=1)),
        },
        "columns": list(df.columns),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }


def save_dataset(df: pd.DataFrame, config: SimulationConfig, out_dir: Path | str) -> tuple[Path, Path]:
    """Write `<stem>.csv` and its `<stem>.meta.json` sidecar; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_stem(config)
    csv_path = out_dir / f"{stem}.csv"
    meta_path = out_dir / f"{stem}.meta.json"

    df.to_csv(csv_path, index=False, float_format="%.10g")
    meta = build_metadata(config, df, csv_path)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path, meta_path
