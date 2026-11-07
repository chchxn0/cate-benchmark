"""Generate one synthetic dataset and save it with a metadata sidecar.

Usage (from the project root):
    python scripts/generate_data.py
    python scripts/generate_data.py --n 10000 --treat-prob 0.3 --effect interaction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation import (  # noqa: E402
    EFFECT_FUNCTIONS,
    SimulationConfig,
    covariate_columns,
    effect_diagnostics,
    generate_data,
    save_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "simulation.yaml")
    parser.add_argument("--n", type=int)
    parser.add_argument("--treat-prob", type=float)
    parser.add_argument("--effect", choices=list(EFFECT_FUNCTIONS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--noise-sd", type=float)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    params = dict(cfg["default"])
    overrides = {
        "n": args.n,
        "treat_prob": args.treat_prob,
        "effect": args.effect,
        "seed": args.seed,
        "noise_sd": args.noise_sd,
    }
    params.update({k: v for k, v in overrides.items() if v is not None})
    config = SimulationConfig(**params)
    out_dir = args.out_dir or PROJECT_ROOT / cfg["output"]["synthetic_dir"]

    df = generate_data(config)
    csv_path, meta_path = save_dataset(df, config, out_dir)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"Config: {config}")
    print(f"Saved data    : {csv_path.relative_to(PROJECT_ROOT) if csv_path.is_relative_to(PROJECT_ROOT) else csv_path}")
    print(f"Saved metadata: {meta_path.relative_to(PROJECT_ROOT) if meta_path.is_relative_to(PROJECT_ROOT) else meta_path}")

    print("\n[First 5 rows]")
    print(df.head())

    n_treated = int(df["T"].sum())
    print("\n[Treatment groups]")
    print(f"Treated (T=1): {n_treated}  ({n_treated / len(df):.3f})")
    print(f"Control (T=0): {len(df) - n_treated}  ({1 - n_treated / len(df):.3f})")

    print("\n[true_cate]")
    print(f"mean = {df['true_cate'].mean():.4f}, sd = {df['true_cate'].std(ddof=1):.4f}")

    X = df[covariate_columns(config.n_covariates)].to_numpy()
    print(f"\n[Effect-function diagnostics on this dataset's X (n={config.n})]")
    print(effect_diagnostics(X, config.noise_sd))

    ref_n = cfg.get("diagnostics", {}).get("reference_n", 200_000)
    X_ref = np.random.default_rng(config.seed).standard_normal((ref_n, config.n_covariates))
    print(f"\n[Effect-function diagnostics on a reference draw (n={ref_n}, seed={config.seed})]")
    print(effect_diagnostics(X_ref, config.noise_sd))


if __name__ == "__main__":
    main()
