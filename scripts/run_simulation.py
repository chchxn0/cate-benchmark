"""Repeated simulation experiment over sample size, treatment ratio and effect complexity.

Usage (from the project root):
    python scripts/run_simulation.py --only-baseline --reps 3 --tag trial   # smoke test
    python scripts/run_simulation.py                                        # OFAT, n_reps from config
    python scripts/run_simulation.py --design full --tag full               # 27-condition factorial
    python scripts/run_simulation.py --resume                               # continue an interrupted run
    python scripts/run_simulation.py --n 1000 --resume                      # add conditions to an existing tag
    python scripts/run_simulation.py --tag formal --list-conditions         # list ids + completed reps (read-only)
    python scripts/run_simulation.py --tag formal --condition-id c03 --resume   # one condition
    python scripts/run_simulation.py --tag formal --condition-id c02 --resume --max-new-reps 5   # batch of 5

An existing <tag>_runs.csv is never replaced unless --overwrite is given.
Rows are appended to <tag>_runs.csv after every replication, so an interrupted
run can be resumed; the summary is rebuilt from all rows at the end.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import (  # noqa: E402
    BASELINE_MODEL,
    CONDITION_KEYS,
    build_conditions,
    check_condition_ids,
    check_runs_integrity,
    condition_status,
    model_fit_counts,
    paired_differences,
    pending_replications,
    run_replication,
    select_conditions,
    summarize_runs,
    timed,
)


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--design", choices=["ofat", "full"])
    parser.add_argument("--reps", type=int, help="number of replications (default: n_reps in config)")
    parser.add_argument("--only-baseline", action="store_true", help="run the baseline condition only")
    parser.add_argument("--n", type=int, help="run only conditions with this sample size")
    parser.add_argument("--condition-id", nargs="+", metavar="ID",
                        help="run only these condition ids (e.g. c01); see --list-conditions")
    parser.add_argument("--list-conditions", action="store_true",
                        help="print all condition ids with completed replications for --tag, then exit (read-only)")
    parser.add_argument("--tag", default="simulation", help="output file prefix")
    parser.add_argument("--resume", action="store_true", help="skip (condition, rep) pairs already in <tag>_runs.csv")
    parser.add_argument("--overwrite", action="store_true", help="delete an existing <tag>_runs.csv and start over")
    parser.add_argument("--max-new-reps", type=int, metavar="K",
                        help="run at most K not-yet-completed replications, then stop (rerun to continue)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(args.models.read_text(encoding="utf-8"))
    design = args.design or exp_cfg["design"]
    n_reps = args.reps or exp_cfg["n_reps"]
    seeds_cfg = exp_cfg["seeds"]

    all_conditions = build_conditions(exp_cfg["baseline"], exp_cfg["factors"], design)
    out_dir = PROJECT_ROOT / exp_cfg["output_dir"]
    runs_path = out_dir / f"{args.tag}_runs.csv"

    if args.list_conditions:
        existing = pd.read_csv(runs_path) if runs_path.exists() else None
        print(f"Design: {design} | tag: {args.tag} | runs file: {_rel(runs_path)}"
              f"{'' if existing is not None else ' (not created yet)'}")
        print(condition_status(all_conditions, existing, n_reps).to_string(index=False))
        return

    try:
        conditions = select_conditions(
            all_conditions,
            condition_ids=args.condition_id,
            n=args.n,
            baseline=exp_cfg["baseline"] if args.only_baseline else None,
        )
    except ValueError as err:
        raise SystemExit(f"Condition filter error: {err}")

    if args.max_new_reps is not None and args.max_new_reps < 1:
        raise SystemExit("--max-new-reps must be >= 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.tag}_summary.csv"
    meta_path = out_dir / f"{args.tag}_metadata.json"
    paired_path = out_dir / f"{args.tag}_paired.csv"

    expected_models = {"T-Learner", "X-Learner", "Causal Forest", BASELINE_MODEL}
    done: set[tuple] = set()
    if runs_path.exists():
        if args.resume:
            prev = pd.read_csv(runs_path)
            done = check_runs_integrity(prev, seeds_cfg, expected_models)
            check_condition_ids(prev, all_conditions)
        elif args.overwrite:
            others = set(pd.read_csv(runs_path)["condition_id"]) - set(conditions["condition_id"])
            if others:
                raise SystemExit(f"--overwrite refused: {runs_path.name} also holds conditions {sorted(others)} "
                                 "outside the current selection. Use a different --tag or --resume.")
            runs_path.unlink()
        else:
            raise SystemExit(f"{runs_path} already exists. Use --resume to continue it or --overwrite to replace it.")

    fits = model_fit_counts(model_cfg)
    total_jobs = len(conditions) * n_reps
    print(f"Design: {design} | conditions: {len(conditions)} | reps: {n_reps} | replications: {total_jobs}")
    print(f"Estimator fits: {total_jobs * fits['estimators']} "
          f"(base-learner fits: {total_jobs * (fits['T-Learner'] + fits['X-Learner'] + fits['Causal Forest'])})")
    print(conditions.to_string(index=False))
    if done:
        print(f"Resuming: {len(done)} replications (all conditions) already in {runs_path.name}")

    all_pending = pending_replications(conditions, n_reps, done)
    jobs = pending_replications(conditions, n_reps, done, args.max_new_reps)
    if not jobs:
        print("Nothing to run: every selected replication is already complete. No files were modified.")
        return
    print(f"Pending in selection: {len(all_pending)} | running now: {len(jobs)} -> "
          + ", ".join(f"{c['condition_id']}:rep{r}" for c, r in jobs), flush=True)

    timings = []
    for cond, rep in jobs:
        rows, elapsed = timed(run_replication, cond, rep, model_cfg, seeds_cfg)
        for row in rows:
            row["replication_wall_sec"] = elapsed
        pd.DataFrame(rows).to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)
        timings.append(elapsed)
        rmse_txt = ", ".join(f"{r['model']}={r['rmse']:.4f}" for r in rows)
        print(f"[{cond['condition_id']} n={cond['n']} p={cond['treat_prob']} {cond['effect']} rep={rep}] "
              f"{elapsed:.1f}s, peak RSS {rows[0]['peak_rss_mb']:.0f} MB | {rmse_txt}", flush=True)
    remaining = len(all_pending) - len(jobs)
    print(f"Ran {len(jobs)} replication(s); {remaining} still pending in this selection.", flush=True)

    runs = pd.read_csv(runs_path)
    check_runs_integrity(runs, seeds_cfg, expected_models)
    summary = summarize_runs(runs, exp_cfg["confidence_level"], n_reps)
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    paired = paired_differences(runs, exp_cfg["confidence_level"], n_reps)
    paired.to_csv(paired_path, index=False, float_format="%.6g")

    invocation = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "conditions": conditions["condition_id"].tolist(),
        "replications_run": len(timings),
        "replications_skipped_resume": total_jobs - len(all_pending),
        "max_new_reps": args.max_new_reps,
        "replications_pending_after": remaining,
        "wall_time_sec": sum(timings),
    }
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if (args.resume and meta_path.exists()) else {}
    metadata = {
        "created_at_utc": previous.get("created_at_utc", invocation["finished_at_utc"]),
        "updated_at_utc": invocation["finished_at_utc"],
        "invocations": previous.get("invocations", []) + [invocation],
        "design": design,
        "baseline": exp_cfg["baseline"],
        "only_baseline": args.only_baseline,
        "n_reps": n_reps,
        "n_conditions_in_design": len(all_conditions),
        "conditions_completed": sorted(runs["condition_id"].unique().tolist()),
        "conditions": all_conditions.to_dict(orient="records"),
        "seeds": {
            **seeds_cfg,
            "rule": "data_seed = data_seed_base + rep (data + split); model_seed = model_seed_base + rep",
        },
        "confidence_level": exp_cfg["confidence_level"],
        "ci_method": "mean +/- t_{(1+level)/2, k-1} * sd / sqrt(k) over replications",
        "models_config": model_cfg,
        "reference_model": f"{BASELINE_MODEL}: training difference-in-means applied to every test unit",
        "fits_per_replication": fits,
        "paired_method": "diff = RMSE(a) - RMSE(b) matched on (condition, rep); paired t CI; Holm within condition",
        "outputs": {"runs": runs_path.name, "summary": summary_path.name, "paired": paired_path.name},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **{pkg: version(pkg) for pkg in ["numpy", "pandas", "scikit-learn", "econml", "scipy"]},
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    cols = ["condition_id", "n", "treat_prob", "effect", "model", "n_reps",
            "rmse_mean", "rmse_sd", "rmse_ci_low", "rmse_ci_high", "mae_mean", "train_time_mean", "precision_note"]
    print("\n[Summary]")
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n[Paired RMSE differences: model_a - model_b]")
    pcols = ["condition_id", "model_a", "model_b", "n_pairs", "diff_mean", "diff_ci_low", "diff_ci_high", "p_holm", "note"]
    print(paired[pcols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSaved: " + ", ".join(_rel(p) for p in (runs_path, summary_path, paired_path, meta_path)))


if __name__ == "__main__":
    main()
