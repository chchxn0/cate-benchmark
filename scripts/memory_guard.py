"""Run a command while sampling its memory; terminate it if a limit is exceeded.

Usage:
    python scripts/memory_guard.py --limit-mb 7000 --log results/tables/x_memory.csv -- \
        python scripts/run_simulation.py --n 100000 --reps 1 --tag probe_n100000

Samples the RSS of the child process tree every `--interval` seconds and writes
a timeline CSV (elapsed_sec, rss_mb). If RSS exceeds `--limit-mb`, the child is
terminated (SIGTERM, then SIGKILL) and the guard exits with code 3.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time


def tree_rss_mb(pid: int) -> float:
    """Sum of RSS over `pid` and its descendants, via `ps` (macOS / Linux)."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) == 3:
            p, pp, r = map(int, parts)
            children.setdefault(pp, []).append(p)
            rss[p] = r
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(children.get(p, []))
    return total / 1024  # ps reports KiB


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit-mb", type=float, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--log", required=True, help="timeline CSV path")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        parser.error("no command given")

    start = time.perf_counter()
    proc = subprocess.Popen(cmd)
    peak, killed = 0.0, False
    with open(args.log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "rss_mb"])
        while proc.poll() is None:
            rss = tree_rss_mb(proc.pid)
            peak = max(peak, rss)
            writer.writerow([f"{time.perf_counter() - start:.2f}", f"{rss:.1f}"])
            f.flush()
            if rss > args.limit_mb:
                print(f"[memory_guard] RSS {rss:.0f} MB > limit {args.limit_mb:.0f} MB — terminating", file=sys.stderr)
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.kill(proc.pid, signal.SIGKILL)
                    proc.wait()
                killed = True
                break
            time.sleep(args.interval)

    elapsed = time.perf_counter() - start
    print(f"[memory_guard] exit={proc.returncode} wall={elapsed:.1f}s sampled_peak_rss={peak:.0f} MB"
          f"{' (KILLED: limit exceeded)' if killed else ''}", file=sys.stderr)
    return 3 if killed else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
