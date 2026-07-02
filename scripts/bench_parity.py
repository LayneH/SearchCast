#!/usr/bin/env python
"""Benchmark, parity, and self-test harness for the optuna_ridge pipeline.

Subcommands:
  run       Run a reduced but representative search slice and record per-trial
            results + wall time into an output directory.
  compare   Compare two `run` output directories (val MSE per trial, chosen
            alphas, local_results.csv) and report parity metrics.
  selftest  Numerical unit checks for the solver/data-pipeline invariants.

Typical workflow:
  python scripts/bench_parity.py run --out exps/bench/baseline
  ... make changes ...
  python scripts/bench_parity.py run --out exps/bench/candidate
  python scripts/bench_parity.py compare exps/bench/baseline exps/bench/candidate
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Canonical config from scripts/reproduce.sh, shrunk to a representative slice:
# pooled multi-series group, 3-fold expanding CV, 3 spread-out horizon groups.
DATASETS = {
    "etth1": {"csv": "data/ETTh1.csv", "sgs": 7},
    "etth2": {"csv": "data/ETTh2.csv", "sgs": 7},
    "weather": {"csv": "data/weather.csv", "sgs": 21},
    "exchange": {"csv": "data/exchange_rate.csv", "sgs": 8},
}


def cmd_run(args):
    cfg = DATASETS[args.dataset]
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "-u", "optuna_ridge.py",
        "--input_csv", cfg["csv"],
        "--output_dir", out_dir,
        "--scaler_scope", "local", "--scaler_method", "mean",
        "--local_horizon_group_size", "24",
        "--local_series_group_size", str(cfg["sgs"]),
        "--instance_norm",
        "--n_folds", "3",
        "--pool_series",
        "--n_trials", str(args.n_trials),
        "--seed", str(args.seed),
        "--horizon_subset", args.horizon_subset,
        "--trial_log", os.path.join(out_dir, "trials.csv"),
    ]
    if args.precision != "fp64":
        cmd += ["--precision", args.precision]
    if args.noise == "none":
        cmd += ["--fixed_noise_type", "none"]
    if args.profile:
        cmd += ["--profile"]
    cmd += args.extra_args

    print("Running:", " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        sys.exit(result.returncode)

    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True).stdout.strip()
    git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                    capture_output=True, text=True).stdout.strip())
    meta = {
        "dataset": args.dataset,
        "n_trials": args.n_trials,
        "precision": args.precision,
        "seed": args.seed,
        "horizon_subset": args.horizon_subset,
        "noise": args.noise,
        "extra_args": args.extra_args,
        "wall_time_s": round(wall, 2),
        "git_rev": git_rev,
        "git_dirty": git_dirty,
        "cmd": cmd,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWall time: {wall:.1f}s. Outputs in {out_dir}")


def _load_trials(out_dir):
    """Returns {(study, trial): row_dict} from trials.csv."""
    import csv as _csv
    path = os.path.join(out_dir, "trials.csv")
    trials = {}
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            key = (row["study"], int(row["trial"]))
            row["value"] = float(row["value"]) if row["value"] else None
            row["best_alpha"] = float(row["best_alpha"]) if row["best_alpha"] else None
            row["params"] = json.loads(row["params"])
            row["mse_per_alpha"] = json.loads(row["mse_per_alpha"])
            trials[key] = row
    return trials


def _rel_diff(a, b):
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def cmd_compare(args):
    ta, tb = _load_trials(args.dir_a), _load_trials(args.dir_b)
    if set(ta) != set(tb):
        only_a, only_b = set(ta) - set(tb), set(tb) - set(ta)
        print(f"WARNING: trial sets differ (only in A: {len(only_a)}, only in B: {len(only_b)})")
    common = sorted(set(ta) & set(tb))
    if not common:
        print("FAIL: no common trials to compare")
        sys.exit(1)

    param_mismatches = []
    alpha_mismatches = []
    max_val_rel = 0.0
    max_val_rel_key = None
    max_msepa_rel = 0.0
    for key in common:
        a, b = ta[key], tb[key]
        if a["params"] != b["params"]:
            param_mismatches.append(key)
            continue  # different HPs make value comparison meaningless
        if a["value"] is not None and b["value"] is not None:
            rd = _rel_diff(a["value"], b["value"])
            if rd > max_val_rel:
                max_val_rel, max_val_rel_key = rd, key
        if a["best_alpha"] != b["best_alpha"]:
            alpha_mismatches.append(key)
        for ma, mb in zip(a["mse_per_alpha"], b["mse_per_alpha"]):
            max_msepa_rel = max(max_msepa_rel, _rel_diff(ma, mb))

    n = len(common)
    n_matched = n - len(param_mismatches)
    alpha_frac = (n_matched - len(alpha_mismatches)) / max(1, n_matched)
    print(f"Compared {n} trials:")
    print(f"  identical params:        {n_matched}/{n}")
    print(f"  identical chosen alpha:  {n_matched - len(alpha_mismatches)}"
          f"/{n_matched} (of param-matched, {alpha_frac:.1%})")
    print(f"  max rel diff val MSE:    {max_val_rel:.3e}" +
          (f" at {max_val_rel_key}" if max_val_rel_key else ""))
    print(f"  max rel diff mse/alpha:  {max_msepa_rel:.3e}")
    if param_mismatches:
        print(f"  param mismatches at: {param_mismatches[:10]}"
              + (" ..." if len(param_mismatches) > 10 else ""))

    # local_results.csv: final selected models and their test MSEs
    try:
        import pandas as pd
        la = pd.read_csv(os.path.join(args.dir_a, "local_results.csv"))
        lb = pd.read_csv(os.path.join(args.dir_b, "local_results.csv"))
        if len(la) == len(lb):
            tm = max(_rel_diff(x, y) for x, y in zip(la["test_mse"], lb["test_mse"]))
            print(f"  max rel diff test MSE:   {tm:.3e} over {len(la)} refit models")
        else:
            print(f"  local_results.csv row counts differ: {len(la)} vs {len(lb)}")
    except FileNotFoundError:
        print("  (local_results.csv missing in one run; skipped)")

    for meta_dir in (args.dir_a, args.dir_b):
        meta_path = os.path.join(meta_dir, "run_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  {meta_dir}: wall={meta['wall_time_s']}s rev={meta['git_rev'][:8]}"
                  f"{'+dirty' if meta['git_dirty'] else ''}")

    msepa_tol = args.msepa_tol if args.msepa_tol is not None else args.tol
    ok = (not param_mismatches
          and alpha_frac >= args.alpha_match
          and max_val_rel <= args.tol
          and max_msepa_rel <= msepa_tol)
    print("PARITY: " + ("PASS" if ok else
                        f"FAIL (tol={args.tol:g}, msepa_tol={msepa_tol:g}, "
                        f"alpha_match>={args.alpha_match:g})"))
    sys.exit(0 if ok else 1)


# ==========================================
# Self-tests
# ==========================================
def _selftest_windowing():
    """get_context_and_horizons must match a naive per-window loop."""
    import torch
    from optuna_ridge import get_context_and_horizons

    torch.manual_seed(0)
    S, T, L = 3, 200, 16
    horizons = [1, 5, 24]
    data = torch.randn(S, T)
    X, Y = get_context_and_horizons(data, L, horizons)
    H_max = max(horizons)
    n_windows = T - (L + H_max) + 1
    assert X.shape == (S, n_windows, L), X.shape
    assert Y.shape == (S, n_windows, len(horizons)), Y.shape
    for s in range(S):
        for i in range(0, n_windows, 37):
            assert torch.equal(X[s, i], data[s, i:i + L])
            for j, h in enumerate(horizons):
                assert torch.equal(Y[s, i, j], data[s, i + L + h - 1])
    return "windowing matches naive loop"


def _selftest_ridge_closed_form():
    """RidgeSolver.solve (no scaler) must match the closed-form normal equations,
    including not regularizing the intercept."""
    import torch
    from optuna_ridge import RidgeSolver

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, L, H = 500, 7, 3
    X = torch.randn(N, L, dtype=torch.float64)
    Y = torch.randn(N, H, dtype=torch.float64)
    alphas = torch.tensor([1e-3, 1.0, 100.0], dtype=torch.float64, device=device)

    solver = RidgeSolver(device)
    Theta = solver.solve(X, Y, alphas, scaler=None, chunk_size=128)

    ones = torch.ones(N, 1, dtype=torch.float64)
    Xi = torch.cat([ones, X], dim=1)
    G = Xi.T @ Xi
    B = Xi.T @ Y
    for k, alpha in enumerate(alphas.cpu()):
        reg = torch.eye(L + 1, dtype=torch.float64) * alpha
        reg[0, 0] = 0.0  # intercept not regularized
        ref = torch.linalg.solve(G + reg, B)
        err = (Theta[k].cpu() - ref).abs().max().item()
        assert err < 1e-9, f"alpha={alpha}: max abs err {err}"
    return "solve matches closed form (intercept unregularized)"


def _selftest_local_norm_solve():
    """LocalNormScaler path: per-window centering + appended scale feature,
    reproduced against an explicit dense construction."""
    import torch
    from optuna_ridge import RidgeSolver, LocalNormScaler, StandardStrategy

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, L, H = 400, 8, 2
    X = torch.randn(N, L, dtype=torch.float64) * 3 + 5
    Y = torch.randn(N, H, dtype=torch.float64)
    alphas = torch.tensor([0.5], dtype=torch.float64, device=device)

    last_k = 4
    solver = RidgeSolver(device)
    scaler = LocalNormScaler(StandardStrategy(), L, last_k)
    # chunk_size >= N so the per-chunk scaler fit sees all rows at once,
    # matching the dense reference below
    Theta = solver.solve(X, Y, alphas, scaler=scaler, chunk_size=N)

    wins = X[:, -last_k:]
    center = wins.mean(dim=-1, keepdim=True)
    var = torch.var(wins - center, dim=-1, keepdim=True, correction=0)
    scale = torch.sqrt(var + 1e-5)
    Xt = torch.cat([X - center, scale], dim=-1)  # (N, L+1), no intercept
    Yt = Y - center
    G = Xt.T @ Xt
    B = Xt.T @ Yt
    ref = torch.linalg.solve(G + 0.5 * torch.eye(L + 1, dtype=torch.float64), B)
    err = (Theta[0].cpu() - ref).abs().max().item()
    assert err < 1e-9, f"max abs err {err}"
    return "local-norm solve matches dense reference (all features regularized)"


def _selftest_pooled_3d_solve():
    """solve()/predict() on 3D (S, N, L) windows must equal the flattened 2D
    formulation (pooled model), up to chunk-order rounding."""
    import torch
    from optuna_ridge import RidgeSolver, LocalNormScaler, StandardStrategy

    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, N, L, H = 3, 300, 6, 2
    X3 = torch.randn(S, N, L, dtype=torch.float64)
    Y3 = torch.randn(S, N, H, dtype=torch.float64)
    alphas = torch.tensor([1e-2, 10.0], dtype=torch.float64, device=device)

    solver = RidgeSolver(device)
    for scaler_factory in (lambda: None,
                           lambda: LocalNormScaler(StandardStrategy(), L, 3)):
        t3 = solver.solve(X3, Y3, alphas, scaler=scaler_factory(), chunk_size=128)
        t2 = solver.solve(X3.reshape(-1, L), Y3.reshape(-1, H), alphas,
                          scaler=scaler_factory(), chunk_size=128)
        err = (t3 - t2).abs().max().item()
        assert err < 1e-12, f"3D vs 2D solve differ: {err}"

        p3 = solver.predict(X3, t3, scaler=scaler_factory(), chunk_size=128)
        p2 = solver.predict(X3.reshape(-1, L), t3, scaler=scaler_factory(), chunk_size=128)
        err = (p3 - p2).abs().max().item()
        assert err < 1e-12, f"3D vs 2D predict differ: {err}"
    return "3D windows == flattened 2D (solve & predict), both scaler paths"


def _selftest_fused_val_mse():
    """val_mse / val_mse_batched must match the unfused predict()+mean reference."""
    import torch
    from optuna_ridge import (RidgeSolver, LocalNormScaler, GlobalScaler,
                              StandardStrategy)

    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, N, L, H, K = 3, 250, 5, 4, 3
    X = torch.randn(S, N, L, dtype=torch.float64) + 2
    Y = torch.randn(S, N, H, dtype=torch.float64) + 2
    alphas = torch.logspace(-2, 2, K, dtype=torch.float64, device=device)
    solver = RidgeSolver(device)

    def make_scalers():
        gs = GlobalScaler(StandardStrategy())
        gs.fit(X.reshape(1, -1))
        return [None, LocalNormScaler(StandardStrategy(), L, L), gs]

    # Pooled/single path
    for scaler in make_scalers():
        theta = solver.solve(X, Y, alphas, scaler=scaler, chunk_size=97)
        fused = solver.val_mse(X, Y, theta, scaler=scaler, chunk_size=97)
        pred = solver.predict(X, theta, scaler=scaler, chunk_size=97)
        if isinstance(scaler, GlobalScaler):
            pred = scaler.inv_transform(pred)
        ref = ((pred.cpu() - Y.reshape(-1, H).unsqueeze(0)) ** 2).mean(dim=(-2, -1))
        err = (fused - ref).abs().max().item()
        assert err < 1e-12, f"{type(scaler).__name__}: fused vs ref {err}"

    # Batched (per-series models) path
    for scaler in make_scalers():
        theta_b = solver.solve_batched(X, Y, alphas, scaler=scaler, chunk_size=97)
        fused = solver.val_mse_batched(X, Y, theta_b, scaler=scaler, chunk_size=97)
        per_series = []
        for s in range(S):
            pred = solver.predict(X[s], theta_b[:, s], scaler=scaler, chunk_size=97)
            if isinstance(scaler, GlobalScaler):
                pred = scaler.inv_transform(pred)
            per_series.append(((pred.cpu() - Y[s].unsqueeze(0)) ** 2).mean(dim=(-2, -1)))
        ref = torch.stack(per_series, dim=1).mean(dim=1)
        err = (fused - ref).abs().max().item()
        assert err < 1e-12, f"batched {type(scaler).__name__}: fused vs ref {err}"
    return "fused val MSE == predict()+mean reference (single & batched, all scalers)"


def _selftest_solve_batched_per_series():
    """solve_batched must equal an independent solve() per series."""
    import torch
    from optuna_ridge import RidgeSolver

    torch.manual_seed(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, N, L, H, K = 4, 200, 5, 3, 5
    X = torch.randn(S, N, L, dtype=torch.float64)
    Y = torch.randn(S, N, H, dtype=torch.float64)
    alphas = torch.logspace(-3, 3, K, dtype=torch.float64, device=device)
    solver = RidgeSolver(device)

    theta_b = solver.solve_batched(X, Y, alphas, scaler=None, chunk_size=64)
    for s in range(S):
        theta_s = solver.solve(X[s], Y[s], alphas, scaler=None, chunk_size=64)
        err = (theta_b[:, s].cpu() - theta_s.cpu()).abs().max().item()
        assert err < 1e-10, f"series {s}: batched vs single {err}"
    return "solve_batched matches per-series solve"


def _selftest_mixed_precision():
    """mixed precision (fp32 Gram matmuls, fp64 accumulate/solve) must track the
    fp64 solution closely and pick the same alpha on a realistic problem."""
    import torch
    from optuna_ridge import RidgeSolver, LocalNormScaler, StandardStrategy

    torch.manual_seed(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, L, H = 4000, 64, 8
    # float32 inputs, like the real (standardized) datasets
    t = torch.arange(N + L + H, dtype=torch.float32)
    series = torch.sin(t / 17) + 0.1 * torch.randn_like(t)
    wins = series.unfold(0, L + H, 1)[:N]
    X, Y = wins[:, :L], wins[:, L:]
    alphas = torch.logspace(-6, 4, 21, device=device)

    n_tr = int(N * 0.7)
    X_tr, Y_tr, X_va, Y_va = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]

    s64 = RidgeSolver(device, precision="fp64")
    smx = RidgeSolver(device, precision="mixed")
    scaler64 = LocalNormScaler(StandardStrategy(), L, L)
    scalermx = LocalNormScaler(StandardStrategy(), L, L)

    t64 = s64.solve(X_tr, Y_tr, alphas, scaler=scaler64, chunk_size=1024)
    tmx = smx.solve(X_tr, Y_tr, alphas, scaler=scalermx, chunk_size=1024)
    m64 = s64.val_mse(X_va, Y_va, t64, scaler=scaler64, chunk_size=1024)
    mmx = smx.val_mse(X_va, Y_va, tmx, scaler=scalermx, chunk_size=1024)

    # Best achievable val MSE must match; the chosen alpha must be equally
    # good under fp64 (argmin can hop between statistically indistinguishable
    # neighbors on flat regions of the alpha curve — the run-level gate
    # allows 5% of such hops)
    rel = abs(m64.min().item() - mmx.min().item()) / m64.min().item()
    assert rel < 1e-3, f"best val MSE rel diff {rel}"
    quality_gap = (m64[mmx.argmin()].item() - m64.min().item()) / m64.min().item()
    assert quality_gap < 1e-3, f"mixed-chosen alpha is worse under fp64 by {quality_gap}"
    return (f"mixed tracks fp64: best-MSE rel diff {rel:.1e}, "
            f"chosen-alpha quality gap {quality_gap:.1e}")


SELFTESTS = [
    _selftest_windowing,
    _selftest_ridge_closed_form,
    _selftest_local_norm_solve,
    _selftest_pooled_3d_solve,
    _selftest_fused_val_mse,
    _selftest_solve_batched_per_series,
    _selftest_mixed_precision,
]


def cmd_selftest(args):
    sys.path.insert(0, REPO_ROOT)
    failures = 0
    for fn in SELFTESTS:
        try:
            msg = fn()
            print(f"  PASS {fn.__name__}: {msg}")
        except Exception as e:
            failures += 1
            print(f"  FAIL {fn.__name__}: {e!r}")
    print(f"{len(SELFTESTS) - failures}/{len(SELFTESTS)} self-tests passed")
    sys.exit(1 if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a reduced benchmark slice")
    p_run.add_argument("--dataset", choices=sorted(DATASETS), default="etth1")
    p_run.add_argument("--out", required=True, help="output directory")
    p_run.add_argument("--n_trials", type=int, default=10)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--horizon_subset", type=str, default="0,14,29")
    p_run.add_argument("--precision", choices=["fp64", "mixed", "fp32"], default="fp64")
    p_run.add_argument("--noise", choices=["none", "search"], default="none",
                       help="'none' fixes augmentation off (required for strict parity)")
    p_run.add_argument("--profile", action="store_true")
    p_run.add_argument("extra_args", nargs="*",
                       help="extra args forwarded to optuna_ridge.py (prefix with --)")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="compare two run output directories")
    p_cmp.add_argument("dir_a")
    p_cmp.add_argument("dir_b")
    p_cmp.add_argument("--tol", type=float, default=1e-9,
                       help="max allowed relative diff in val MSE (default: strict fp64 parity)")
    p_cmp.add_argument("--alpha_match", type=float, default=1.0,
                       help="required fraction of param-matched trials with identical chosen "
                            "alpha (use 0.95 for the mixed-precision gate)")
    p_cmp.add_argument("--msepa_tol", type=float, default=None,
                       help="max allowed relative diff in the full per-alpha MSE vector "
                            "(default: same as --tol; extreme alphas are numerically "
                            "sensitive, so the mixed gate typically relaxes this)")
    p_cmp.set_defaults(func=cmd_compare)

    p_st = sub.add_parser("selftest", help="run numerical unit checks")
    p_st.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
