import time
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# GPU side
from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as csp

# sparse-kappa
from sparse_kappa import cond_estimate

# baseline
import torch


# ----------------------------
# Utilities
# ----------------------------
def now() -> float:
    return time.perf_counter()


def rel_err(est: float, ref: float) -> float:
    if ref == 0 or not np.isfinite(ref):
        return float("nan")
    return abs(est - ref) / abs(ref)


def to_torch_csr(A_np: np.ndarray, density: float, seed: int) -> csp.csr_matrix:
    """
    Create a PyTorch CSR sparse matrix with approximately given density,
    using A_np values but sparsified by a random mask.
    """
    rng = np.random.default_rng(seed)
    n = A_np.shape[0]
    mask = rng.random((n, n)) < density
    A_sparse = A_np * mask
    A_cp = cp.asarray(A_sparse)
    return csp.csr_matrix(A_cp)


def make_random_matrix(
    n: int,
    dist: str,
    seed: int,
    shift: float,
    symmetric: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dist == "normal":
        A = rng.standard_normal((n, n)).astype(np.float64)
    elif dist == "uniform":
        A = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float64)
    else:
        raise ValueError(f"unknown dist: {dist}")

    if symmetric:
        A = 0.5 * (A + A.T)

    # Tikhonov / diagonal shift to avoid near-singularity
    A = A + shift * np.eye(n, dtype=A.dtype)
    return A


def torch_cond_2(A_np: np.ndarray, device: str = "cpu") -> float:
    A = torch.from_numpy(A_np).to(device=device, dtype=torch.float64)
    # svdvals returns singular values in descending order
    s = torch.linalg.svdvals(A)
    smax = float(s[0].cpu())
    smin = float(s[-1].cpu())
    if smin <= 0 or not math.isfinite(smin):
        return float("inf")
    return smax / smin


def torch_cond_1(A_np: np.ndarray, device: str = "cpu") -> float:
    """
    Baseline for 1-norm condition number:
      kappa_1(A) = ||A||_1 * ||A^{-1}||_1
    computed explicitly via inverse (feasible for n<=500).
    """
    A = torch.from_numpy(A_np).to(device=device, dtype=torch.float64)
    normA1 = float(torch.linalg.matrix_norm(A, ord=1).cpu())
    Ainv = torch.linalg.inv(A)
    normAinv1 = float(torch.linalg.matrix_norm(Ainv, ord=1).cpu())
    return normA1 * normAinv1


@dataclass
class RunResult:
    n: int
    density: float
    symmetric: bool
    norm: int
    method: str
    solver: Optional[str]
    est: float
    ref: float
    rel_err: float
    sec: float
    ok: bool
    error: Optional[str] = None


# ----------------------------
# Main benchmark
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--n-min", type=int, default=200)
    ap.add_argument("--n-max", type=int, default=500)
    ap.add_argument("--density", type=float, default=0.05, help="sparsity density for CSR")
    ap.add_argument("--shift", type=float, default=5.0, help="diagonal shift to reduce singularity risk")
    ap.add_argument("--dist", type=str, default="normal", choices=["normal", "uniform"])
    ap.add_argument("--symmetric", action="store_true", help="generate symmetric A (then A^T=A)")
    ap.add_argument("--torch-device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", type=str, default="bench_results.jsonl")
    args = ap.parse_args()

    # sparse-kappa methods (based on repo code/README strings)
    methods_1 = [
        "hager",
        "hager-higham",      # block higham-tisseur in code
        "power",
        "oettli-prager",
        "monte-carlo",
    ]
    methods_2 = [
        "power",
        "lanczos",
        "lanczos_unsym",
        "golub-kahan",
        "svds",
        "eigsh",
        "lobpcg",
    ]

    # solvers in docs/solvers.rst
    solvers = ["auto", "lu", "lsmr", "cg", "gmres", "direct"]

    results: List[RunResult] = []

    # warm up GPU
    cp.cuda.Device().synchronize()

    for t in range(args.trials):
        n = int(np.random.default_rng(1234 + t).integers(args.n_min, args.n_max + 1))
        seed = 2026_000 + t

        A_np = make_random_matrix(
            n=n,
            dist=args.dist,
            seed=seed,
            shift=args.shift,
            symmetric=args.symmetric,
        )

        # Build sparse GPU matrix (CSR)
        A_csr = to_torch_csr(A_np, density=args.density, seed=seed + 1)

        # ---------------- baseline refs (CPU or CUDA torch) ----------------
        # Baselines measured on torch-device; note: time not used as key metric here
        ref1 = None
        ref2 = None
        try:
            ref2 = torch_cond_2(A_np, device=args.torch_device)
        except Exception as e:
            ref2 = float("nan")
            print(f"[trial {t}] torch cond2 failed: {e}")

        try:
            ref1 = torch_cond_1(A_np, device=args.torch_device)
        except Exception as e:
            ref1 = float("nan")
            print(f"[trial {t}] torch cond1 failed: {e}")

        # ---------------- sparse-kappa: 1-norm ----------------
        for method in methods_1:
            for solver in solvers:
                t0 = now()
                ok = True
                est = float("nan")
                err = None
                try:
                    # Ensure all GPU ops finish before/after timing
                    cp.cuda.Device().synchronize()
                    t1 = now()

                    # NOTE: method/solver kwargs may vary by method; sparse-kappa accepts extra kwargs.
                    # For fair-ish comparison, we keep defaults; you can tune max_iter, block_size, etc.
                    est = cond_estimate(
                        A_csr,
                        norm=1,
                        method=method,
                        solver=solver,
                        return_dict=False,
                    )

                    cp.cuda.Device().synchronize()
                    t2 = now()
                    sec = t2 - t1
                except Exception as e:
                    cp.cuda.Device().synchronize()
                    sec = now() - t0
                    ok = False
                    err = repr(e)

                results.append(
                    RunResult(
                        n=n,
                        density=args.density,
                        symmetric=args.symmetric,
                        norm=1,
                        method=method,
                        solver=solver,
                        est=float(est) if ok else float("nan"),
                        ref=float(ref1) if ref1 is not None else float("nan"),
                        rel_err=rel_err(float(est), float(ref1)) if (ok and ref1 is not None) else float("nan"),
                        sec=float(sec),
                        ok=ok,
                        error=err,
                    )
                )

        # ---------------- sparse-kappa: 2-norm ----------------
        for method in methods_2:
            # 2-norm methods in this repo do not take 'solver' (most are matvec/eigensolver based),
            # but passing solver could error. So we keep solver=None and call without solver.
            t0 = now()
            ok = True
            est = float("nan")
            err = None
            try:
                cp.cuda.Device().synchronize()
                t1 = now()

                est = cond_estimate(
                    A_csr,
                    norm=2,
                    method=method,
                    return_dict=False,
                )

                cp.cuda.Device().synchronize()
                t2 = now()
                sec = t2 - t1
            except Exception as e:
                cp.cuda.Device().synchronize()
                sec = now() - t0
                ok = False
                err = repr(e)

            results.append(
                RunResult(
                    n=n,
                    density=args.density,
                    symmetric=args.symmetric,
                    norm=2,
                    method=method,
                    solver=None,
                    est=float(est) if ok else float("nan"),
                    ref=float(ref2) if ref2 is not None else float("nan"),
                    rel_err=rel_err(float(est), float(ref2)) if (ok and ref2 is not None) else float("nan"),
                    sec=float(sec),
                    ok=ok,
                    error=err,
                )
            )

        print(f"[trial {t+1}/{args.trials}] n={n} done, collected {len(results)} rows so far.")

    # write JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # quick summary (group by norm/method/solver)
    def key(r: RunResult):
        return (r.norm, r.method, r.solver or "")

    groups: Dict[Tuple[Any, ...], List[RunResult]] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)

    print("\n=== SUMMARY (mean over successful runs) ===")
    for k, rows in sorted(groups.items()):
        ok_rows = [x for x in rows if x.ok and np.isfinite(x.est) and np.isfinite(x.ref)]
        if not ok_rows:
            print(f"{k}: no successful comparable runs")
            continue
        mean_t = float(np.mean([x.sec for x in ok_rows]))
        med_t = float(np.median([x.sec for x in ok_rows]))
        mean_e = float(np.mean([x.rel_err for x in ok_rows if np.isfinite(x.rel_err)]))
        med_e = float(np.median([x.rel_err for x in ok_rows if np.isfinite(x.rel_err)]))
        print(f"{k}: time mean={mean_t:.4f}s med={med_t:.4f}s | rel_err mean={mean_e:.3e} med={med_e:.3e} | n={len(ok_rows)}")


if __name__ == "__main__":
    main()