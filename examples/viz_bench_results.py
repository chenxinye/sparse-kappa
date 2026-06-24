import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except Exception:
    sns = None


def read_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {line_no}: {e}")
    if not rows:
        raise RuntimeError("No rows found in JSONL.")
    df = pd.DataFrame(rows)
    return df


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric
    for col in ["n", "density", "norm", "est", "ref", "rel_err", "sec"]:
        if col in df.columns:
            df[col] = safe_float_series(df[col])

    if "ok" in df.columns:
        df["ok"] = df["ok"].astype(bool)
    else:
        df["ok"] = True

    # A run is "comparable" if ok and finite rel_err and finite sec
    df["finite_err"] = np.isfinite(df["rel_err"].to_numpy())
    df["finite_sec"] = np.isfinite(df["sec"].to_numpy())
    df["comparable"] = df["ok"] & df["finite_err"] & df["finite_sec"]

    # Normalize solver column (norm=2 has solver=None)
    if "solver" not in df.columns:
        df["solver"] = ""
    df["solver"] = df["solver"].fillna("")

    group_cols = ["norm", "method", "solver"]

    def agg(g: pd.DataFrame) -> pd.Series:
        total = len(g)
        ok = int(g["ok"].sum())
        comp = int(g["comparable"].sum())

        gg = g[g["comparable"]]
        out = {
            "rows_total": total,
            "rows_ok": ok,
            "rows_comparable": comp,
            "success_rate_ok": ok / total if total else np.nan,
            "success_rate_comparable": comp / total if total else np.nan,
            "n_mean": float(gg["n"].mean()) if comp else np.nan,
            "time_mean_s": float(gg["sec"].mean()) if comp else np.nan,
            "time_median_s": float(gg["sec"].median()) if comp else np.nan,
            "time_p90_s": float(gg["sec"].quantile(0.90)) if comp else np.nan,
            "relerr_mean": float(gg["rel_err"].mean()) if comp else np.nan,
            "relerr_median": float(gg["rel_err"].median()) if comp else np.nan,
            "relerr_p90": float(gg["rel_err"].quantile(0.90)) if comp else np.nan,
        }
        return pd.Series(out)

    summary = df.groupby(group_cols, dropna=False).apply(agg).reset_index()
    summary = summary.sort_values(["norm", "method", "solver"], ascending=True).reset_index(drop=True)
    return summary


def write_markdown_table(df: pd.DataFrame, path: str, max_rows: Optional[int] = None) -> None:
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    md = out.to_markdown(index=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md + "\n")


def plot_norm1(df: pd.DataFrame, outdir: str) -> None:
    d1 = df[(df["norm"] == 1)].copy()
    if d1.empty:
        return
    d1["solver"] = d1["solver"].fillna("")
    d1["comparable"] = d1["comparable"].astype(bool)
    d1c = d1[d1["comparable"]].copy()
    if d1c.empty:
        return

    # Time by method & solver
    plt.figure(figsize=(14, 6))
    if sns:
        sns.boxplot(data=d1c, x="method", y="sec", hue="solver")
        sns.stripplot(data=d1c, x="method", y="sec", hue="solver",
                      dodge=True, alpha=0.25, size=2, linewidth=0)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    else:
        # Fallback: simple scatter
        methods = sorted(d1c["method"].unique())
        for i, m in enumerate(methods):
            y = d1c.loc[d1c["method"] == m, "sec"]
            x = np.full(len(y), i)
            plt.scatter(x, y, s=8, alpha=0.5)
        plt.xticks(range(len(methods)), methods, rotation=30, ha="right")
    plt.yscale("log")
    plt.title("1-norm: runtime by method (and solver)")
    plt.xlabel("method")
    plt.ylabel("seconds (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_by_method_norm1.png"), dpi=200)
    plt.close()

    # Error by method & solver
    plt.figure(figsize=(14, 6))
    if sns:
        sns.boxplot(data=d1c, x="method", y="rel_err", hue="solver")
        sns.stripplot(data=d1c, x="method", y="rel_err", hue="solver",
                      dodge=True, alpha=0.25, size=2, linewidth=0)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    else:
        methods = sorted(d1c["method"].unique())
        for i, m in enumerate(methods):
            y = d1c.loc[d1c["method"] == m, "rel_err"]
            x = np.full(len(y), i)
            plt.scatter(x, y, s=8, alpha=0.5)
        plt.xticks(range(len(methods)), methods, rotation=30, ha="right")
    plt.yscale("log")
    plt.title("1-norm: relative error by method (and solver)")
    plt.xlabel("method")
    plt.ylabel("relative error (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "err_by_method_norm1.png"), dpi=200)
    plt.close()


def plot_norm2(df: pd.DataFrame, outdir: str) -> None:
    d2 = df[(df["norm"] == 2)].copy()
    if d2.empty:
        return
    d2["comparable"] = d2["comparable"].astype(bool)
    d2c = d2[d2["comparable"]].copy()
    if d2c.empty:
        return

    # Time by method
    plt.figure(figsize=(12, 6))
    if sns:
        sns.boxplot(data=d2c, x="method", y="sec")
        sns.stripplot(data=d2c, x="method", y="sec", alpha=0.25, size=2, linewidth=0)
    else:
        methods = sorted(d2c["method"].unique())
        for i, m in enumerate(methods):
            y = d2c.loc[d2c["method"] == m, "sec"]
            x = np.full(len(y), i)
            plt.scatter(x, y, s=8, alpha=0.5)
        plt.xticks(range(len(methods)), methods, rotation=30, ha="right")
    plt.yscale("log")
    plt.title("2-norm: runtime by method")
    plt.xlabel("method")
    plt.ylabel("seconds (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_by_method_norm2.png"), dpi=200)
    plt.close()

    # Error by method
    plt.figure(figsize=(12, 6))
    if sns:
        sns.boxplot(data=d2c, x="method", y="rel_err")
        sns.stripplot(data=d2c, x="method", y="rel_err", alpha=0.25, size=2, linewidth=0)
    else:
        methods = sorted(d2c["method"].unique())
        for i, m in enumerate(methods):
            y = d2c.loc[d2c["method"] == m, "rel_err"]
            x = np.full(len(y), i)
            plt.scatter(x, y, s=8, alpha=0.5)
        plt.xticks(range(len(methods)), methods, rotation=30, ha="right")
    plt.yscale("log")
    plt.title("2-norm: relative error by method")
    plt.xlabel("method")
    plt.ylabel("relative error (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "err_by_method_norm2.png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="bench_results.jsonl")
    ap.add_argument("--outdir", default="bench_viz")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = read_jsonl(args.inp)

    # Ensure expected columns exist
    for col in ["norm", "method", "solver", "sec", "rel_err", "ok", "n"]:
        if col not in df.columns:
            df[col] = np.nan

    # Build summary
    summary = make_summary(df)
    summary_csv = os.path.join(args.outdir, "summary.csv")
    summary_md = os.path.join(args.outdir, "summary.md")
    summary.to_csv(summary_csv, index=False)
    write_markdown_table(summary, summary_md)

    # Save cleaned full table too
    df_out = df.copy()
    df_out.to_csv(os.path.join(args.outdir, "runs.csv"), index=False)

    # Plots
    plot_norm1(df_out, args.outdir)
    plot_norm2(df_out, args.outdir)

    print(f"Wrote:\n  {summary_csv}\n  {summary_md}\n  {os.path.join(args.outdir, 'runs.csv')}")
    print(f"Figures (if data available):")
    for fn in [
        "time_by_method_norm1.png",
        "err_by_method_norm1.png",
        "time_by_method_norm2.png",
        "err_by_method_norm2.png",
    ]:
        p = os.path.join(args.outdir, fn)
        if os.path.exists(p):
            print("  " + p)


if __name__ == "__main__":
    main()