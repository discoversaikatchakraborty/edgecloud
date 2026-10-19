from pathlib import Path

from core import run_all

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    summary, ci, ablation, sweep = run_all(root / "results")
    cols = ["macro_f1", "errors", "mean_bytes", "p95_latency_s", "review_rate"]
    print("Completed controlled benchmark")
    print(summary.groupby(["scenario", "method"])[cols].mean().round(4).to_string())
