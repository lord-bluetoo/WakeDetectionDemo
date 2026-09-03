from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\ccs\Desktop\硕士期间(2025-2028)\科研工作\尾迹检测\experiment_results")

runs = {
    "Baseline": ROOT / "pilot20_s42_baseline",
    "Geometry": ROOT / "pilot20_s42_geometry",
}

summary = []

for method, run in runs.items():
    df = pd.read_csv(run / "results.csv")
    df.columns = df.columns.str.strip()

    metric_columns = [c for c in df.columns if c.startswith("metrics/")]
    map5095_column = next(c for c in metric_columns if "mAP50-95" in c)
    map50_column = next(
        c for c in metric_columns
        if "mAP50" in c and "mAP50-95" not in c
    )
    precision_column = next(c for c in metric_columns if "precision" in c)
    recall_column = next(c for c in metric_columns if "recall" in c)

    best_index = df[map5095_column].idxmax()
    best = df.loc[best_index]

    summary.append({
        "方法": method,
        "最佳epoch": int(best["epoch"]) + 1,
        "Precision": float(best[precision_column]),
        "Recall": float(best[recall_column]),
        "mAP50": float(best[map50_column]),
        "mAP50-95": float(best[map5095_column]),
    })

summary = pd.DataFrame(summary)
print(summary.to_string(index=False))

baseline = summary.loc[summary["方法"] == "Baseline", "mAP50-95"].iloc[0]
geometry = summary.loc[summary["方法"] == "Geometry", "mAP50-95"].iloc[0]

print(f"Geometry 相对 Baseline 的 mAP50-95 变化：{geometry - baseline:+.4f}")