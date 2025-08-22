#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze & compare YOLO11 classification runs (no MLflow).
- Reads args.yaml + results.csv under each run directory.
- Builds a summary table of hyperparams + metrics.
- Generates an interactive HTML report with Plotly:
  * Top-1 Accuracy vs Epoch (per run)
  * Top-5 Accuracy vs Epoch (per run, if available)
  * Loss vs Epoch (per run, if available)
  * Hyperparameter/metrics summary table
- Exports summary.csv for spreadsheet work.

Usage:
    python analyze_yolo_classification_runs.py --runs-dir path/to/runs --out-dir analysis_output
"""

import os
import sys
import glob
import argparse
import yaml
import pandas as pd
from typing import Dict, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------
# Helpers
# -------------------------------

# Common column aliases seen in YOLO classification results
ACC1_ALIASES = [
    "top1_acc", "top1", "accuracy", "acc", "val/accuracy",
    "metrics/accuracy", "metrics/top1", "val/top1", "cls_acc", "metrics/accuracy_top1"
]
ACC5_ALIASES = [
    "top5_acc", "top5", "metrics/top5", "val/top5", "cls_top5", "metrics/accuracy_top5"
]
LOSS_ALIASES = [
    "val/loss", "loss", "train/loss", "metrics/loss", "cls_loss"
]
# Learning rate per group in results.csv
LR_GROUP_ALIASES = ["lr/pg0", "lr/pg1", "lr/pg2", "pg0", "pg1", "pg2"]

# Training time column in results.csv
TIME_ALIASES = ["time", "training_time", "elapsed_time"]
LR_ALIASES = ["lr/pg0", "lr/pg1", "lr/pg2", "lr1", "lr2", "pg0", "pg1", "pg2"]
EPOCH_ALIASES = ["epoch", "epochs", "Epoch"]

def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df (exact match)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_run(run_path: str) -> Optional[Dict]:
    """Load a run directory containing args.yaml and results.csv."""
    args_path = os.path.join(run_path, "args.yaml")
    results_path = os.path.join(run_path, "results.csv")

    if not (os.path.isfile(args_path) and os.path.isfile(results_path)):
        return None

    with open(args_path, "r") as f:
        args = yaml.safe_load(f) or {}

    df = pd.read_csv(results_path)
    if df.empty:
        return None

    # Normalize epoch column if missing by using row index
    epoch_col = first_present(df, EPOCH_ALIASES)
    if epoch_col is None:
        df = df.reset_index().rename(columns={"index": "epoch"})
        epoch_col = "epoch"

    # Ensure integer epochs where possible
    try:
        df[epoch_col] = df[epoch_col].astype(int)
    except Exception:
        pass

    return {
        "run_name": os.path.basename(run_path.rstrip(os.sep)),
        "args": args,
        "df": df,
        "epoch_col": epoch_col
    }

def format_float(x, ndigits=4):
    try:
        return float(x)
    except Exception:
        return None

# -------------------------------
# Core
# -------------------------------

def analyze(runs_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    run_dirs = [p for p in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]
    runs = []
    for p in run_dirs:
        r = load_run(p)
        if r is not None:
            runs.append(r)

    if not runs:
        print("No valid runs found (need args.yaml + results.csv in each subfolder).", file=sys.stderr)
        sys.exit(1)

    # Build summaries + long-form data for plots
    summaries = []
    acc1_series = []
    acc5_series = []
    loss_series = []

    for run in runs:
        run_name = run["run_name"]
        args = run["args"]
        df = run["df"]
        epoch_col = run["epoch_col"]

        acc1_col = first_present(df, ACC1_ALIASES)
        # Learning rate per parameter group
        lr0_col = first_present(df, ["lr/pg0", "pg0"])
        lr1_col = first_present(df, ["lr/pg1", "pg1"])
        lr2_col = first_present(df, ["lr/pg2", "pg2"])

        lr0_val = format_float(df[lr0_col].iloc[-1]) if lr0_col else None
        lr1_val = format_float(df[lr1_col].iloc[-1]) if lr1_col else None
        lr2_val = format_float(df[lr2_col].iloc[-1]) if lr2_col else None

        # Training time
        time_col = first_present(df, TIME_ALIASES)
        total_time = format_float(df[time_col].iloc[-1]) if time_col else None
        acc5_col = first_present(df, ACC5_ALIASES)
        loss_col = first_present(df, LOSS_ALIASES)

        # Compute best/last metrics
        best_acc1, best_acc1_epoch, last_acc1 = None, None, None
        best_acc5, best_acc5_epoch, last_acc5 = None, None, None
        last_loss = None
        # Differences last - best
        diff_top1 = None
        diff_top5 = None
        diff_loss = None
        if acc1_col:
            best_idx = df[acc1_col].idxmax()
            best_acc1 = format_float(df.loc[best_idx, acc1_col])
            best_acc1_epoch = int(df.loc[best_idx, epoch_col])
            last_acc1 = format_float(df[acc1_col].iloc[-1])
            # Differences last - best
            diff_top1 = format_float(last_acc1 - best_acc1)
            # Store curve
            acc1_series.append(
                pd.DataFrame({
                    "epoch": df[epoch_col],
                    "value": df[acc1_col],
                    "metric": "Top-1 Accuracy",
                    "run": run_name
                })
            )

        if acc5_col:
            best_idx5 = df[acc5_col].idxmax()
            best_acc5 = format_float(df.loc[best_idx5, acc5_col])
            best_acc5_epoch = int(df.loc[best_idx5, epoch_col])
            last_acc5 = format_float(df[acc5_col].iloc[-1])
            # Differences last - best
            diff_top5 = format_float(last_acc5 - best_acc5)
            # Store curve
            acc5_series.append(
                pd.DataFrame({
                    "epoch": df[epoch_col],
                    "value": df[acc5_col],
                    "metric": "Top-5 Accuracy",
                    "run": run_name
                })
            )

        if loss_col:
            last_loss = format_float(df[loss_col].iloc[-1])
            # Differences last - best (requires best_idx from acc1_col)
            if acc1_col:
                diff_loss = format_float(last_loss - df.loc[best_idx, loss_col])
            loss_series.append(
                pd.DataFrame({
                    "epoch": df[epoch_col],
                    "value": df[loss_col],
                    "metric": "Loss",
                    "run": run_name
                })
            )

        # Summaries (hyperparams vary by YOLO version; grab common ones safely)
        summaries.append({
            "run": run_name,
            "epochs_cfg": args.get("epochs"),
            "patience": args.get("patience"),
            "batch_size": args.get("batch") or args.get("batch_size"),
            "learning_rate": args.get("lr0") or args.get("learning_rate"),
            "optimizer": args.get("optimizer"),
            "seed": args.get("seed"),
            "imgsz": args.get("imgsz") or args.get("imgsz_max") or args.get("imgsz_train"),
            "model": args.get("model"),
            "dataset": args.get("data") or args.get("dataset"),
            "best_top1": best_acc1,
            "best_top1_epoch": best_acc1_epoch,
            "final_top1": last_acc1,
            "best_top5": best_acc5,
            "best_top5_epoch": best_acc5_epoch,
            "final_top5": last_acc5,
            "final_loss": last_loss,
            "lr_pg0": lr0_val,
            "lr_pg1": lr1_val,
            "lr_pg2": lr2_val,
            "total_time": total_time,
            "diff_top1": diff_top1,
            "diff_top5": diff_top5,
            "diff_loss": diff_loss,
            "hsv_s": args.get("hsv_s"),
            "hsv_v": args.get("hsv_v"),
            "fliplr": args.get("fliplr"),
            "degrees": args.get("degrees"),
            "scale": args.get("scale"),
            "shear": args.get("shear"),
            "perspevtive": args.get("perspective"),
            "mosaic": args.get("mosaic"),
            "erasing": args.get("erasing")
        })

    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # -------------------------------
    # Build interactive HTML
    # -------------------------------
    # Create figures (Top-1, Top-5, Loss) only if we have data
    def line_figure(series_list: List[pd.DataFrame], title: str, y_title: str) -> Optional[go.Figure]:
        if not series_list:
            return None
        fig = go.Figure()
        for df in series_list:
            run_name = df["run"].iloc[0]
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df["value"],
                mode="lines+markers", name=run_name,
                hovertemplate="Epoch=%{x}<br>Value=%{y:.4f}<br>Run=%{fullData.name}<extra></extra>"
            ))
        fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title=y_title, hovermode="x unified")
        return fig

    acc1_fig = line_figure(acc1_series, "Top-1 Accuracy vs Epoch", "Top-1 Accuracy")
    acc5_fig = line_figure(acc5_series, "Top-5 Accuracy vs Epoch", "Top-5 Accuracy")
    loss_fig = line_figure(loss_series, "Loss vs Epoch", "Loss")

    # Compose single HTML
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write("<title>YOLO Classification Run Comparison</title>")
        # Load Plotly once via CDN for a lightweight file; set to 'cdn' per figure where needed.
        f.write("</head><body style='max-width:1200px;margin:40px auto;font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif;'>")
        f.write("<h1>YOLO Classification Run Comparison</h1>")

        f.write("<h2>Hyperparameters & Metrics Summary</h2>")
        # Basic styling for table readability
        f.write("""
        <style>
        table {border-collapse: collapse; width: 100%;}
        th, td {border: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 14px;}
        th {background: #f3f4f6;}
        </style>
        """)
        # Format floats to 4 decimals in HTML
        f.write(summary_df.to_html(index=False, na_rep="", float_format=lambda x: f"{x:.4f}"))
        
        # Table for last vs best differences
        diff_cols = ["run", "diff_top1", "diff_top5", "diff_loss"]
        diff_df = summary_df[diff_cols]

        f.write("<h3>Last vs Best Epoch Differences</h3>")
        f.write("""
        <style>
        table.diff {border-collapse: collapse; width: 50%;}
        table.diff th, table.diff td {border: 1px solid #e5e7eb; padding: 6px; text-align: left; font-size: 14px;}
        table.diff th {background: #f9fafb;}
        </style>
        """)
        f.write(diff_df.to_html(index=False, na_rep="", float_format=lambda x: f"{x:.4f}", classes="diff"))

        # Insert figures
        if acc1_fig:
            f.write("<h2 style='margin-top:32px;'>Top-1 Accuracy</h2>")
            f.write(acc1_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        else:
            f.write("<p><em>No Top-1 accuracy columns were found in results.csv files.</em></p>")

        if acc5_fig:
            f.write("<h2 style='margin-top:32px;'>Top-5 Accuracy</h2>")
            f.write(acc5_fig.to_html(full_html=False, include_plotlyjs=False))
        else:
            f.write("<p><em>No Top-5 accuracy columns were found in results.csv files.</em></p>")

        if loss_fig:
            f.write("<h2 style='margin-top:32px;'>Loss</h2>")
            f.write(loss_fig.to_html(full_html=False, include_plotlyjs=False))
        else:
            f.write("<p><em>No loss columns were found in results.csv files.</em></p>")

        f.write("<hr><p style='color:#6b7280;font-size:12px;'>Generated by analyze_yolo_classification_runs.py</p>")
        f.write("</body></html>")

    print(f"✅ Report: {html_path}")
    print(f"✅ Summary CSV: {summary_csv}")

# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare YOLO11 classification runs without MLflow.")
    parser.add_argument("--runs-dir", type=str, required=True,
                        help="Directory containing run subfolders (each with args.yaml and results.csv).")
    parser.add_argument("--out-dir", type=str, default="analysis_output",
                        help="Output directory for report.html and summary.csv.")
    args = parser.parse_args()

    analyze(args.runs_dir, args.out_dir)

if __name__ == "__main__":
    main()