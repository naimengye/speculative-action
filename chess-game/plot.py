"""
Unified plotting script: time-token scatter plots or combined bar plots.

Subcommands:
  time-token  Plot time saved vs token usage (optionally with confidence policy).
  bars        Combined bar charts (accuracy + time saved) from multiple analysis dirs.
"""

import argparse
import csv
import glob
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# -------- time-token plot (from plot_time_token.py) --------

def _load_csv_data(csv_path: str, skip_target_10: bool = True):
    """Load (time_saved, token_wasted, num_pred, target_step) from CSV. Returns dict keyed by (num_pred, target_step)."""
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            time_saved = float(row[6])
            tokens_saved = float(row[9])
            num_pred = int(row[3])
            target_step = int(row[1])
            if skip_target_10 and target_step == 10:
                continue
            token_wasted = -tokens_saved
            key = (num_pred, target_step)
            if key not in data:
                data[key] = (time_saved, token_wasted, num_pred, target_step)
    return data


def create_time_token_plot(base_path: str, with_confidence: bool = False) -> None:
    """
    Create scatter plot: time (saved or spent) vs token wasted.
    If with_confidence, reads both CSVs and plots with step markers and confidence policy as num_pred=4.
    """
    csv_path = f"{base_path}/analysis_results.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping...")
        return

    data = _load_csv_data(csv_path)

    if with_confidence:
        confidence_csv_path = f"{base_path}/analysis_results_with_confidence.csv"
        if not os.path.exists(confidence_csv_path):
            print(f"Warning: {confidence_csv_path} not found, skipping...")
            return
        confidence_data = {}
        with open(confidence_csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                time_saved = float(row[6])
                tokens_saved = float(row[9])
                num_pred = 4
                target_step = int(row[1])
                if target_step == 10:
                    continue
                token_wasted = -tokens_saved
                key = target_step
                if key not in confidence_data:
                    confidence_data[key] = (time_saved, token_wasted, num_pred, target_step)
        for (time_saved, token_wasted, num_pred, target_step) in confidence_data.values():
            key = (num_pred, target_step)
            if key not in data:
                data[key] = (time_saved, token_wasted, num_pred, target_step)

    time_saved_list = []
    token_wasted_list = []
    number_predictions = []
    target_steps_list = []
    for (time_saved, token_wasted, num_pred, target_step) in data.values():
        time_saved_list.append(time_saved)
        token_wasted_list.append(token_wasted)
        number_predictions.append(num_pred)
        target_steps_list.append(target_step)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {1: "#2E86AB", 2: "#A23B72", 3: "#F18F01", 4: "#4CAF50"}
    color_list = [colors.get(pred, "#333") for pred in number_predictions]

    if with_confidence:
        time_spent_list = [100 - t for t in time_saved_list]
        markers = {20: "o", 30: "s", 40: "p", 50: "D"}
        ax.grid(True, alpha=0.5, linestyle="-", linewidth=0.8)
        ax.set_facecolor("white")

        step_groups = {}
        for i, (x, y, num_pred, target) in enumerate(zip(time_spent_list, token_wasted_list, number_predictions, target_steps_list)):
            if num_pred <= 3:
                if target not in step_groups:
                    step_groups[target] = []
                step_groups[target].append((num_pred, x, y))

        for target_step in sorted(step_groups.keys()):
            points = step_groups[target_step]
            points.sort(key=lambda p: p[0])
            x_vals = [p[1] for p in points]
            y_vals = [p[2] for p in points]
            ax.plot(x_vals, y_vals, color="gray", linewidth=1.5, alpha=0.5, linestyle="-")

        for i, (x, y, num_pred, target) in enumerate(zip(time_spent_list, token_wasted_list, number_predictions, target_steps_list)):
            s = 200 if target == 40 else (110 if target == 50 else 150)
            ax.scatter(
                x, y, c=colors.get(num_pred, "#333"), s=s, alpha=1.0, zorder=3,
                marker=markers.get(target, "o"), edgecolors="white", linewidths=1.2,
            )

        speculation_groups = {}
        for x, y, num_pred in zip(time_spent_list, token_wasted_list, number_predictions):
            if num_pred not in speculation_groups:
                speculation_groups[num_pred] = {"x": [], "y": []}
            speculation_groups[num_pred]["x"].append(x)
            speculation_groups[num_pred]["y"].append(y)

        for num_pred, pts in speculation_groups.items():
            x_vals = np.array(pts["x"])
            y_vals = np.array(pts["y"])
            label_text = "confidence-based\nthreshold policy" if num_pred == 4 else f"{num_pred} speculation{'s' if num_pred > 1 else ''}"
            if num_pred == 1:
                label_x, label_y = np.mean(x_vals), np.min(y_vals) - 7
            elif num_pred == 2:
                label_x, label_y = np.max(x_vals) + 4, np.max(y_vals)
            elif num_pred == 3:
                label_x, label_y = np.mean(x_vals) - 2, np.mean(y_vals)
            else:
                label_x, label_y = np.mean(x_vals), np.max(y_vals) + 4
            ax.text(label_x, label_y, label_text, fontsize=14, fontweight="semibold",
                    ha="center", va="bottom", color=colors.get(num_pred, "#333"), zorder=5)

        ax.set_xlabel("Percentage of Time Spent (%)", fontsize=18)
        ax.set_ylabel("Percentage of Extra Tokens Used (%)", fontsize=18)
        legend_elements = [
            plt.Line2D([0], [0], marker=markers[s], color="w", markerfacecolor="gray",
                       markeredgecolor="black", markersize=10, label=f"{s} steps")
            for s in sorted(markers.keys())
        ]
        ax.legend(handles=legend_elements, title="Target Steps", loc="lower left",
                  fontsize=12, title_fontsize=13, bbox_to_anchor=(0.05, 0.25), frameon=True, edgecolor="0.8", framealpha=0.85)
    else:
        ax.grid(True, alpha=0.5, linestyle="-", linewidth=0.8)
        ax.set_facecolor("white")
        ax.scatter(time_saved_list, token_wasted_list, c=color_list, s=100, alpha=0.7)
        for i, (x, y, target) in enumerate(zip(time_saved_list, token_wasted_list, target_steps_list)):
            ax.annotate(str(target), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set_xlabel("Percentage of Time Saved (%)", fontsize=14)
        ax.set_ylabel("Percentage of Extra Tokens Used (%)", fontsize=14)
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[pred], markersize=12,
                       label=f"{pred} prediction{'s' if pred > 1 else ''}")
            for pred in sorted(k for k in colors if k != 4)
        ]
        ax.legend(handles=legend_elements, title="Number of Predictions", loc="lower right",
                  fontsize=12, title_fontsize=13, bbox_to_anchor=(0.95, 0.1))

    out_path = f"{base_path}/plot_with_confidence.pdf" if with_confidence else f"{base_path}/plot.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def main_time_token(args) -> None:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.base_dir is not None:
        config.setdefault("paths", {})["sample_trajectories"] = args.base_dir
    base_dir = config.get("paths", {}).get("sample_trajectories")
    if not base_dir:
        base_dir = input("Enter the base directory: ").strip()
    if not base_dir or not os.path.exists(base_dir):
        print("Error: Base directory not found.")
        raise SystemExit(1)

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if args.confidence:
            if not os.path.exists(os.path.join(folder_path, "analysis_results_with_confidence.csv")):
                print(f"Skipping {folder_path}: analysis_results_with_confidence.csv not found")
                continue
        else:
            if not os.path.exists(os.path.join(folder_path, "analysis_results.csv")):
                print(f"Skipping {folder_path}: analysis_results.csv not found")
                continue
        print(f"Processing folder: {folder_path}")
        create_time_token_plot(folder_path, with_confidence=args.confidence)


# -------- combined bar plot (from plot_combined_bars.py) --------

def find_all_analysis_directories(root_path):
    """
    Automatically find all directories containing analysis_results.csv files.
    """
    pattern = os.path.join(root_path, "**", "analysis_results.csv")
    csv_files = glob.glob(pattern, recursive=True)
    directories = [os.path.dirname(csv_file) for csv_file in csv_files]
    return directories


def create_combined_bar_plots(base_directories, target_steps=[30, 50], output_path=None, speculative_window_accuracy=True):
    """
    Creates bar charts showing guess accuracy and time saved percentage
    for different numbers of predictions, combining data from multiple CSV files
    and showing confidence intervals for specific target steps.
    """
    all_data = {}

    for base_dir in base_directories:
        csv_path = os.path.join(base_dir, "analysis_results.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping...")
            continue

        print(f"Reading data from: {csv_path}")

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                target_steps_val = int(row[1])
                num_predictions = int(row[3])
                time_saved = float(row[6])
                if speculative_window_accuracy:
                    accuracy = float(row[12]) * 100
                else:
                    accuracy = float(row[11]) * 100

                key = (target_steps_val, num_predictions)

                if key not in all_data:
                    all_data[key] = {'time_saved': [], 'accuracy': []}

                all_data[key]['time_saved'].append(time_saved)
                all_data[key]['accuracy'].append(accuracy)

    n_target_steps = len(target_steps)
    fig, axes = plt.subplots(1, n_target_steps, figsize=(6 * n_target_steps, 5))

    if n_target_steps == 1:
        axes = [axes]

    for plot_idx, target_step in enumerate(target_steps):
        ax = axes[plot_idx]

        step_data = {}
        for (ts, num_pred), values in all_data.items():
            if ts == target_step:
                step_data[num_pred] = values

        if not step_data:
            ax.text(0.5, 0.5, f'No data for {target_step} steps',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Target Steps: {target_step}')
            continue

        predictions = sorted(step_data.keys())
        avg_accuracy = []
        avg_time_saved = []
        accuracy_errors = []
        time_saved_errors = []
        sample_sizes = []

        for pred in predictions:
            acc_values = step_data[pred]['accuracy']
            time_values = step_data[pred]['time_saved']

            acc_mean = np.mean(acc_values)
            time_mean = np.mean(time_values)

            n_acc = len(acc_values)
            n_time = len(time_values)

            if n_acc > 1:
                acc_std = np.std(acc_values, ddof=1)
                acc_error = stats.t.ppf(0.975, n_acc-1) * acc_std / np.sqrt(n_acc)
            else:
                acc_error = 0

            if n_time > 1:
                time_std = np.std(time_values, ddof=1)
                time_error = stats.t.ppf(0.975, n_time-1) * time_std / np.sqrt(n_time)
            else:
                time_error = 0

            avg_accuracy.append(acc_mean)
            avg_time_saved.append(time_mean)
            accuracy_errors.append(acc_error)
            time_saved_errors.append(time_error)
            sample_sizes.append(n_acc)

        x = np.arange(len(predictions))
        width = 0.35

        bars1 = ax.bar(x - width/2, avg_accuracy, width,
                      label='Speculative Accuracy (%)', color='skyblue', alpha=0.8,
                      yerr=accuracy_errors, capsize=8)
        bars2 = ax.bar(x + width/2, avg_time_saved, width,
                      label='Time Saved (%)', color='lightcoral', alpha=0.8,
                      yerr=time_saved_errors, capsize=8)

        for i, (bar, error) in enumerate(zip(bars1, accuracy_errors)):
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height + error),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12)

        for i, (bar, error) in enumerate(zip(bars2, time_saved_errors)):
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height + error),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=14)

        ax.set_ylabel('Percentage', fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{pred} prediction{"s" if pred > 1 else ""}' for pred in predictions], fontsize=16)
        ax.legend(fontsize=15)
        ax.grid(True, alpha=0.3, axis='y')

        max_val = max(max([a + e for a, e in zip(avg_accuracy, accuracy_errors)]),
                     max([t + e for t, e in zip(avg_time_saved, time_saved_errors)]))
        ax.set_ylim(0, max_val * 1.15)

        print(f"\nTarget Steps: {target_step}")
        print("-" * 40)
        for i, pred in enumerate(predictions):
            print(f"{pred} prediction{'s' if pred > 1 else ''}: "
                  f"Accuracy={avg_accuracy[i]:.1f}±{accuracy_errors[i]:.1f}%, "
                  f"Time Saved={avg_time_saved[i]:.1f}±{time_saved_errors[i]:.1f}% "
                  f"(n={sample_sizes[i]})")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"\nPlot saved to: {output_path}")

    plt.show()


def main_bars(args) -> None:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.target_steps is not None:
        config.setdefault("analysis", {})["target_steps"] = args.target_steps
    target_steps = config.get("analysis", {}).get("target_steps")
    if target_steps is None:
        print("Error: --target-steps required or set config analysis.target_steps")
        sys.exit(1)
    root_path = args.root_path
    if root_path is None:
        root_path = input("Enter the root path: ").strip()
    if not root_path or not os.path.exists(root_path):
        print("Error: Root path not found.")
        sys.exit(1)
    directories = find_all_analysis_directories(root_path)
    print(f"Found {len(directories)} directories with analysis data")
    output_path = args.output if args.output is not None else os.path.join(root_path, "combined_bar_plots.pdf")
    # Accuracy: --speculative-accuracy → row[12], otherwise row[11]
    speculative_window_accuracy = args.speculative_accuracy
    if directories:
        create_combined_bar_plots(directories, target_steps=target_steps, output_path=output_path, speculative_window_accuracy=speculative_window_accuracy)
    else:
        print("No analysis_results.csv files found!")


# -------- CLI --------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified plotting: time-token scatter or combined bar charts.",
    )
    parser.add_argument("--config", default="config.yml", help="Path to config YAML")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Plot type")

    # time-token
    p_tt = subparsers.add_parser("time-token", help="Plot time saved vs token usage (optionally with confidence)")
    p_tt.add_argument("--base-dir", "-b", default=None, help="Base directory (default: from config paths.sample_trajectories)")
    p_tt.add_argument("--confidence", action="store_true", help="Include confidence results and time-spent / step markers")
    p_tt.set_defaults(func=main_time_token)

    # bars
    p_bars = subparsers.add_parser("bars", help="Combined bar charts (accuracy + time saved) from multiple dirs")
    p_bars.add_argument("--root-path", "-r", default=None, help="Root directory to search for analysis_results.csv (recursive)")
    p_bars.add_argument("--output", "-o", default=None, help="Output PDF path (default: <root_path>/combined_bar_plots.pdf)")
    p_bars.add_argument("--target-steps", type=int, nargs="+", default=None, help="Target steps to plot (default: from config analysis.target_steps)")
    p_bars.add_argument("--speculative-accuracy", action="store_true", help="Use speculative/window accuracy (row 12); otherwise use step accuracy (row 11)")
    p_bars.add_argument("--step-accuracy", action="store_true", help="Use step accuracy (row 11); default when --speculative-accuracy is not set")
    p_bars.set_defaults(func=main_bars)

    args = parser.parse_args()
    # Pass config path to subcommand (parent already parsed --config)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
