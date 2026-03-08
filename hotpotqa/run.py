import os
import json
import argparse
from os.path import join

from src import constants
from src.runner import HotPotQARun
from src.utils import Utils
from src.metrics import Metrics
from src.grapher import Grapher


AGENT_MODEL_NAMES = [
    "google/gemini-2.5-flash",
    "openai/gpt-4",
    "openai/gpt-5",
]

GUESS_MODEL_NAMES = [
    "openai/gpt-5-nano",
    "openai/gpt-4.1-nano",
    "openai/gpt-3.5-turbo",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
]


def compute_metrics(runner, save=False):
    all_avg_metrics = {}
    for agent in AGENT_MODEL_NAMES:
        all_avg_metrics[agent] = {}
        for guess_model in GUESS_MODEL_NAMES:
            runner.model_name = agent
            runner.guess_model_name = guess_model
            runner.recalc_base_traj_path()
            if not os.path.exists(runner.base_traj_path):
                print(f"Path {runner.base_traj_path} does not exist. Skipping...")
                continue
            avg_metrics_dict, n_samples = Metrics.get_action_specific_avg_metric(
                runner.base_traj_path, get_time=True
            )
            print(
                f"AVERAGE METRICS for agent {agent} using guess model {guess_model}:\n",
                json.dumps(avg_metrics_dict, indent=4),
                f"\nfor {n_samples} observations",
            )
            all_avg_metrics[agent][guess_model] = avg_metrics_dict
    metrics_df = Utils.convert_json_to_csv(all_avg_metrics)
    if save:
        Utils.save_json(
            all_avg_metrics,
            join("./run_metrics", f"all_avg_metrics_top{constants.guess_num_actions}.json"),
        )
        Utils.save_file(
            metrics_df.to_csv(index=False),
            join("./run_metrics", f"all_avg_metrics_top{constants.guess_num_actions}.csv"),
        )


def compute_cumulative_metrics(runner, save=False):
    all_avg_metrics = {}
    for agent in AGENT_MODEL_NAMES:
        all_avg_metrics[agent] = {}
        for guess_model in GUESS_MODEL_NAMES:
            runner.model_name = agent
            runner.guess_model_name = guess_model
            runner.recalc_base_traj_path()
            if not os.path.exists(runner.base_traj_path):
                print(f"Path {runner.base_traj_path} does not exist. Skipping...")
                continue
            avg_metrics_dict, n_samples = Metrics.cum_metrics(runner.base_traj_path)
            print(
                f"AVERAGE METRICS for agent {agent} using guess model {guess_model}:\n",
                json.dumps(avg_metrics_dict, indent=4),
                f"\nfor {n_samples} observations",
            )
            all_avg_metrics[agent][guess_model] = avg_metrics_dict
    if save:
        Utils.save_json(all_avg_metrics, join("./run_metrics", "list_metrics_top1_3.json"))


def main():
    parser = argparse.ArgumentParser(description="HotPotQA Speculative Actions Runner")
    parser.add_argument("--getmetric", action="store_true", help="Compute and print average metrics")
    parser.add_argument("--getmetric2", action="store_true", help="Compute and print cumulative metrics")
    parser.add_argument("--savemetrics", action="store_true", help="Save computed metrics to disk")
    parser.add_argument("--graph", action="store_true", help="Plot agent time comparison graphs")
    parser.add_argument("--graph2", action="store_true", help="Plot top-1 vs top-3 comparison graphs")
    parser.add_argument("--graph3", action="store_true", help="Plot detailed metric comparison graphs")
    parser.add_argument("--noprint", action="store_false", help="Suppress command line output")
    parser.add_argument("--norun", action="store_false", help="Skip running the experiment")
    parser.add_argument("--modelname", default=constants.openrouter_model_name, help="Agent model name")
    parser.add_argument("--guessmodelname", default=constants.openrouter_guess_model_name, help="Guess model name")
    parser.add_argument("--cleanuptrajs", action="store_true", help="Clean up incomplete trajectories")
    args = parser.parse_args()

    runner = HotPotQARun(
        model_name=args.modelname,
        guess_model_name=args.guessmodelname,
        to_print_output=args.noprint,
    )

    if args.cleanuptrajs:
        Utils.cleanup_trajs(runner.base_traj_path)

    if args.norun:
        runner.run(webthink_simulate=True, skip_done=True)
        Utils.cleanup_trajs(runner.base_traj_path)

    if args.getmetric:
        compute_metrics(runner, save=args.savemetrics)

    if args.getmetric2:
        compute_cumulative_metrics(runner, save=args.savemetrics)

    if args.graph:
        data = Utils.read_json(
            join("./run_metrics", f"all_avg_metrics_top{constants.guess_num_actions}.json")
        )
        for agent in data.keys():
            Grapher.graph_agent_times(data, agent=agent)

    if args.graph2:
        data = Utils.read_json(join("./run_metrics", "comparision_top1_top3.json"))
        Grapher.graph_metric_comparison(data)

    if args.graph3:
        data = Utils.read_json(join("./run_metrics", "list_metrics_top1_3.json"))
        Grapher.graph_metric3(data)


if __name__ == "__main__":
    main()
