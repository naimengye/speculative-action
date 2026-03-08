import os

from .utils import Utils


class Metrics:

    @staticmethod
    def get_action_metrics(normal_dict, sim_dict, k=None, sparse=False):
        """Compute per-action-type accuracy metrics.

        Args:
            normal_dict: Normal trajectory dict with "actions" key.
            sim_dict: Simulated trajectory dict with "actions" key.
            k: If None, check all sim actions (any-match). If 1, check only first sim action.
            sparse: If True, compare only action type (Search/Lookup/Finish), not arguments.
        """
        normal_actions = normal_dict["actions"]
        sim_actions = sim_dict["actions"]
        metric_dict = {"general": 0}
        count_dict = {"general": 0}

        for idx, (na, sa) in enumerate(zip(normal_actions, sim_actions)):
            if idx == 0:
                continue

            score = Metrics._score_action_pair(na, sa, k, sparse)
            metric_dict["general"] += score
            count_dict["general"] += 1

            na_name = Metrics.get_action_name(na)
            if na_name is None:
                continue
            if na_name not in metric_dict:
                metric_dict[na_name] = 0
                count_dict[na_name] = 0
            metric_dict[na_name] += score
            count_dict[na_name] += 1

        if count_dict["general"] == 0:
            return metric_dict

        for key in metric_dict:
            metric_dict[key] /= count_dict[key]
        return metric_dict

    @staticmethod
    def _score_action_pair(na, sa, k, sparse):
        """Score a single (normal_action, sim_action) pair."""
        if isinstance(sa, list):
            if not sa:
                return 0
            if k == 1:
                return Metrics.compare_action(na, sa[0], sparse)
            else:
                return Metrics.compare_actions(na, sa, sparse)
        else:
            return Metrics.compare_action(na, sa, sparse)

    @staticmethod
    def get_actions_metric(dict1, dict2, sparse=False):
        actions1 = dict1["actions"]
        actions2 = dict2["actions"]
        assert len(actions1) == len(actions2), "Different number of actions taken in wiki and guess"
        n = len(actions1)
        score = 0
        for a1, a2 in zip(actions1, actions2):
            score += Metrics.compare_action(a1, a2, sparse)
        return score / n

    @staticmethod
    def compare_actions(na, simulated_actions_list, sparse=False):
        total_score = 0
        for sa in simulated_actions_list:
            score = Metrics.compare_action(na, sa, sparse)
            total_score += score
        return 1 if total_score > 0 else 0

    @staticmethod
    def compare_action(action1, action2, sparse=False):
        action1 = action1.lower()
        action2 = action2.lower()
        if sparse:
            return 1 if Metrics.get_action_name(action1) == Metrics.get_action_name(action2) else 0
        else:
            return 1 if action1 == action2 else 0

    @staticmethod
    def get_action_name(full_action):
        full_action = full_action.strip()
        ind = full_action.find('[')
        if ind < 0:
            return None
        else:
            return full_action[:ind]

    @staticmethod
    def get_avg_time_taken(obs_dict):
        avg_time = 0
        n = 0
        for i, t in enumerate(obs_dict["time_taken"]):
            if isinstance(obs_dict["actions"][i], str):
                if "search[" in obs_dict["actions"][i].lower():
                    n += 1
                    avg_time += t
            elif isinstance(obs_dict["actions"][i], list):
                for action in obs_dict["actions"][i]:
                    if "search[" in action.lower():
                        n += 1
                        avg_time += t
                        break
        if n == 0:
            return 0
        return avg_time / n

    @staticmethod
    def get_action_specific_avg_metric(dir_path, get_time=False):
        avg_metrics_dict = {
            "general": 0, "Search": 0, "Lookup": 0, "Finish": 0,
            "normal_avg_searchtime": 0, "sim_avg_searchtime": 0,
        }
        metric_names = ["general", "Search", "Lookup", "Finish"]
        n_metrics = 0
        n_times = 0
        for direc in os.listdir(dir_path):
            try:
                metrics_dict = Utils.read_json(os.path.join(dir_path, direc, "metrics.json"))
                if get_time:
                    normal_obs_dict = Utils.read_json(os.path.join(dir_path, direc, "normalobs.json"))
                    sim_obs_dict = Utils.read_json(os.path.join(dir_path, direc, "simobs.json"))
                    normal_avg_searchtime = Metrics.get_avg_time_taken(normal_obs_dict)
                    try:
                        sim_avg_searchtime = Metrics.get_avg_time_taken(sim_obs_dict)
                    except AttributeError:
                        sim_avg_searchtime = 0
                    avg_metrics_dict["normal_avg_searchtime"] += normal_avg_searchtime
                    avg_metrics_dict["sim_avg_searchtime"] += sim_avg_searchtime
                    n_times += 1
            except FileNotFoundError as e:
                print("Filenotfound", os.path.join(dir_path, direc))
                print(e)
                continue
            for metric in metric_names:
                avg_metrics_dict[metric] += metrics_dict.get(metric, 0)
            n_metrics += 1

        for key in metric_names:
            try:
                avg_metrics_dict[key] /= n_metrics
            except ZeroDivisionError:
                avg_metrics_dict[key] = None

        if n_times > 0:
            avg_metrics_dict["normal_avg_searchtime"] /= n_times
            avg_metrics_dict["sim_avg_searchtime"] /= n_times

        return avg_metrics_dict, n_metrics

    @staticmethod
    def recalculate_metrics(base_traj_path):
        for folder_name in os.listdir(base_traj_path):
            save_dir = os.path.join(base_traj_path, str(folder_name))
            normal_observations_dict = Utils.read_json(os.path.join(save_dir, "normalobs.json"))
            sim_observations_dict = Utils.read_json(os.path.join(save_dir, "simobs.json"))
            metrics_file_path = os.path.join(save_dir, "metrics.json")
            if os.path.exists(metrics_file_path):
                os.remove(metrics_file_path)
            metric_dict = Metrics.get_action_metrics(normal_observations_dict, sim_observations_dict, sparse=False)
            Utils.save_json(metric_dict, metrics_file_path)

    @staticmethod
    def cum_metrics(base_traj_path):
        avg_metrics_dict = {}
        n_metrics = 0
        metric_names = ["general", "Search", "Lookup", "Finish"]
        for metric in metric_names:
            avg_metrics_dict[metric + "_top1"] = []
            avg_metrics_dict[metric + "_top3"] = []
        avg_metrics_dict["normal_avg_searchtime"] = []
        avg_metrics_dict["sim_avg_searchtime"] = []

        for folder_name in os.listdir(base_traj_path):
            save_dir = os.path.join(base_traj_path, str(folder_name))
            normal_observations_dict = Utils.read_json(os.path.join(save_dir, "normalobs.json"))
            sim_observations_dict = Utils.read_json(os.path.join(save_dir, "simobs.json"))
            metric_dict_top1 = Metrics.get_action_metrics(
                normal_observations_dict, sim_observations_dict, k=1, sparse=False
            )
            metric_dict_top3 = Metrics.get_action_metrics(
                normal_observations_dict, sim_observations_dict, k=3, sparse=False
            )
            for metric in metric_names:
                avg_metrics_dict[metric + "_top1"].append(metric_dict_top1.get(metric, 0))
                avg_metrics_dict[metric + "_top3"].append(metric_dict_top3.get(metric, 0))
            avg_metrics_dict["normal_avg_searchtime"].append(Metrics.get_avg_time_taken(normal_observations_dict))
            avg_metrics_dict["sim_avg_searchtime"].append(Metrics.get_avg_time_taken(sim_observations_dict))
            n_metrics += 1

        return avg_metrics_dict, n_metrics
