import os
import json
import shutil
import re
import pandas as pd

from . import constants


class Utils:

    @staticmethod
    def read_json(path):
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(obj, path, delete_prev_file=False):
        if os.path.exists(path) and delete_prev_file:
            os.remove(path)
        with open(path, "w") as f:
            json.dump(obj, f, indent=4)

    @staticmethod
    def read_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def save_file(string, path, delete_prev_file=False):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(path) and delete_prev_file:
            os.remove(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(string)

    @staticmethod
    def join_prompt(*args):
        output = ""
        for arg in args:
            output += arg
        return output

    @staticmethod
    def append_file(string, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(string + "\n")

    @staticmethod
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def delete_dir(path, nested=False):
        if os.path.isdir(path):
            try:
                os.rmdir(path)
            except OSError as e:
                if nested:
                    shutil.rmtree(path)
                else:
                    raise e

    @staticmethod
    def check_dir(dir_path):
        for file_name in constants.trajectory_filenames:
            path = os.path.join(dir_path, file_name)
            if not os.path.exists(path):
                print(f"{dir_path} does not have {file_name} file")

    @staticmethod
    def check_all_dirs(base_path):
        for direc in os.listdir(base_path):
            Utils.check_dir(os.path.join(base_path, direc))

    @staticmethod
    def extract_ith_step_info(i, text, trajectory_type):
        pattern = rf"{trajectory_type.upper()} TRAJECTORY:\n Thought {i}: (.*?)\nAction {i}: (.*?)\nObservation {i}: (.*?)\n"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            action = match.group(2).strip()
            observation = match.group(3).strip()
            return {"thought": thought, "action": action, "observation": observation}
        else:
            return {"thought": None, "action": None, "observation": None}

    @staticmethod
    def process_obs(base_path):
        not_processed = []
        for direc in os.listdir(base_path):
            log_path = os.path.join(base_path, direc, "log.txt")
            normal_obs_path = os.path.join(base_path, direc, "normalobs.json")
            sim_obs_path = os.path.join(base_path, direc, "simobs.json")
            logs = Utils.read_file(log_path)
            normal_obs = Utils.read_json(normal_obs_path)
            sim_obs = Utils.read_json(sim_obs_path)
            new_normal_obs = {}
            new_normal_obs["prompt"] = normal_obs["prompt"]
            new_sim_obs = {}
            new_sim_obs["prompt"] = sim_obs["prompt"]
            try:
                normal_first_steps = Utils.extract_ith_step_info(1, logs, "normal")
                sim_first_steps = Utils.extract_ith_step_info(1, logs, "simulation")
                new_normal_obs["actions"] = normal_obs["actions"][normal_obs["actions"].index(normal_first_steps["action"]):]
                new_normal_obs["thoughts"] = normal_obs["thoughts"][normal_obs["thoughts"].index(normal_first_steps["thought"]):]
                new_normal_obs["observations"] = normal_obs["observations"][normal_obs["observations"].index(normal_first_steps["observation"]):]
                new_sim_obs["actions"] = sim_obs["actions"][sim_obs["actions"].index(sim_first_steps["action"]):]
                new_sim_obs["thoughts"] = sim_obs["thoughts"][sim_obs["thoughts"].index(sim_first_steps["thought"]):]
                new_sim_obs["observations"] = sim_obs["observations"][sim_obs["observations"].index(sim_first_steps["observation"]):]
                assert len(new_normal_obs["thoughts"]) == len(new_normal_obs["actions"]) == len(new_normal_obs["observations"]), "lengths of thoughts, actions and observations are unequal for normal obs"
                assert len(new_sim_obs["thoughts"]) == len(new_sim_obs["actions"]) == len(new_sim_obs["observations"]), "lengths of thoughts, actions and observations are unequal for sim obs"
            except Exception as e:
                print(e)
                not_processed.append(direc)
                continue

            Utils.save_json(new_normal_obs, normal_obs_path, delete_prev_file=True)
            Utils.save_json(new_sim_obs, sim_obs_path, delete_prev_file=True)
        print(len(not_processed))
        Utils.save_file(str(not_processed), "./not_processed.txt")

    @staticmethod
    def cleanup_trajs(trajs_folder):
        for direc in os.listdir(trajs_folder):
            datapoint_path = os.path.join(trajs_folder, direc)
            if Utils.is_dirty_traj(datapoint_path):
                print(f"Cleaning up {datapoint_path}")
                Utils.delete_dir(datapoint_path, nested=True)

    @staticmethod
    def is_dirty_traj(datapoint_path):
        files = ["metrics.json", "normalobs.json", "simobs.json", "log.txt"]
        try:
            for file_name in files:
                assert os.path.exists(os.path.join(datapoint_path, file_name)), f"{file_name} is missing"
        except AssertionError:
            return True
        return False

    @staticmethod
    def convert_json_to_csv(json_data):
        df_dict = {}
        df_dict["agent model"] = []
        df_dict["guess model"] = []
        for agent_model in json_data.keys():
            for guess_model in json_data[agent_model].keys():
                df_dict["agent model"].append(agent_model)
                df_dict["guess model"].append(guess_model)
                for key in json_data[agent_model][guess_model].keys():
                    if key in df_dict.keys():
                        df_dict[key].append(json_data[agent_model][guess_model][key])
                    else:
                        df_dict[key] = [json_data[agent_model][guess_model][key]]
        df = pd.DataFrame.from_dict(df_dict)
        return df

    @staticmethod
    def convert_json_to_csv2(json_path, csv_path):
        json_data = Utils.read_json(json_path)
        csv_dict = {}
        for data in json_data:
            for key in data.keys():
                if key not in csv_dict.keys():
                    csv_dict[key] = [data[key]]
                else:
                    csv_dict[key].append(data[key])
        df = pd.DataFrame.from_dict(csv_dict)
        df.to_csv(csv_path, index=False)

    @staticmethod
    def avg(my_list):
        if len(my_list) == 0:
            return None
        return sum(my_list) / len(my_list)
