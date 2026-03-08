import os
import time
import json
import random
import re
import requests
from os.path import join

from . import constants
from .utils import Utils
from .metrics import Metrics
from .prompts import PromptTemplates
from .llm_client import LLMClient
from . import environment
from . import wrappers


class HotPotQARun:
    def __init__(self, model_name="gemini", guess_model_name="gemini", to_print_output=True):
        self.llm = LLMClient(
            model_name=model_name,
            temperature=constants.temperature,
            max_tokens=constants.max_output_tokens,
            top_p=constants.top_p,
        )
        self.model_name = model_name
        self.guess_model_name = guess_model_name
        self.env = self._get_env()
        self.simulation_observations_dict = {}
        self.current_index = None
        self.base_traj_path = self.recalc_base_traj_path()
        self.to_print_output = to_print_output

    def recalc_base_traj_path(self):
        btp = (
            "./run_metrics/agent_"
            + self.model_name.split('/')[-1]
            + "_top" + str(constants.guess_num_actions)
            + "/trajs_" + self.guess_model_name.split("/")[-1]
        )
        self.base_traj_path = btp
        return btp

    def _get_env(self):
        env = environment.WikiEnv(guess_model_name=self.guess_model_name)
        env = wrappers.HotPotQAWrapper(env, split="dev")
        env = wrappers.LoggingWrapper(env)
        env = wrappers.HistoryWrapper(env, obs_format="history")
        return env

    def log(self, *args, save_log=True):
        text = " ".join(str(a) for a in args).strip() + "\n"
        if self.to_print_output:
            print(text, end="")
        if save_log:
            log_path = join(self.base_traj_path, str(self.current_index), "log.txt")
            Utils.append_file(text, log_path)

    def step(self, env, action, simulate=False):
        if simulate:
            start = time.perf_counter()
            obs, r, done, info = env.step(action, step_type="simulate")
            end = time.perf_counter()
            return env.sim_obs, r, done, info, end - start

        attempts = 0
        while attempts < 10:
            try:
                start = time.perf_counter()
                obs, r, done, info = env.step(action)
                end = time.perf_counter()
                return obs, r, done, info, end - start
            except requests.exceptions.Timeout:
                attempts += 1

    def extract_action(self, action_string):
        search_pattern = r'[Ss]earch\[[^\]]+\]'
        lookup_pattern = r'[Ll]ookup\[[^\]]+\]'
        finish_pattern = r'[Ff]inish\[[^\]]+\]'
        for pattern in [search_pattern, lookup_pattern, finish_pattern]:
            output = re.search(pattern, action_string)
            if output:
                return output.group()
        return None

    def extract_actions(self, action_string):
        search_pattern = r'[Ss]earch\[[^\]]+\]'
        lookup_pattern = r'[Ll]ookup\[[^\]]+\]'
        finish_pattern = r'[Ff]inish\[[^\]]+\]'
        outputs = []
        for pattern in [search_pattern, lookup_pattern, finish_pattern]:
            outputs += re.findall(pattern, action_string)
        return outputs

    def action_lowercase(self, action):
        index = action.find("[")
        if index == -1:
            return action[0].lower() + action[1:]
        else:
            return action[:index].lower() + action[index:]

    def separate_thought_and_action(self, i, thought_action):
        action_condition = f"Action {i}: " in thought_action
        thought_condition = f"Thought {i}: " in thought_action
        thought, action = None, None
        if thought_condition and action_condition:
            thought, action = thought_action.strip().split(f"Action {i}: ")
            thought = thought.strip().split(f"Thought {i}: ")[1]
        elif thought_condition:
            thought = thought_action.strip().split(f"Thought {i}: ")[1]
            action = None
        elif action_condition:
            action = thought_action.strip().split(f"Action {i}: ")[1]
            thought = "Let me do the action " + action
        return thought, action

    def separate_thought_and_actions(self, i, thought_action):
        try:
            split = thought_action.split(f"Thought {i}: ")
            thought = split[0].strip() if len(split) == 1 else split[1].strip()
        except IndexError:
            thought = thought_action.strip()
        actions = self.extract_actions(thought_action)
        return thought, actions

    def generate_thought_actions(self, i, running_prompt, n_calls_badcalls, num_actions=1, max_retries=1):
        n_calls_badcalls[0] += 1

        thought_action = self.llm.call(
            running_prompt + PromptTemplates.ACTION_GUESS_PROMPT.format(i=i, num_guesses=num_actions),
            stop=None,
        )
        thought, actions = self.separate_thought_and_actions(i, thought_action)

        retry_attempt = 1
        while not actions and retry_attempt <= max_retries:
            self.log(f"  [Retry {retry_attempt}/{max_retries}] No actions found, retrying...")
            n_calls_badcalls[0] += 1
            n_calls_badcalls[1] += 1
            temp_actions = self.llm.call(
                running_prompt + PromptTemplates.RETRY_PROMPT.format(
                    attempt=retry_attempt, role=constants.agent_role, num_guesses=num_actions
                )
            )
            actions = self.extract_actions(temp_actions)
            retry_attempt += 1

        if not actions:
            raise ValueError("Action not found in LLM output after all retries")

        return thought, actions

    def webthink(self, idx=None, prompt=None, to_print=True, n=8, simulate=False):
        done = False
        running_prompt = prompt
        question = self.env.reset(idx=idx)
        running_prompt += question + "\n"
        self.env.normal_trajectory_dict["prompt"] = running_prompt

        if to_print:
            self.log(f"\n{'='*60}")
            self.log(f"  Question [{idx}]: {question}")
            self.log(f"{'='*60}")

        if simulate:
            sim_running_prompt = running_prompt
            self.env.sim_trajectory_dict["prompt"] = sim_running_prompt

        n_calls_badcalls = [0, 0]

        for i in range(1, n):
            if to_print:
                self.log(f"\n--- Step {i}/{n-1} ---")
            try:
                thought, actions = self.generate_thought_actions(
                    i, running_prompt, n_calls_badcalls, max_retries=constants.max_agent_retries
                )
                action = actions[0]
                if simulate:
                    sim_thought, sim_actions = self.generate_thought_actions(
                        i, sim_running_prompt, n_calls_badcalls,
                        num_actions=constants.guess_num_actions,
                        max_retries=constants.max_guess_retries,
                    )
                    sim_running_prompt = running_prompt
            except ValueError as e:
                self.log(f"[ERROR] {e}", save_log=False)
                continue

            # Normal trajectory
            obs, r, done, info, normal_traj_time_taken = self.step(self.env, self.action_lowercase(action))
            obs = obs.replace('\\n', '')
            self.env.update_traj_dict_records(thought, action, obs, normal_traj_time_taken, False)
            next_step_string = PromptTemplates.NEXT_STEP_PROMPT.format(i=i, thought=thought, action=action, obs=obs)
            running_prompt += next_step_string
            if to_print:
                self.log(f"  [Normal]  Thought: {thought}")
                self.log(f"            Action:  {action}")
                self.log(f"            Obs:     {obs[:200]}{'...' if len(obs) > 200 else ''}")
                self.log(f"            Time:    {normal_traj_time_taken:.3f}s")

            if simulate:
                sim_obs, sim_r, sim_done, sim_info, sim_traj_time_taken = self.step(
                    self.env, self.action_lowercase(action), simulate=True
                )
                sim_obs = sim_obs.replace('\n', '')
                self.env.update_traj_dict_records(sim_thought, sim_actions, sim_obs, sim_traj_time_taken, True)
                next_sim_step_string = PromptTemplates.NEXT_STEP_PROMPT.format(
                    i=i, thought=thought, action=action, obs=sim_obs
                )
                if to_print:
                    self.log(f"  [Sim]     Thought: {sim_thought}")
                    self.log(f"            Actions: {sim_actions}")
                    self.log(f"            Obs:     {sim_obs[:200]}{'...' if len(sim_obs) > 200 else ''}")
                    self.log(f"            Time:    {sim_traj_time_taken:.3f}s")
                sim_running_prompt += next_sim_step_string

            if done:
                break

        if not done:
            obs, r, done, info, time_taken = self.step(self.env, "finish[]")
        if to_print:
            self.log(f"\n  Result: em={info.get('em', 'N/A')}  f1={info.get('f1', 'N/A')}")
        info.update({'n_calls': n_calls_badcalls[0], 'n_badcalls': n_calls_badcalls[1], 'traj': running_prompt})
        return info

    def run(self, webthink_simulate=False, skip_done=False):
        from google.genai.errors import ClientError, ServerError

        idxs = list(range(constants.num))
        random.Random(constants.random_seed).shuffle(idxs)

        webthink_examples = Utils.read_json(
            join(constants.prompts_folder, constants.prompt_file)
        ).get('webthink_simple6')
        webthink_prompt = Utils.join_prompt(
            PromptTemplates.REACT_INSTRUCTION, webthink_examples, PromptTemplates.PROMPT_INSTRUCTION
        )

        rewards = []
        infos = []
        old_time = time.time()

        for i in idxs[:constants.n_samples_to_run]:
            self.current_index = i
            current_dir_path = join(self.base_traj_path, str(self.current_index))

            log_path = join(current_dir_path, "log.txt")
            if skip_done:
                if os.path.exists(log_path):
                    self.log(f"[SKIP] Index {i} already done", save_log=False)
                    continue

            Utils.delete_file(log_path)

            try:
                info = self.webthink(
                    i, prompt=webthink_prompt, to_print=True,
                    n=constants.n_steps_to_run, simulate=webthink_simulate,
                )

            except ClientError as e:
                self.log(f"[ERROR] Client error, sleeping {constants.client_error_sleep_time}s: {e}", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(constants.client_error_sleep_time)
                continue

            except ServerError as e:
                self.log(f"[ERROR] Server error, sleeping {constants.server_error_sleep_time}s: {e}", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(constants.server_error_sleep_time)
                continue

            except ZeroDivisionError:
                self.log("[ERROR] ZeroDivisionError, skipping sample", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                continue

            except Exception as e:
                Utils.delete_dir(current_dir_path, nested=True)
                raise e

            rewards.append(info['em'])
            infos.append(info)

            avg_reward = sum(rewards) / len(rewards)
            avg_time = (time.time() - old_time) / len(rewards)
            self.log(
                f"\n  [Progress] {len(rewards)} samples done | "
                f"Total reward: {sum(rewards)} | "
                f"Avg reward: {avg_reward:.4f} | "
                f"Avg time/sample: {avg_time:.1f}s"
            )

            normal_observations_dict = self.env.normal_trajectory_dict
            sim_observations_dict = self.env.sim_trajectory_dict
            Utils.save_json(normal_observations_dict, join(current_dir_path, "normalobs.json"))
            Utils.save_json(sim_observations_dict, join(current_dir_path, "simobs.json"))
            try:
                metric_dict = Metrics.get_action_metrics(
                    normal_observations_dict, sim_observations_dict, sparse=False
                )
            except ZeroDivisionError:
                self.log("[WARN] ZeroDivisionError in metrics, skipping this sample", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                continue
            self.log(f"  [Metrics]  idx={self.current_index}  {metric_dict}", save_log=False)
            Utils.save_json(metric_dict, join(current_dir_path, "metrics.json"))

        self.env.write()
