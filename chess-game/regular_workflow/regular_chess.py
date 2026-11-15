"""
Chess Regular Game Runner

A chess game runner that implements regular execution
where the agent makes a move and the opponent makes a move sequentially.
"""

import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from os.path import join
from typing import Dict, List, Optional, Tuple, Any

import chess
from openai import OpenAI

import textarena as ta
from utils import Utils
import yaml


class Config:
    """Configuration management with YAML support"""
    
    def __init__(self, config_path: Optional[str] = "./config.yml"):
        if config_path and config_path.endswith('.yml'):
            self._load_from_yaml(config_path)
    
    def _load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # API Configuration
            self.openai_api_key = config['api']['openai']['key']
            self.openrouter_api_key = config['api']['openrouter']['key']
            # Optional OpenAI-compatible server such as vLLM; defaults to the OpenAI API
            self.openai_base_url = config['api']['openai'].get('base_url') or "https://api.openai.com/v1"

            # Model Configuration
            self.openai_model_name = config['models']['openai']['main']
            self.openai_guess_model_name = config['models']['openai']['guess']
            self.openrouter_model_name = config['models']['openrouter']['main']
            self.openrouter_guess_model_name = config['models']['openrouter']['guess']

            # Game Configuration
            self.num_chess_players = config['game']['num_players']
            self.client_error_sleep_time = config['game']['error_sleep_time']
            self.server_error_sleep_time = config['game']['error_sleep_time']
            self.stop_after = config['game']['stop_after']
            self.agent_name0 = config['game']['agent_name0']
            self.agent_name1 = config['game']['agent_name1']
            self.num_guesses = config['game']['num_guesses']

            # Guess Model Configuration
            self.guess_model_name = config['guess']['model_name']
            self.guess_provider = config['guess']['provider']

            # Path Configuration
            self.trajectories_path = config['paths']['trajectories']

            # Prompts Configuration
            self.standard_game_prompt = config['prompts']['standard_game']
            self.guess_prompt = config['prompts']['guess']
            self.retry_prompt = config['prompts']['retry']

        except Exception as e:
            print(f"Error loading YAML config: {e}")

class ChessActionCleaner:
    """Utility class for cleaning and validating chess actions"""
    
    UCI_PATTERN = re.compile(r'\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]')
    
    @classmethod
    def clean_action(cls, action: Optional[str]) -> Optional[str]:
        """
        Clean and validate a chess action string.
        
        Args:
            action: Raw action string from agent
            
        Returns:
            Cleaned UCI move in format [move] or None if invalid
        """
        if action is None:
            return None
            
        # Find all matches and take the last one
        matches = cls.UCI_PATTERN.findall(action)
        if matches:
            return f'[{matches[-1]}]'
        
        return None
    
    @classmethod
    def clean_actions(cls, action: Optional[str]) -> List[str]:
        """
        Clean and validate multiple chess action strings.
        
        Args:
            action: Raw action string from agent that may contain multiple moves
            
        Returns:
            List of cleaned UCI moves in format [move]
        """
        if action is None:
            return []
            
        matches = cls.UCI_PATTERN.findall(action)
        return [f'[{move}]' for move in matches]


class GameLogger:
    
    def __init__(self, base_path: str, run_id: str):
        self.base_path = base_path
        self.run_id = run_id
        self.log_path = join(base_path, str(run_id), "log.txt")
    
    def log(self, level: str, *args, save_log: bool = True) -> None:
        """Log message with specified level"""
        message = f"{level.upper()} {' '.join(str(arg) for arg in args)}\n"
        print(message, end='')
        
        if save_log:
            Utils.append_file(message, self.log_path)


class AgentManager:
    
    def __init__(self, config: Config):
        self.config = config

    # Created on first use, so a run only needs the key of the provider it calls
    # (openai>=3 raises as soon as a client is built with an empty key).
    @cached_property
    def openai_client(self) -> OpenAI:
        return OpenAI(api_key=self.config.openai_api_key, base_url=self.config.openai_base_url)

    @cached_property
    def openrouter_client(self) -> OpenAI:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.config.openrouter_api_key)
    
    def create_agents(self, agent0_name: str, agent1_name: str) -> Dict[int, Any]:

        agents = {}
        
        for i, agent_name in enumerate([agent0_name, agent1_name]):
            if agent_name == "OpenRouter":
                agents[i] = ta.agents.OpenRouterAgent(
                    model_name=self.config.openrouter_model_name,
                    system_prompt=self.config.standard_game_prompt,
                    api_key=self.config.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    verbose=False
                )
            elif agent_name == "OpenAI":
                agents[i] = ta.agents.OpenAIAgent(
                    model_name=self.config.openai_model_name,
                    system_prompt=self.config.standard_game_prompt,
                    api_key=self.config.openai_api_key,
                    base_url=self.config.openai_base_url,
                    verbose=False
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_name}")
        
        return agents
    
    def call_guess_llm(self, prompt: str, model_name: str, retries: int = 3) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        for attempt in range(retries):
            try:
                if model_name.startswith("gpt") or model_name.startswith("o"):
                    response = self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": self.config.standard_game_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        reasoning_effort="low",
                        n=1
                    )
                    usage = response.usage
                    if usage:
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens
                        total_tokens = usage.total_tokens
                    return response.choices[0].message.content.strip(), input_tokens, output_tokens, total_tokens
                
                elif "/" in model_name:  # OpenRouter model
                    response = self.openrouter_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": self.config.standard_game_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        reasoning_effort="low"
                    )
                    usage = response.usage
                    if usage:
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens
                        total_tokens = usage.total_tokens
                    return response.choices[0].message.content.strip(), input_tokens, output_tokens, total_tokens
                
            except Exception as e:
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    return None, None, None, None
        
        return None, None, None, None


class RegularChessRunner:
    """
    Main class for running speculative chess games.
    
    This runner implements parallel speculation where a guess model predicts
    the opponent's move while the actual agent is thinking, allowing for
    faster gameplay through speculative execution.
    """
    
    def __init__(
        self, config: Config
    ):
        """Initialize the chess runner with configuration"""
        self.config = config
        self.agent_manager = AgentManager(self.config)
        
        self.agent0_name = self.config.agent_name0
        self.agent1_name = self.config.agent_name1
        self.guess_model_name = self.config.guess_model_name
        self.num_guesses = self.config.num_guesses

        self.current_run_id: Optional[str] = None
        self.base_traj_path = f"{self.config.trajectories_path}/{self.agent0_name}_vs_{self.agent1_name}"
        self.logger: Optional[GameLogger] = None
        
        # Initialize environment
        self.env = self._create_environment()
        print("Chess environment initialized successfully")
    
    def _create_environment(self) -> ta.Env:
        env = ta.make(env_id="Chess-v0")
        return env
    
    def _get_valid_moves(self) -> List[str]:
        """Get list of valid moves in UCI format"""
        return [f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves]
    
    def _guess_action(self, observation: str, retries: int = 3) -> Tuple[Optional[str], float, Optional[int], Optional[int], Optional[int]]:
        start_pred_time = time.perf_counter()
        prompt = observation + self.config.guess_prompt.format(num_guesses=1)
        raw_output, input_tokens, output_tokens, total_tokens = self.agent_manager.call_guess_llm(prompt, self.guess_model_name, retries)
        end_pred_time = time.perf_counter()
        prediction_time = end_pred_time - start_pred_time
        
        if self.logger:
            self.logger.log("SIMULATION GUESS OUTPUT", raw_output)
        
        return ChessActionCleaner.clean_action(raw_output), prediction_time, input_tokens, output_tokens, total_tokens

    def _guess_actions(self, observation: str, retries: int = 3) -> Tuple[Optional[List[str]], float, Optional[int], Optional[int], Optional[int]]:
        start_pred_time = time.perf_counter()
        prompt = observation + self.config.guess_prompt.format(num_guesses=self.num_guesses)
        raw_output, input_tokens, output_tokens, total_tokens = self.agent_manager.call_guess_llm(prompt, self.guess_model_name, retries)
        end_pred_time = time.perf_counter()
        prediction_time = end_pred_time - start_pred_time
        
        if self.logger:
            self.logger.log("SIMULATION GUESS OUTPUT", raw_output)
        
        return ChessActionCleaner.clean_actions(raw_output), prediction_time, input_tokens, output_tokens, total_tokens
    
    def _agent_call_with_retry(
        self, 
        agent: Any, 
        observation: str, 
        player_id: int, 
        valid_moves: List[str],
        retries: int = 3
    ) -> Tuple[Optional[str], int, int, int]:
        role = "White" if player_id == 0 else "Black"
        
        for attempt in range(retries):
            raw_action, input_tokens, output_tokens, total_tokens = agent(observation)
            cleaned_action = ChessActionCleaner.clean_action(raw_action)
            
            if cleaned_action and cleaned_action in valid_moves:
                return cleaned_action, input_tokens, output_tokens, total_tokens
            
            if self.logger:
                self.logger.log("RETRY", f"Attempt {attempt + 1} failed for {role} because {raw_action}")
            
            observation += self.config.retry_prompt.format(
                attempt=attempt + 1, 
                role=role
            )
        
        return None, 0, 0, 0
    
    def _current_agent_task(self, agent: Any, observation: str, player_id: int) -> Tuple[Optional[str], float, int, int, int]:
        """Execute the current agent's move selection"""
        start_time = time.perf_counter()
        valid_moves = self._get_valid_moves()

        role = "White" if player_id == 0 else "Black"

        truncated_observation = f"[GAME] You are playing as {role} in a game of Chess. Make your moves in UCI format enclosed in square brackets (e.g., [e2e4]).\n[GAME] The current board is:\n{Utils.board_with_coords(self.env.state.game_state['board'])}\n[GAME] The valid moves are: {valid_moves}."
        move, input_tokens, output_tokens, total_tokens = self._agent_call_with_retry(agent, truncated_observation, player_id, valid_moves)
        end_time = time.perf_counter()

        return move, end_time - start_time, input_tokens, output_tokens, total_tokens
    
    def _speculation_task(self, agent: Any, observation: str, player_id: int) -> Tuple[List[str], List[str], List[float], List[float], List[float], List[int], List[int], List[int], List[int], List[int], List[int]]:
        """Execute speculation: predict opponent move and prepare response"""

        role = "White" if player_id == 0 else "Black"
        valid_moves = self._get_valid_moves()

        truncated_observation = f"[GAME] You are playing as {role} in a game of Chess. Make your moves in UCI format enclosed in square brackets (e.g., [e2e4]).\n[GAME] The current board is:\n{Utils.board_with_coords(self.env.state.game_state['board'])}\n[GAME] The valid moves are: {valid_moves}."

        prediction_results = self._guess_actions(truncated_observation, retries=3)

        if prediction_results is None or prediction_results[0] is None:
            return [], [], [], [], [], [], [], [], [], [], []

        predictions = prediction_results[0]
        individual_prediction_times = [prediction_results[1]] * len(predictions)
        input_prediction_tokens = [prediction_results[2] or 0] * len(predictions)
        output_prediction_tokens = [prediction_results[3] or 0] * len(predictions)
        total_prediction_tokens = [prediction_results[4] or 0] * len(predictions)

        # Remove None predictions and get unique predictions
        valid_predictions = [p for p in predictions]
        valid_prediction_times = [individual_prediction_times[i] for i in range(len(individual_prediction_times))]

        if not valid_predictions:
            return [], [], [], [], [], [], [], [], [], [], []

        # Simulate the predicted moves in parallel
        with ThreadPoolExecutor(max_workers=len(valid_predictions)) as executor:
            speculation_futures = [
                executor.submit(self._simulate_and_speculate, agent, observation, player_id, prediction)
                for prediction in valid_predictions
            ]
            speculations_results = [future.result() for future in speculation_futures]

        print(f"Speculations: {speculations_results}")
        
        speculations: List[str] = []
        individual_speculation_times: List[float] = []
        input_speculation_tokens: List[int] = []
        output_speculation_tokens: List[int] = []
        total_speculation_tokens: List[int] = []

        for speculation in speculations_results:
            if speculation is not None and speculation[0] is not None:
                speculations.append(speculation[0])
                individual_speculation_times.append(speculation[1])
                input_speculation_tokens.append(speculation[2])
                output_speculation_tokens.append(speculation[3])
                total_speculation_tokens.append(speculation[4])
        
        total_times: List[float] = []
        for i in range(len(valid_prediction_times)):
            total_times.append(valid_prediction_times[i] + individual_speculation_times[i])

        return valid_predictions, speculations, valid_prediction_times, individual_speculation_times, total_times, input_prediction_tokens, output_prediction_tokens, total_prediction_tokens, input_speculation_tokens, output_speculation_tokens, total_speculation_tokens

    def _simulate_and_speculate(
        self, 
        agent: Any, 
        observation: str, 
        player_id: int, 
        predicted_move: str
    ) -> Tuple[Optional[str], float, int, int, int]:
        """Simulate predicted move and generate speculative response"""

        start_time_speculate = time.perf_counter()
        
        # Execute predicted move on board
        move_uci = predicted_move.lower().replace("[", "").replace("]", "")
        predicted_chess_move = chess.Move.from_uci(move_uci)
        # Make a copy of the board and push the move to the copy
        board_copy = self.env.state.game_state["board"].copy()
        board_copy.push(predicted_chess_move)
        
        # Build new observation with board state
        board_str = Utils.board_with_coords(board_copy)     
        valid_moves = [f'[{move.uci()}]' for move in board_copy.legal_moves]
        valid_moves_in_string = ", ".join(valid_moves)
        
        spec_role = "White" if player_id == 1 else "Black"
        new_observation = f"[GAME] You are playing as {spec_role} in a game of Chess. Make your moves in UCI format enclosed in square brackets (e.g., [e2e4]).\n[GAME] The current board is:\n{board_str}\n[GAME] The valid moves are: {valid_moves_in_string}."

        
        # Get speculative move
        speculation, input_tokens, output_tokens, total_tokens = self._agent_call_with_retry(agent, new_observation, player_id, valid_moves)
        
        if self.logger:
            self.logger.log("SIMULATION SPECULATION OUTPUT", speculation)
        
        return speculation, time.perf_counter() - start_time_speculate, input_tokens, output_tokens, total_tokens
    
    def _execute_game_loop(
        self,
        agents: Dict[int, Any],
        stop_after: Optional[int] = None
    ) -> Tuple[Dict[int, Any], Any, Any, float]:
        """Main game execution loop"""
        self.env.reset(num_players=self.config.num_chess_players)
        
        steps_info = {}
        step_count = 0
        done = False

        current_agent = agents[0]
        other_agent = agents[1]
        
        regular_time = 0.0


        # Initialize game state
        player_id, observation = self.env.get_observation()
        
        while not done:

            player_id, observation = self.env.get_observation()


            current_future = self._current_agent_task (current_agent, observation, player_id)
                    
            current_move, time_taken1, input_tokens1, output_tokens1, total_tokens1 = current_future
                
            # Update timing counters
            regular_time += time_taken1
            
            # Record step information
            steps_info[step_count] = {
                "player_id": player_id,
                "current_observation": observation,
                "current_move": current_move,
                "time_taken_current_agent": time_taken1,
                "input_tokens_current_agent": input_tokens1,
                "output_tokens_current_agent": output_tokens1,
                "total_tokens_current_agent": total_tokens1,
            }
            
            if self.logger:
                self.logger.log("INFO", f"STEP {step_count}:", Utils.dict_to_str(steps_info[step_count]))
                self.logger.log('-' * 100)
            
            # Execute move
            done, info = self.env.step(current_move)
            step_count += 1
            
            if stop_after and step_count >= stop_after:
                break
            
            # Swap agents for next turn
            current_agent, other_agent = other_agent, current_agent
        
        rewards, game_info = self.env.close()
        return steps_info, rewards, game_info, regular_time
    
    def run(self, stop_after: int = 20) -> None:
        """Run a complete chess game with speculative execution"""
        # Setup run
        self.current_run_id = str(uuid.uuid4())
        self.logger = GameLogger(self.base_traj_path, self.current_run_id)
        
        current_dir_path = join(self.base_traj_path, self.current_run_id)
        
        # Create agents
        agents = self.agent_manager.create_agents(self.agent0_name, self.agent1_name)
        
        self.logger.log("INFO", f"Starting run {self.current_run_id} with agents: {self.agent0_name} and {self.agent1_name}")
        
        try:
            # Execute game
            steps_info, rewards, game_info, regular_time = self._execute_game_loop(
                agents, stop_after
            )
            
            # Save results
            Utils.save_json(steps_info, join(current_dir_path, "stepsinfo.json"))
            Utils.save_json(rewards, join(current_dir_path, "rewards.json"))
            Utils.save_json(game_info, join(current_dir_path, "game_info.json"))
            Utils.save_json(regular_time, join(current_dir_path, "time_checker_regular.json"))
            
            self.logger.log("INFO", f"Run completed for {self.current_run_id}")
            
        except Exception as e:
            if self.logger:
                self.logger.log("ERROR", str(e))
            raise


def main():
    """Main execution function"""
    import argparse
    p = argparse.ArgumentParser(description="Run regular chess (generate trajectories without speculation).")
    p.add_argument("--config", default="config.yml", help="Path to config YAML (default: config.yml)")
    p.add_argument("--trajectories-dir", default=None, help="Output directory for trajectories (overrides config)")
    p.add_argument("--stop-after", type=int, default=None, help="Stop after N steps (default: from config)")
    args = p.parse_args()

    config = Config(args.config)
    if args.trajectories_dir is not None:
        config.trajectories_path = args.trajectories_dir.rstrip("/")

    runner = RegularChessRunner(config=config)
    stop_after = args.stop_after if args.stop_after is not None else config.stop_after
    start_time = time.time()
    runner.run(stop_after=stop_after)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()