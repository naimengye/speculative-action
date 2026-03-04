import os
import sys

# Ensure chess-game root is on path so "import textarena" in Speculative_Chess resolves
_chess_game_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _chess_game_root not in sys.path:
    sys.path.insert(0, _chess_game_root)

from openai import OpenAI
import json
from Speculative_Chess import ChessActionCleaner
from typing import Optional, Tuple, List
import time
import yaml
import argparse


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file. Uses chess-game/config.yml when run from speculative_workflow/."""
    if config_path is None:
        path = os.path.join(_chess_game_root, "config.yml")
        if not os.path.isfile(path):
            path = "config.yml"
    else:
        path = config_path
    with open(path, "r") as f:
        return yaml.safe_load(f)

def truncate_chess_observation(observation_string: str) -> str:
    """
    Extract the last board state and valid moves from a chess observation string.
    """
    lines = observation_string.strip().split('\n')
    
    # Find the last board representation
    last_board_start = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Current board:" in lines[i]:
            last_board_start = i
            break
    
    if last_board_start == -1:
        return "No board found in observation"
    
    # Find where the board ends (after the coordinate line "a b c d e f g h")
    board_end = -1
    for i in range(last_board_start + 1, len(lines)):
        if "a b c d e f g h" in lines[i]:
            board_end = i
            break
    
    if board_end == -1:
        return "Could not find end of board"
    
    valid_moves_line = ""
    for i in range(board_end + 1, len(lines)):
        if "Valid moves:" in lines[i]:
            valid_moves_line = lines[i]
            break
    
    truncated_lines = []
    
    if lines[0].startswith("[GAME] You are playing"):
        truncated_lines.append(lines[0])
        truncated_lines.append(lines[1])  # Move instruction line
    
    for i in range(last_board_start, board_end + 1):
        truncated_lines.append(lines[i])
    
    if valid_moves_line:
        truncated_lines.append(valid_moves_line)
    
    return '\n'.join(truncated_lines)


def call_guess_llm(
    prompt: str,
    model_name: str,
    provider: str,
    config: dict,
    retries: int = 3,
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """
    Call LLM for guess prediction with retry logic. Returns the response text, input tokens, output tokens, and total tokens.
    """
    for attempt in range(retries):
        try:
            if provider == "openrouter":
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=config["api"]["openrouter"]["key"]
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": config["prompts"]["standard_game"]},
                        {"role": "user", "content": prompt}
                    ],
                    reasoning_effort="low"
                )
            else:  # openai
                client = OpenAI(api_key=config["api"]["openai"]["key"])
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": config["prompts"]["standard_game"]},
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

        except Exception as e:
            print(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                return None, None, None, None

    return None, None, None, None


def guess_actions(
    observation: str,
    config: dict,
    retries: int = 3,
) -> Tuple[Optional[List[str]], float, Optional[int], Optional[int], Optional[int]]:
    """
    Generate guess predictions for next moves. Reads num_guesses, model, provider from config.
    """
    num_guesses = config["game"]["num_guesses"]
    guess_model_name = config["guess"]["model_name"]
    guess_provider = config["guess"]["provider"]

    start_pred_time = time.perf_counter()
    prompt = observation + config["prompts"]["guess"].format(num_guesses=config["game"]["num_guesses"])
    raw_output, input_tokens, output_tokens, total_tokens = call_guess_llm(
        prompt=prompt,
        model_name=guess_model_name,
        provider=guess_provider,
        config=config,
        retries=retries,
    )
    end_pred_time = time.perf_counter()
    prediction_time = end_pred_time - start_pred_time

    return ChessActionCleaner.clean_actions(raw_output), prediction_time, input_tokens, output_tokens, total_tokens


def process_trajectory_with_guesses(
    steps_info_path: str,
    config: dict,
    retries: int = 3,
    add_spec: bool = False,
) -> str:
    """
    Process a speculative trajectory file with guesses. Reads game/guess settings from config.
    """
    num_guesses = config["game"]["num_guesses"]
    new_step_info: dict = {}
    
    with open(steps_info_path, 'r') as f:
        step_info = json.load(f)

    prev_match = False
    next_match = False
    
    # Process each step
    for step in step_info:
        print(f"Processing step: {step}")
        truncated_observation = truncate_chess_observation(step_info[step]["current_observation"])
        if next_match:
            prev_match = True
            next_match = False

        if (not add_spec) or (add_spec and len(step_info[step]["current_pred"]) >= num_guesses):
            predictions = step_info[step]["current_pred"]
            prediction_time_list = step_info[step]["time_taken_prediction"]
            prediction_time = prediction_time_list[0] if prediction_time_list else []
            guess_input_tokens_list = step_info[step]["input_tokens_prediction"]
            guess_input_tokens = guess_input_tokens_list[0] if guess_input_tokens_list else []
            guess_output_tokens_list = step_info[step]["output_tokens_prediction"]
            guess_output_tokens = guess_output_tokens_list[0] if guess_output_tokens_list else []
            guess_total_tokens_list = step_info[step]["total_tokens_prediction"]
            guess_total_tokens = guess_total_tokens_list[0] if guess_total_tokens_list else []

            if step_info[step]["current_move"] in step_info[step]["current_pred"]:
                next_match = True
                idx = step_info[step]["current_pred"].index(step_info[step]["current_move"])
                next_step_time = step_info[step]["time_taken_speculation"][idx]
                next_step_input_tokens = step_info[step]["input_tokens_speculation"][idx]
                next_step_output_tokens = step_info[step]["output_tokens_speculation"][idx]
                next_step_total_tokens = step_info[step]["total_tokens_speculation"][idx]
            else:
                next_match = False
        
        elif add_spec and (len(step_info[step]["current_pred"]) < num_guesses):      
            predictions, prediction_time, guess_input_tokens, guess_output_tokens, guess_total_tokens = guess_actions(
                observation=truncated_observation,
                config=config,
                retries=retries,
            )

        if step not in new_step_info:
            new_step_info[step] = {}
        if not prev_match:
            new_step_info[step]["current_observation"] = truncated_observation
            new_step_info[step]["current_move"] = step_info[step]["current_move"]
            new_step_info[step]["time_taken_current_agent"] = step_info[step]["time_taken_current_agent"]
            new_step_info[step]["input_tokens_current_agent"] = step_info[step]["input_tokens_current_agent"]
            new_step_info[step]["output_tokens_current_agent"] = step_info[step]["output_tokens_current_agent"]
            new_step_info[step]["total_tokens_current_agent"] = step_info[step]["total_tokens_current_agent"]

            new_step_info[step]["guessed_moves"] = predictions
            new_step_info[step]["guess_prediction_time"] = prediction_time
            new_step_info[step]["guess_input_tokens"] = guess_input_tokens
            new_step_info[step]["guess_output_tokens"] = guess_output_tokens
            new_step_info[step]["guess_total_tokens"] = guess_total_tokens
        
        elif prev_match:
            prev_match = False
            new_step_info[step]["current_observation"] = truncated_observation
            new_step_info[step]["current_move"] = step_info[step]["current_move"]
            new_step_info[step]["time_taken_current_agent"] = next_step_time
            new_step_info[step]["input_tokens_current_agent"] = next_step_input_tokens
            new_step_info[step]["output_tokens_current_agent"] = next_step_output_tokens
            new_step_info[step]["total_tokens_current_agent"] = next_step_total_tokens

            new_step_info[step]["guessed_moves"] = predictions
            new_step_info[step]["guess_prediction_time"] = prediction_time
            new_step_info[step]["guess_input_tokens"] = guess_input_tokens
            new_step_info[step]["guess_output_tokens"] = guess_output_tokens
            new_step_info[step]["guess_total_tokens"] = guess_total_tokens
        
        print(f"Step {step} - Predictions: {predictions}")
        print("--------------------------------")

    output_dir = os.path.dirname(steps_info_path)
    safe_model_name = guess_model_name.replace('-', '_').replace('/', '_')
    output_filename = f"steps_info_{safe_model_name}_guess_{num_guesses}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(new_step_info, f, indent=4)
    
    print(f"Speculative trajectory with guess predictions processing complete! Output saved to: {output_path}")
    return output_path


def main():
    """
    Main function to process a speculative trajectory file with guess predictions.
    """
    p = argparse.ArgumentParser(description="Add guess predictions to a speculative trajectory (stepsinfo.json).")
    p.add_argument("--config", default=None, help="Path to config YAML (default: chess-game/config.yml or config.yml)")
    p.add_argument("--stepsinfo", "-i", default=None, help="Path to stepsinfo.json (or trajectory dir containing it)")
    p.add_argument("--num-guesses", type=int, default=None, help="Number of guesses (default: from config)")
    p.add_argument("--add-spec", action="store_true", help="Add speculative predictions for non-speculative windows")
    args = p.parse_args()

    step_info_path = args.stepsinfo
    if step_info_path is None and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        step_info_path = sys.argv[1]
    if step_info_path is None:
        step_info_path = input("Enter path to stepsinfo.json file: ").strip()
    if os.path.isdir(step_info_path):
        step_info_path = os.path.join(step_info_path, "stepsinfo.json")
    if not os.path.exists(step_info_path):
        print(f"Error: File not found: {step_info_path}")
        sys.exit(1)

    config = load_config(args.config)
    if args.num_guesses is not None:
        config.setdefault("game", {})["num_guesses"] = args.num_guesses
    output_path = process_trajectory_with_guesses(
        steps_info_path=step_info_path,
        config=config,
        add_spec=args.add_spec,
    )
    print(f"\nSpeculative trajectory with guess predictions processing complete! Output saved to:\n  {output_path}")


if __name__ == "__main__":
    main()

