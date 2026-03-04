from openai import OpenAI
import json
from typing import Optional, Tuple, List
from utils import truncate_chess_observation
import time
import os
import yaml
import re
import sys


def load_config(config_path: str = "config.yml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_confidence_score(raw_output: str) -> List[float]:
    """Extract the confidence scores from the raw output."""
    try:
        # Look for pattern like [75,80,90,90] in the output
        stripped = raw_output.strip()
        if "[" in stripped and "]" in stripped:
            score_str = stripped.split("[")[1].split("]")[0]
            # Split by comma and convert to floats
            scores = [float(s.strip()) for s in score_str.split(",")]
            return scores  
        else:
            # Try to extract numbers from the output as fallback
            numbers = re.findall(r'\d+', stripped)
            if numbers:
                scores = [float(num) for num in numbers]
                return scores
            return []
    except (ValueError, IndexError):
        print(f"Warning: Could not extract confidence scores from: {raw_output[:100]}")
        return []


class PromptTemplates:
    """Prompt templates for chess game interactions."""

    PREDICTION_PROMPT = (
        "Given the chess board positions from the current state, analyze how confident you are that the proposed moves are what you will play next. "
        "Consider the game context, piece positions, and strategic implications. "
        "Return a list of numerical confidence scores between 0 and 100, where: "
        "- 0-20: Very unlikely move "
        "- 21-40: Somewhat unlikely move "
        "- 41-60: Possible move "
        "- 61-80: Likely move "
        "- 81-100: Very likely/forced move "
        "IMPORTANT FORMAT: Your response must be a comma-separated list of numbers between 0 and 100 in the format [CONFIDENCE_SCORES], like: [75, 80, 90, 90]. "
        "Focus on providing an accurate assessment of your confidence in this prediction."
        "The proposed moves are: {guesses}. The current state of the board is {observation}."
    )

def call_prediction_llm(
    prompt: str,
    model_name: str,
    provider: str,
    config: dict,
    retries: int = 3,
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """
    Call LLM for confidence prediction with retry logic.

    Returns:
        Tuple of (response_text, input_tokens, output_tokens, total_tokens)
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
                        {"role": "system", "content": PromptTemplates.PREDICTION_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    reasoning_effort="low"
                )
            else:  # openai
                client = OpenAI(api_key=config["api"]["openai"]["key"])
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": PromptTemplates.PREDICTION_PROMPT},
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


def guess_action_confidence_score(
    observation: str,
    config: dict,
    retries: int = 3,
    guesses: Optional[List[str]] = None,
    ground_truth_move: Optional[str] = None,
) -> Tuple[List[float], float]:
    """
    Generate confidence scores for given guesses. Reads model/provider from config.
    """
    guess_model_name = config["guess"]["model_name"]
    guess_provider = config["guess"]["provider"]

    if guesses is None:
        raise ValueError("guesses cannot be None")
    if ground_truth_move is None:
        raise ValueError("ground_truth_move cannot be None")

    guesses.append(ground_truth_move)

    prompt = PromptTemplates.PREDICTION_PROMPT.format(guesses=guesses, observation=observation)

    print(f"Prompt: {prompt}")

    start_pred_time = time.perf_counter()
    raw_output, input_tokens, output_tokens, total_tokens = call_prediction_llm(
        prompt=prompt,
        model_name=guess_model_name,
        provider=guess_provider,
        config=config,
        retries=retries,
    )
    print("--------------------------------")
    print(f"Raw output: {raw_output}")
    print("--------------------------------")
    end_pred_time = time.perf_counter()
    prediction_time = end_pred_time - start_pred_time

    return extract_confidence_score(raw_output), prediction_time


def process_trajectory_with_guesses(
    guess_info_path: str,
    config: dict,
    retries: int = 3,
) -> str:
    """
    Process a guess info file with confidence predictions. Reads game/guess settings from config.
    """
    num_guesses = config["game"]["num_guesses"]
    guess_model_name = config["guess"]["model_name"]
    guess_provider = config["guess"]["provider"]
    new_step_info: dict = {}

    # Load the guess_info file
    with open(guess_info_path, 'r') as f:
        guess_info = json.load(f)



    # Save the new steps_info file
    output_dir = os.path.dirname(guess_info_path)
    # Replace both '-' and '/' with '_' to avoid directory path issues
    safe_model_name = guess_model_name.replace('-', '_').replace('/', '_')
    output_filename = f"steps_info_{safe_model_name}_confidence_prediction.json"
    output_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            new_step_info = json.load(f)
    else:
        new_step_info = {}
    

    # Process each step
    for step in guess_info:
        print(f"Processing step: {step}")
        truncated_observation = truncate_chess_observation(guess_info[step]["current_observation"])


        ground_truth_move = guess_info[step]["current_move"]
        guessed_moves = guess_info[step]["guessed_moves"]

        #TODO: hack
        if "confidence_scores" in new_step_info[step]:
            confidence_scores = new_step_info[step]["confidence_scores"]
            prediction_times = new_step_info[step]["prediction_times"]
        else:
            # Make guesses
            confidence_scores, prediction_times = guess_action_confidence_score(
                observation=truncated_observation,
                config=config,
                retries=retries,
                guesses=guessed_moves,
                ground_truth_move=ground_truth_move,
            )

        new_step_info[step] = guess_info[step]
        
        
        # Add the guesses to the step_info
        new_step_info[step]["confidence_scores"] = confidence_scores
        new_step_info[step]["prediction_times"] = prediction_times
        new_step_info[step]["guessed_moves + ground_truth_move"] = guessed_moves + [ground_truth_move]
        
        print(f"Step {step} - Confidence Scores: {confidence_scores}")
        print(f"Step {step} - Prediction Times: {prediction_times}")
        print("--------------------------------")
    
        # update the output file with the current new step info
        with open(output_path, 'w') as f:
            json.dump(new_step_info, f, indent=4)

    
    # Check if file exists and overwrite it
    with open(output_path, 'w') as f:
        json.dump(new_step_info, f, indent=4)
    
    print(f"Saved new steps_info file to: {output_path}")
    return output_path


def main():
    """
    Main function to process a guess info file with confidence predictions.

    Usage:
        python confidence-prediction.py [--config CONFIG] [--input PATH]
    """
    import argparse
    p = argparse.ArgumentParser(description="Add confidence scores to a steps_info*_guess_*.json file.")
    p.add_argument("--config", default="config.yml", help="Path to config YAML (default: config.yml)")
    p.add_argument("--input", "-i", default=None, help="Path to steps_info*_guess_*.json (or dir containing it)")
    args = p.parse_args()

    guess_info_path = args.input
    if guess_info_path is None:
        guess_info_path = input("Enter path to guess info .json file: ").strip()
    if guess_info_path and os.path.isdir(guess_info_path):
        import glob
        candidates = glob.glob(os.path.join(guess_info_path, "steps_info*_guess_*.json"))
        if not candidates:
            print(f"Error: No steps_info*_guess_*.json found in {guess_info_path}")
            sys.exit(1)
        guess_info_path = candidates[0]
    if not guess_info_path or not os.path.exists(guess_info_path):
        print(f"Error: File not found: {guess_info_path}")
        sys.exit(1)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print("Processing trajectory with confidence predictions...")
    output_path = process_trajectory_with_guesses(
        guess_info_path=guess_info_path,
        config=config,
        retries=3,
    )
    print(f"\n✓ Processing complete! Output saved to:\n  {output_path}")


if __name__ == "__main__":
    main()

