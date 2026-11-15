import json
import os

openrouter_api_key = "your-openrouter-api-key"
openrouter_model_name = "openai/gpt-4"
openrouter_guess_model_name = "openai/gpt-5-nano"

prompts_folder = "./prompts/"
prompt_file = "prompts_naive.json"
agent_role = "Question Answering Agent"

random_seed = 248
num = 7405
n_steps_to_run = 8
n_samples_to_run = 20
client_error_sleep_time = 60
server_error_sleep_time = 60
trajectory_filenames = ["log.txt", "normalobs.json", "simobs.json", "metrics.json"]
guess_num_actions = 3
max_agent_retries = 1
max_guess_retries = 3

# Agent LLM settings
max_output_tokens = 1000
top_p = 1
temperature = 1

# Guess LLM settings
max_guess_output_tokens = 100
guess_top_p = 0.9
guess_temperature = 0.1

# Optional OpenAI-compatible endpoints (e.g. local vLLM or SGLang servers), one per role.
agent_base_url = os.environ.get("AGENT_BASE_URL")
guess_base_url = os.environ.get("GUESS_BASE_URL")
# Extra request fields for those endpoints, e.g. to turn off Qwen3 thinking:
# LOCAL_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
local_extra_body = json.loads(os.environ.get("LOCAL_EXTRA_BODY", "null"))
