import os
from functools import cached_property

import openai
from google import genai
from google.genai import types

from . import constants

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _api_key(name, env_var):
    # The environment wins over constants.py.
    return os.environ.get(env_var) or getattr(constants, name, None)


class LLMClient:
    """Unified LLM client supporting Gemini, OpenAI, OpenRouter and OpenAI-compatible servers.

    Clients are created on first use, so a run only needs the key of the provider it calls.
    With base_url set, every call goes to that server and the model name is sent as is.
    """

    def __init__(self, model_name, temperature, max_tokens, top_p, base_url=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.base_url = base_url

    @cached_property
    def gemini_client(self):
        return genai.Client(api_key=_api_key("gemini_api_key", "GEMINI_API_KEY"))

    @cached_property
    def openai_client(self):
        return openai.OpenAI(api_key=_api_key("openai_api_key", "OPENAI_API_KEY"))

    @cached_property
    def openrouter_client(self):
        return openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=_api_key("openrouter_api_key", "OPENROUTER_API_KEY"),
        )

    @cached_property
    def local_client(self):
        # vLLM and SGLang accept any key unless started with --api-key.
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=_api_key("local_api_key", "LOCAL_API_KEY") or "EMPTY",
        )

    def call(self, prompt, stop=None):
        if self.base_url:
            return self._local_call(prompt, stop)
        if self.model_name.startswith("gemini"):
            return self._gemini_call(prompt, stop)
        elif self.model_name.startswith("gpt"):
            return self._openai_call(prompt, stop)
        else:
            return self._openrouter_call(prompt, stop)

    def _gemini_call(self, prompt, stop):
        if stop is not None:
            config = types.GenerateContentConfig(stop_sequences=stop)
        else:
            config = types.GenerateContentConfig()
        response = self.gemini_client.models.generate_content(
            model=self.model_name, contents=prompt, config=config
        )
        return str(response.text)

    def _chat_completion(self, client, prompt, extra_body=None):
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            extra_body=extra_body,
        )
        return response.choices[0].message.content

    def _openai_call(self, prompt, stop):
        return self._chat_completion(self.openai_client, prompt)

    def _openrouter_call(self, prompt, stop):
        return self._chat_completion(self.openrouter_client, prompt)

    def _local_call(self, prompt, stop):
        return self._chat_completion(
            self.local_client, prompt, extra_body=constants.local_extra_body
        )
