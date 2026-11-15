import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from src import constants
from src.llm_client import LLMClient
from src.runner import HotPotQARun

KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
STUB_REPLY = "Thought 1: stub. Action 1: Search[Columbia University]"


class _ChatStub(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /chat/completions endpoint that records requests."""

    received = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append((self.path, body))
        payload = json.dumps({
            "id": "stub",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": STUB_REPLY},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class LocalEndpointTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatStub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _ChatStub.received.clear()

    def test_call_goes_to_base_url_with_model_name_unchanged(self):
        client = LLMClient("Qwen/Qwen3-1.7B", 0.1, 50, 0.9, base_url=self.base_url)
        self.assertEqual(client.call("hello"), STUB_REPLY)
        path, body = _ChatStub.received[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["model"], "Qwen/Qwen3-1.7B")
        self.assertEqual(body["max_tokens"], 50)
        self.assertEqual(body["messages"][-1]["content"], "hello")

    def test_local_extra_body_is_sent(self):
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        with mock.patch.object(constants, "local_extra_body", extra):
            LLMClient("Qwen/Qwen3-1.7B", 0.1, 50, 0.9, base_url=self.base_url).call("hi")
        self.assertEqual(_ChatStub.received[-1][1]["chat_template_kwargs"], extra["chat_template_kwargs"])


class ProviderRoutingTest(unittest.TestCase):

    def test_constructs_without_any_api_key(self):
        env = {k: v for k, v in os.environ.items() if k not in KEY_VARS}
        with mock.patch.dict(os.environ, env, clear=True):
            client = LLMClient("openai/gpt-4", 1, 10, 1)
        self.assertIsNone(client.base_url)

    def test_hosted_routing_is_unchanged(self):
        cases = {
            "gemini-2.5-flash": "_gemini_call",
            "gpt-4": "_openai_call",
            "openai/gpt-4": "_openrouter_call",
        }
        for model_name, method in cases.items():
            with self.subTest(model_name=model_name), \
                    mock.patch.object(LLMClient, method, return_value="ok") as patched:
                self.assertEqual(LLMClient(model_name, 1, 10, 1).call("hi"), "ok")
                patched.assert_called_once()


class RunnerWiringTest(unittest.TestCase):

    def test_base_urls_reach_agent_and_speculator(self):
        runner = HotPotQARun(
            model_name="Qwen/Qwen3-14B",
            guess_model_name="Qwen/Qwen3-1.7B",
            to_print_output=False,
            agent_base_url="http://agent.invalid/v1",
            guess_base_url="http://guess.invalid/v1",
        )
        self.assertEqual(runner.llm.base_url, "http://agent.invalid/v1")
        self.assertEqual(runner.env.unwrapped.guess_llm.base_url, "http://guess.invalid/v1")


if __name__ == "__main__":
    unittest.main()
