"""Both runners send every call to api.openai.base_url when config.yml sets it.

A local stub stands in for an OpenAI-compatible server (vLLM, SGLang). It plays
the first legal move and guesses the first three, so a short game runs with no
network and no API key. Run from chess-game/: python -m unittest discover tests
"""
import json
import os
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

from regular_workflow import regular_chess
from speculative_workflow import Speculative_Chess

MODEL = "gpt-oss-20b"


class _ChessStub(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body))
        prompt = body["messages"][-1]["content"]
        listed = prompt[prompt.find("The valid moves are:"):].split("\n", 1)[0]
        moves = re.findall(r"[a-h][1-8][a-h][1-8][qrbn]?", listed)
        is_guess = "candidates" in prompt
        content = ", ".join(f"[{m}]" for m in moves[:3]) if is_guess else f"[{moves[0]}]"
        payload = json.dumps({
            "id": "stub", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class BaseUrlTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ChessStub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.tmp = tempfile.TemporaryDirectory()
        with open("config.yml") as f:
            cfg = yaml.safe_load(f)
        cfg["api"]["openai"]["key"] = "EMPTY"
        cfg["api"]["openai"]["base_url"] = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"
        cfg["api"]["openrouter"]["key"] = "unused"
        cfg["models"]["openai"]["main"] = cfg["models"]["openai"]["guess"] = MODEL
        cfg["guess"]["model_name"] = MODEL
        cfg["game"]["agent_name0"] = cfg["game"]["agent_name1"] = "OpenAI"
        cls.config_path = os.path.join(cls.tmp.name, "config.yml")
        with open(cls.config_path, "w") as f:
            yaml.safe_dump(cfg, f)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

    def setUp(self):
        _ChessStub.requests.clear()

    def _config(self, module):
        config = module.Config(self.config_path)
        config.trajectories_path = self.tmp.name
        return config

    def test_default_is_openai(self):
        config = regular_chess.Config("config.yml")
        self.assertEqual(config.openai_base_url, "https://api.openai.com/v1")

    def test_regular_game_uses_base_url(self):
        regular_chess.RegularChessRunner(self._config(regular_chess)).run(stop_after=4)
        self.assertEqual(len(_ChessStub.requests), 4)
        for path, body in _ChessStub.requests:
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(body["model"], MODEL)

    def test_speculative_game_uses_base_url(self):
        Speculative_Chess.SpeculativeChessRunner(self._config(Speculative_Chess)).run(stop_after=4)
        guesses = [body for _, body in _ChessStub.requests if "candidates" in body["messages"][-1]["content"]]
        self.assertTrue(guesses)
        self.assertTrue(all(body.get("reasoning_effort") == "low" for body in guesses))
        self.assertTrue(all(body["model"] == MODEL for _, body in _ChessStub.requests))


if __name__ == "__main__":
    unittest.main()
