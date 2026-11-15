"""The runners keep going when a model reply or a guess can't be used.

A local stub stands in for an OpenAI-compatible server. It plays the first legal
move and guesses the first three, and each test switches on one kind of bad reply.
Run from chess-game/: python -m unittest discover tests
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
from textarena.agents.basic_agents import OpenAIAgent

MODEL = "gpt-oss-20b"


class _Stub(BaseHTTPRequestHandler):
    lock = threading.Lock()
    guess_reply = None     # fixed guess text in place of the first three legal moves
    no_move_for_black = 0  # the next N replies to a Black move prompt name no move
    empty_replies = 0      # the next N replies have no text (content is null)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        listed = prompt[prompt.find("The valid moves are:"):].split("\n", 1)[0]
        moves = re.findall(r"[a-h][1-8][a-h][1-8][qrbn]?", listed)
        is_guess = "candidates" in prompt
        stub = type(self)
        with stub.lock:
            if stub.empty_replies > 0:
                stub.empty_replies -= 1
                content = None
            elif is_guess:
                content = stub.guess_reply or ", ".join(f"[{m}]" for m in moves[:3])
            elif "playing as Black" in prompt and stub.no_move_for_black > 0:
                stub.no_move_for_black -= 1
                content = "I am not sure."
            else:
                content = f"[{moves[0]}]" if moves else ""
        payload = json.dumps({
            "id": "stub", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "length" if content is None else "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class SpeculationErrorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"
        cls.tmp = tempfile.TemporaryDirectory()
        with open("config.yml") as f:
            cfg = yaml.safe_load(f)
        cfg["api"]["openai"]["key"] = "EMPTY"
        cfg["api"]["openai"]["base_url"] = cls.base_url
        cfg["api"]["openrouter"]["key"] = "unused"
        cfg["models"]["openai"]["main"] = cfg["models"]["openai"]["guess"] = MODEL
        cfg["guess"]["model_name"] = MODEL
        cfg["game"]["agent_name0"] = cfg["game"]["agent_name1"] = "OpenAI"
        cls.cfg = cfg
        cls.config_path = cls._write_config("config.yml", cfg)

    @classmethod
    def _write_config(cls, name, cfg):
        path = os.path.join(cls.tmp.name, name)
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        return path

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

    def setUp(self):
        _Stub.guess_reply = None
        _Stub.no_move_for_black = _Stub.empty_replies = 0

    def _run(self, module, runner_cls, config_path=None, moves=4):
        config = module.Config(config_path or self.config_path)
        config.trajectories_path = self.tmp.name
        runner_cls(config).run(stop_after=moves)

    def test_reply_without_text_reads_as_empty(self):
        # Reasoning models return content=None when they run out of tokens; this used to raise AttributeError.
        _Stub.empty_replies = 1
        agent = OpenAIAgent(model_name=MODEL, api_key="EMPTY", base_url=self.base_url)
        text, *_ = agent._make_request("[GAME] The valid moves are: ['[e2e4]'].")
        self.assertEqual(text, "")

    def test_branch_with_no_legal_move_is_dropped(self):
        # The three branches of the first window play Black; 9 bad replies use up their 3 attempts each.
        # The guess used to stay in the list without its branch, and the time loop raised IndexError.
        _Stub.no_move_for_black = 9
        self._run(Speculative_Chess, Speculative_Chess.SpeculativeChessRunner)
        self.assertEqual(_Stub.no_move_for_black, 0)

    def test_guess_that_is_not_legal_is_skipped(self):
        # e3e4 isn't a legal move from the start position; board.push used to raise in its branch.
        _Stub.guess_reply = "[e3e4], [a2a3], [b2b3]"
        self._run(Speculative_Chess, Speculative_Chess.SpeculativeChessRunner)

    def test_runs_with_only_an_openai_key(self):
        # With openai>=3 the old code raised while building the unused OpenRouter client.
        cfg = dict(self.cfg, api={"openai": self.cfg["api"]["openai"], "openrouter": {"key": ""}})
        path = self._write_config("openai_only.yml", cfg)
        self._run(regular_chess, regular_chess.RegularChessRunner, path, moves=2)
        self._run(Speculative_Chess, Speculative_Chess.SpeculativeChessRunner, path, moves=2)


if __name__ == "__main__":
    unittest.main()
