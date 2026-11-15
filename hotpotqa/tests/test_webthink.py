import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.runner import HotPotQARun

FINISH = "I can answer this directly.\nAction 1: Finish[yes]"


class _FinishStub(BaseHTTPRequestHandler):
    """OpenAI-compatible endpoint whose every reply finishes the episode."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps({
            "id": "stub",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": FINISH},
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


class SimulateTest(unittest.TestCase):

    def test_finishing_before_any_search(self):
        # Only a search sets sim_obs, so a Finish on the first step used to raise AttributeError.
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FinishStub)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/v1"
            runner = HotPotQARun("agent", "guess", to_print_output=False, agent_base_url=url, guess_base_url=url)
            info = runner.webthink(0, prompt="", to_print=False, n=3, simulate=True)
        finally:
            server.shutdown()
        self.assertEqual(info["answer"], "yes")


if __name__ == "__main__":
    unittest.main()
