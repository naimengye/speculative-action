import json
import unittest

from src.runner import HotPotQARun
from src.wrappers import DATA_DIR, HOTPOTQA_SPLIT_FILE


def make_env():
    runner = HotPotQARun(
        model_name="agent",
        guess_model_name="guess",
        to_print_output=False,
        agent_base_url="http://agent.invalid/v1",
        guess_base_url="http://guess.invalid/v1",
    )
    return runner.env


class EnvResetTest(unittest.TestCase):
    """The runner draws question indices with a seeded shuffle and files each log under its index."""

    @classmethod
    def setUpClass(cls):
        with open(f"{DATA_DIR}/{HOTPOTQA_SPLIT_FILE['dev']}") as f:
            cls.dev = json.load(f)

    def test_reset_asks_the_requested_question(self):
        env = make_env()
        for idx in (0, 6, 7107):
            self.assertEqual(env.reset(idx=idx), f"Question: {self.dev[idx]['question']}")

    def test_runner_reaches_inner_attributes_through_wrappers(self):
        # runner.step reads env.sim_obs and runner.run calls env.write() on the outer wrapper.
        env = make_env()
        env.reset(idx=0)
        self.assertIsNone(env.sim_obs)
        self.assertTrue(callable(env.write))


if __name__ == "__main__":
    unittest.main()
