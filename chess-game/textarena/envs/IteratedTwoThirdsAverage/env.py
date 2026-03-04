import re
from typing import Optional, Dict, Any, Tuple

import textarena as ta

class IteratedTwoThirdsAverageEnv(ta.Env):
    def __init__(self, num_rounds: int=5, min_guess: float=0.0, max_guess: float=100.0):
        self.num_rounds = num_rounds
        self.min_guess = min_guess
        self.max_guess = max_guess

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        self.state.reset(game_state={"round": 1, "points": {0:0, 1:0}, "guesses": {}, "history": []}, player_prompt_function=self._prompt)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in a {self.num_rounds}-round of IteratedTwoThirdsAverage.\nEach round, guess a number between {self.min_guess} and {self.max_guess}.\n"
            "After both guesses, the target is (2/3)x(average of both guesses),\nand the player whose guess is closest to the target wins that round.\nReply with your guess in the format '[<number>]'."
        )

    def get_board_str(self) -> str:
        s = f"Round {self.state.game_state['round']}/{self.num_rounds}\n"
        if self.state.game_state["history"]:
            s += "History:\n"
            for i, past in enumerate(self.state.game_state["history"], start=1): s += (f"  Round {i}: " + ", ".join(f"P{pid}→{guess}" for pid, guess in past.items()) + "\n")
        return s

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        pid = self.state.current_player_id
        m = re.compile(r"\[\s*([0-9]+(?:\.[0-9]*)?)\s*\]").search(action)
        if not m: self.state.set_invalid_move(reason="Invalid format; please submit your guess as “[<number>]”.")
        else:
            guess = float(m.group(1))
            if not (self.min_guess <= guess <= self.max_guess): self.state.set_invalid_move(reason=f"Guess must be between {self.min_guess} and {self.max_guess}.")
            else: # accept guess
                self.state.game_state["guesses"][pid] = guess
                if len(self.state.game_state["guesses"]) == 2:
                    guesses = self.state.game_state["guesses"]
                    avg = sum(guesses.values()) / 2.0
                    target = (2.0 / 3.0) * avg
                    d0 = abs(guesses[0] - target); d1 = abs(guesses[1] - target) # compute distances
                    self.state.game_state["history"].append(guesses.copy()) # update history
                    self.state.add_observation(message=f"Player 0 guessed {guesses[0]}; Player 1 guessed {guesses[1]}. Thus the Target is: {target:.2f}.", observation_type=ta.ObservationType.GAME_MESSAGE)
                    # decide round winner
                    if d0 < d1:     winner = 0
                    elif d1 < d0:   winner = 1
                    else:           winner = None
                    if winner is None: self.state.add_observation(message="Round is a draw.", observation_type=ta.ObservationType.GAME_MESSAGE)
                    else:
                        self.state.game_state["points"][winner] += 1
                        self.state.add_observation(message=f"Player {winner} wins the round!", observation_type=ta.ObservationType.GAME_MESSAGE)
                    # prepare next round
                    self.state.game_state["round"] += 1
                    self.state.game_state["guesses"].clear()
                    # check end-of-game
                    if self.state.game_state["round"] > self.num_rounds:
                        p0 = self.state.game_state["points"][0]
                        p1 = self.state.game_state["points"][1]
                        if p0 > p1:     self.state.set_winner(player_id=0, reason="Player 0 won more rounds.")
                        elif p1 > p0:   self.state.set_winner(player_id=1, reason="Player 1 won more rounds.")
                        else:           self.state.set_draw(reason="Overall game is a draw.")
        return self.state.step()
