import re, random
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.envs.LiarsDice.renderer import create_board_str


class LiarsDiceEnv(ta.Env):
    def __init__(self, num_dice: int = 5):
        """
        Args:
            num_dice (int): Initial number of dice each player starts with.
        """
        self.num_dice = num_dice

    def get_board_str(self): return create_board_str(game_state=self.state.game_state)
    def reset(self, num_players:int, seed: Optional[int] = None):
        assert 2<=num_players<=15, f"The number of players has to be 2<=x<=15, received {num_players}"
        self.state = ta.FFAMultiPlayerState(num_players=num_players, seed=seed)
        remaining_dice = {pid: self.num_dice for pid in range(self.state.num_players)}
        game_state = {"current_bid": {"quantity": 0, "face_value": 0}, "last_bidder_id": None, "remaining_dice": remaining_dice, "dice_rolls": None}
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._roll_new_dice()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in an {self.state.num_players}-player Liar's Dice game.\n"
            "Rules:\n- On your turn, you may either:\n  1) Make a new bid with a higher quantity or higher face (or both) than the current bid; i.e. '[Bid: 3, 4]',\n  2) Call the last bid by typing '[Call]'.\n\n"
            "If you call:\n  - If the actual count of that face value among all dice is less than the bid, the last bidder loses one die.\n"
            "  - Otherwise, the caller loses one die.\nA player who reaches 0 dice is eliminated. The last remaining player wins."
        )

    def _roll_new_dice(self):
        self.state.game_state["current_bid"] = {"quantity": 0, "face_value": 0}
        self.state.game_state["last_bidder_id"] = None
        # Roll new dice only for players still holding dice
        new_dice_rolls = {}
        for pid, count in self.state.game_state["remaining_dice"].items():
            new_dice_rolls[pid] = [random.randint(1, 6) for _ in range(count)]
        self.state.game_state["dice_rolls"] = new_dice_rolls
        for pid, rolled in new_dice_rolls.items(): # Send each player their new private dice
            message = "\nNew round - Remaining dice: " + "; ".join([f"\tPlayer {p}: {d}" for p, d in self.state.game_state["remaining_dice"].items()]) + f"\nYour current Dice are: {', '.join(map(str, rolled))}"
            self.state.add_observation(to_id=pid, message=message, observation_type=ta.ObservationType.GAME_BOARD)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        # 1. Check if action is '[Call]'
        if re.compile(r"\[call\]", re.IGNORECASE).search(action):
            current_bid = self.state.game_state["current_bid"]
            last_bidder_id = self.state.game_state["last_bidder_id"]

            if last_bidder_id is None or current_bid["quantity"] == 0: # No existing bid to call
                self._handle_invalid_move(reason="Call made with no prior bid.")
                return self.state.step(rotate_player=False)

            # Count how many dice across all players match face_value
            total_face_count = 0
            for pid, dice_list in self.state.game_state["dice_rolls"].items():
                total_face_count += dice_list.count(current_bid['face_value'])

            if total_face_count < current_bid["quantity"]: # If the actual count is lower, last bidder was bluffing -> last bidder loses a die
                loser_id = last_bidder_id
                msg = f"Player {self.state.current_player_id} calls! The actual count of face {current_bid['face_value']} is {total_face_count}, which is LESS than {current_bid['quantity']}.\nPlayer {loser_id} (the last bidder) loses one die."
            else: # Otherwise, the caller loses a die
                loser_id = self.state.current_player_id
                msg = f"Player {self.state.current_player_id} calls! The actual count of face {current_bid['face_value']} is {total_face_count}, which is >= {current_bid['quantity']}.\nPlayer {loser_id} (the caller) loses one die."

            self._apply_die_loss(loser_id, msg)
            self._rotate_players()
            return self.state.step(rotate_player=False)

        # 2. Otherwise, check if it is a valid '[Bid: X, Y]'
        bid_match = re.compile(r"\[bid\s*:?\s*(\d+)[,\s]+(\d+)\]", re.IGNORECASE).search(action)
        if bid_match:
            new_quantity = int(bid_match.group(1))
            new_face_value = int(bid_match.group(2))
            is_valid, reason = self._is_valid_bid(new_quantity, new_face_value, self.state.game_state["current_bid"]) # Validate it is strictly higher
            if is_valid:
                self.state.game_state["current_bid"] = {"quantity": new_quantity, "face_value": new_face_value}
                self.state.game_state["last_bidder_id"] = self.state.current_player_id
                self.state.add_observation(message=f"Player {self.state.current_player_id} bids {new_quantity} of face {new_face_value}.", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
                self._rotate_players()
            else: 
                self._handle_invalid_move(reason=f"Invalid bid: {reason}")
            
            return self.state.step(rotate_player=False)

        # 3. If neither a valid call nor bid, it's invalid
        self._handle_invalid_move(reason=f"Action not recognized as either a valid '[Bid: X, Y]' or '[Call]'. Submitted action: {action}")
        return self.state.step(rotate_player=False)
    
    def _handle_invalid_move(self, reason: str):
        # raise invalid move to state and check if the player is eliminated
        was_eliminated = self.state.set_invalid_move(reason=reason)
        if was_eliminated:
            # need to handle environment. Rotate players and start next round
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=f"Player {self.state.current_player_id} was eliminated by invalid move.", observation_type=ta.ObservationType.GAME_MESSAGE)
            self.state.game_state["remaining_dice"][self.state.current_player_id] = 0
            # self.state.add_elimination(self.state.current_player_id)
            self._roll_new_dice()
            self._rotate_players()

    def _rotate_players(self):
        next_pid = self.state.next_alive_player()
        if next_pid is None or len(self.state.elimination_order)>=(self.state.num_players-1): 
            self._set_outcome()
        else: 
            self.state.manually_set_current_player_id(new_player_id=next_pid, force=True)

    def _apply_die_loss(self, loser_id: int, message: str):
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, observation_type=ta.ObservationType.GAME_MESSAGE)
        self.state.game_state["remaining_dice"][loser_id] -= 1
        if self.state.game_state["remaining_dice"][loser_id] == 0: self.state.add_elimination(pid=loser_id) # check if alive
        self._roll_new_dice() # roll dice for next round

    def _is_valid_bid(self, new_quantity: int, new_face_value: int, current_bid: Dict[str, int]) -> Tuple[bool, str]:
        # Standard Liar's Dice rule: new bid must be "higher" in either quantity or face  You cannot lower either value, and the new bid can't be exactly the same.
        if new_quantity < current_bid['quantity']: return False, f"New quantity {new_quantity} is lower than current {current_bid['quantity']}."
        if new_face_value < current_bid['face_value']: return False, f"New face value {new_face_value} is lower than current {current_bid['face_value']}."
        if new_quantity == current_bid['quantity'] and new_face_value == current_bid['face_value']: return False, "Bid is identical to the current bid."
        if not (1 <= new_face_value <= 6): return False, "Face value must be between 1 and 6."
        return True, ""

    def _set_outcome(self):
        final_ranking = self.state.elimination_order + [pid for pid, count in self.state.game_state["remaining_dice"].items() if count > 0]
        reward_dict = {}
        for rank, pid in enumerate(final_ranking):
            reward = -1.0 + 2.0 * (rank / (self.state.num_players - 1))
            reward_dict[pid] = reward
        self.state.set_game_outcome(reward_dict=reward_dict, reason=f"Player {final_ranking[-1]} wins! Final ranking: {final_ranking}")
