import re, os, json, random 
import importlib.resources
from typing import Optional, Tuple, Dict, List, Any, Union

import textarena as ta


class TabooEnv(ta.Env):
    """ Environment for Taboo Game. """
    def __init__(self, categories: Union[str, List[str]], max_rounds: Optional[int], max_attempts_per_player: Optional[int], data_path: Optional[str]=None):
        """
        Initialize the Taboo game environment.
        Args:
            categories (Union[str, List[str]]): Either a single category or a list of categories to include in the game.
                                               If a list is provided, one category will be randomly selected.
            max_turns (int): Maximum number of conversation turns.
            data_path (str, optional): Path to the JSON file containing the taboo words.
        """
        # Handle random selection if categories is a list
        if isinstance(categories, list):
            if not categories:
                raise ValueError("Empty list of categories provided.")
            self.categories = [random.choice(categories)]
        else:
            # Convert single string to list for consistent handling
            self.categories = [categories]
            
        self.max_rounds = max_rounds
        self.max_attempts_per_player = max_attempts_per_player
        self.data = self._load_data(data_path)

    @property
    def terminal_render_keys(self):
        return ["word_to_guess", "taboo_words"]

    def _load_data(self, data_path: Optional[str] = None):
        """Load the word list based on the specified categories from the JSON file."""
        try:
            if data_path is not None:
                # Use provided path
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Taboo words data file not found at: {data_path}")
                with open(data_path, "r", encoding="utf-8") as file:
                    full_data = json.load(file)
            else:
                # Use package resource
                with importlib.resources.files('textarena.envs.Taboo').joinpath('words.json').open('r') as file:
                    full_data = json.load(file)
            
            # Validate that all specified categories exist in the data
            missing_categories = [cat for cat in self.categories if cat not in full_data]
            if missing_categories:
                raise ValueError(f"Categories not found in data file: {', '.join(missing_categories)}")
                
            # Subsample to the selected categories
            data = {}
            for category in self.categories:
                data.update(full_data[category])
                
            if not data:
                raise ValueError(f"No words found for selected categories: {', '.join(self.categories)}")
                
            return data
            
        except Exception as e:
            raise FileNotFoundError(f"Failed to load taboo words data: {str(e)}")

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the Taboo game to its initial state """
        assert num_players % 2 == 0, "Number of players must be even for Taboo game."
        assert num_players >= 4, "Taboo game requires at least 4 players."
        self.state = ta.TeamMultiPlayerState(num_players=num_players, max_turns=None, seed=seed)

        # Calculate number of teams
        num_teams = 2
        
        # Sample word pairs for each team (10 words per team)
        all_pairs = list(self.data.items())
        random.shuffle(all_pairs)
        
        team_word_pairs, words_per_team = {}, 20
        
        for team_id in range(num_teams):
            start_idx = team_id * words_per_team
            end_idx = start_idx + words_per_team
            team_word_pairs[team_id] = all_pairs[start_idx:end_idx].copy()

        # Round and turn setup
        self.round, self.turn_in_round, self.max_turns_per_round = 1, 1, self.max_attempts_per_player * (num_players // num_teams)
        
        # Start with team 0
        current_team = 0
        word_to_guess, taboo_words = team_word_pairs[current_team][0]

        game_state = {
            "word_to_guess": word_to_guess,
            "taboo_words": taboo_words,
            "team_word_pairs": team_word_pairs,
            "team_word_indices": {team_id: 0 for team_id in range(num_teams)},  # Track current word index for each team
            "current_team": current_team,
            "round": self.round,
            "max_rounds": self.max_rounds,
            "turn_in_round": self.turn_in_round,
            "max_turns_per_round": self.max_turns_per_round,
            "score": {team_id: 0 for team_id in range(num_teams)},  # Initialize scores for all teams
            "state_observed": False,  # NEW: Track if current state has been observed
        }

        # Assign roles: player 0 and player N//2 are clue givers of their teams
        role_mapping = {}
        self.team_size = num_players // num_teams
        for i in range(num_players):
            if i % self.team_size == 0:
                role_mapping[i] = "Clue Giver"
            else:
                role_mapping[i] = "Guesser"

        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt, role_mapping=role_mapping)
        self._observe_current_state()

    def _get_team_members(self, player_id: int) -> List[int]:
        """ Get all team members for a given player """
        team_id = player_id // self.team_size
        
        # Get all players in the same team
        team_members = []
        for pid in range(self.state.num_players):
            if pid // self.team_size == team_id:
                team_members.append(pid)
        
        return team_members
    
    def _get_guessers(self) -> List[int]:
        """ Get all guessers across all teams """
        return [pid for pid in range(self.state.num_players) 
                if self.state.role_mapping[pid] == "Guesser"]
    
    def _get_current_team_clue_givers(self, player_id: int) -> List[int]:
        """ Get clue givers from the current player's team """
        current_team_members = self._get_team_members(player_id)
        return [pid for pid in current_team_members if self.state.role_mapping[pid] == "Clue Giver"]

    def _get_team_id(self, player_id: int) -> int:
        """ Get team ID for a given player """
        return player_id // self.team_size

    def _get_next_team(self, current_team: int) -> int:
        """ Get the next team in rotation """
        return (current_team + 1) % (self.state.num_players // self.team_size)

    def _rotate_player_by_logic(
        self,
        rotate_within_team: bool,
        start_from_first_in_team: bool,
        move_to_next_team: bool,
        force_rotation: bool = False,
    ) -> None:
        """
        Handles player rotation based on the logic flags:
        - rotate_within_team: Rotate within the current team to the next player.
        - start_from_first_in_team: Keep the team, but always start from the first player in that team.
        - move_to_next_team: Switch to the next team, starting from their first player.
        - force_rotation: If True, forces the rotation even if an invalid move was made.
        """
        current_team = self._get_team_id(self.state.current_player_id)

        if move_to_next_team:
            next_team = self._get_next_team(current_team)
            next_player_id = next_team * self.team_size
        elif start_from_first_in_team:
            next_player_id = current_team * self.team_size
        elif rotate_within_team:
            within_team_index = self.state.current_player_id % self.team_size
            next_within_team_index = (within_team_index + 1) % self.team_size
            next_player_id = current_team * self.team_size + next_within_team_index
        else:
            # Default: stay on current player
            next_player_id = self.state.current_player_id

        # Use force rotation for invalid moves
        if force_rotation:
            self.state.current_player_id = next_player_id
            self.state.error_count = 0
        else:
            self.state.manually_set_current_player_id(next_player_id)
        
        return next_player_id

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the initial prompt for a player based on their role """
        team_id = self._get_team_id(player_id)
        
        if self.state.role_mapping[player_id] == "Clue Giver":
            # Get the team's first word from their allocated word list
            team_words = game_state["team_word_pairs"][team_id]
            word_to_guess, taboo_words = team_words[0]  # Use first word for initial prompt
            
            prompt = (
                f"You are Player {player_id}, the Clue Giver for Team {team_id} in the Taboo game.\n"
                "Your goal is to provide clues to help the Guesser guess the word without using the taboo words or the word to guess.\n"
                f"You have {self.max_attempts_per_player} each round to assist the Guesser to guess as many words as possible.\n"
                "On your turn, simply type your clue.\n"
            )

        elif self.state.role_mapping[player_id] == "Guesser":
            prompt = (
                f"You are Player {player_id}, the Guesser for Team {team_id} in the Taboo game.\n"
                "Your goal is to guess the secret word based on the clues provided by the Clue Giver.\n"
                f"You have {self.max_attempts_per_player} each round to to guess as many words as possible.\n"
                "On your turn, simply type your guess in squared brackets. For example: ['elephant'].\n"
            )

        else:
            # unexpected
            raise ValueError(f"Unexpected role mapping: {self.state.role_mapping[player_id]}. Expected 'Clue Giver' or 'Guesser'.")

        return prompt 
    
    def _observe_current_state(self) -> None:
        """ Observe the current state of the game for all players
         and notify the current team only about the current word and taboo words """
        
        # NEW: Check if state has already been observed
        if self.state.game_state.get("state_observed", False):
            return
            
        word_to_guess = self.state.game_state["word_to_guess"]
        taboo_words = self.state.game_state["taboo_words"]

        # Notify all players about the current word and taboo words
        if self.state.role_mapping[self.state.current_player_id] == "Clue Giver":
            # Clue Givers should know the current word and taboo words
            self.state.add_observation(to_id=self.state.current_player_id, message=f"The current word to guess is '{word_to_guess}'. Taboo words: {', '.join(taboo_words)}.", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

        self.state.game_state["state_observed"] = True

    def _new_word_selected(self):
        """ Call this method when a new word is selected to reset the observation flag """
        self.state.game_state["state_observed"] = False

    def _is_end_of_round(self) -> bool:
        """ Check if current round has ended """
        return self.state.game_state["turn_in_round"] > self.state.game_state["max_turns_per_round"]

    def _advance_to_next_word(self, team_id: int) -> None:
        """ Advance to next word for the given team """
        self.state.game_state["team_word_indices"][team_id] += 1

    def _switch_to_next_team(self) -> int:
        """ Switch to next team and return the new team ID """
        current_team = self.state.game_state["current_team"]
        next_team = self._get_next_team(current_team)
        self.state.game_state["current_team"] = next_team
        self.state.game_state["turn_in_round"] = 1
        self.state.game_state["round"] += 1
        return next_team

    def _update_word_for_team(self, team_id: int) -> None:
        """ Update the current word and taboo words for the given team """
        word_to_guess, taboo_words = self._get_current_word_for_team(team_id)
        self.state.game_state["word_to_guess"] = word_to_guess
        self.state.game_state["taboo_words"] = taboo_words
        self._new_word_selected()

    def _notify_clue_givers_new_word(self, team_message: str) -> None:
        """ Notify clue givers about the new word """
        clue_givers = self._get_current_team_clue_givers(self.state.current_player_id)
        word_to_guess = self.state.game_state["word_to_guess"]
        taboo_words = self.state.game_state["taboo_words"]
        
        message = f"{team_message} The word to guess is '{word_to_guess}'. Taboo words: {', '.join(taboo_words)}."
        
        for clue_giver_id in clue_givers:
            self.state.add_observation(
                to_id=clue_giver_id, 
                message=message, 
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
            )

    def _notify_team_switch(self, next_team: int) -> None:
        """ Notify all players about team switch and provide appropriate information """
        # Notify ALL players about team switch
        self.state.add_observation(
            message=f"Team {1 - next_team} has finished their turn. It is now Team {next_team}'s turn to play.", 
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )

        # Notify clue givers of the new word
        self._notify_clue_givers_new_word(f"It is Team {next_team}'s turn now.")

    def _handle_correct_guess(self) -> None:
        """ Handle logic when guesser makes a correct guess """
        self.increment_score()
        current_team = self.state.game_state["current_team"]
        self._advance_to_next_word(current_team)
        
        # Check if team has more words, otherwise get next word
        if self.state.game_state["team_word_indices"][current_team] < len(self.state.game_state["team_word_pairs"][current_team]):
            next_word_pair = self.state.game_state["team_word_pairs"][current_team][self.state.game_state["team_word_indices"][current_team]]
            self.state.game_state["word_to_guess"] = next_word_pair[0]
            self.state.game_state["taboo_words"] = next_word_pair[1]
            self._new_word_selected()

        if not self._is_end_of_round():
            # Continue with same team
            self._rotate_player_by_logic(rotate_within_team=True, start_from_first_in_team=True, move_to_next_team=False)
            self._notify_clue_givers_new_word("The next word to guess is")
        else:
            # Switch to next team
            next_team = self._switch_to_next_team()
            self._rotate_player_by_logic(rotate_within_team=False, start_from_first_in_team=True, move_to_next_team=True)
            self._update_word_for_team(next_team)
            self._notify_team_switch(next_team)

    def _handle_incorrect_guess(self) -> None:
        """ Handle logic when guesser makes an incorrect guess """
        if not self._is_end_of_round():
            self._rotate_player_by_logic(rotate_within_team=True, start_from_first_in_team=False, move_to_next_team=False)
        else:
            # End of the round, switch to the next team
            next_team = self._switch_to_next_team()
            self._rotate_player_by_logic(rotate_within_team=False, start_from_first_in_team=True, move_to_next_team=True)
            self._update_word_for_team(next_team)
            self._notify_team_switch(next_team)

    def _handle_clue_giver_invalid_move(self) -> None:
        """ Handle invalid move by clue giver (using taboo words) """
        current_team = self.state.game_state["current_team"]
        self._advance_to_next_word(current_team)  # Penalty: advance word
        
        # Force rotation to next team and get new word for that team
        next_team = self._switch_to_next_team()
        self._rotate_player_by_logic(
            rotate_within_team=False, 
            start_from_first_in_team=False, 
            move_to_next_team=True,
            force_rotation=True
        )
        self._update_word_for_team(next_team)
        self._notify_team_switch(next_team)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's action """
        player_id = self.state.current_player_id
        
        # Notify teammates about the action
        for teammate_id in self._get_team_members(player_id):
            self.state.add_observation(from_id=player_id, to_id=teammate_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        
        self.state.game_state["turn_in_round"] += 1

        # Clue Giver's turn
        if self.state.role_mapping[player_id] == "Clue Giver":
            # Check for taboo words or the word to guess in the clue 
            forbidden_words = self.state.game_state["taboo_words"] + [self.state.game_state["word_to_guess"]]
            pattern = re.compile(r"\b(" + "|".join(map(re.escape, forbidden_words)) + r")\b", re.IGNORECASE)
            
            if pattern.search(action):
                # Clue Giver used a forbidden word.
                reason = f"The Clue Giver (Player {player_id}) mentioned a taboo word, or the target word."
                terminated_by_invalid = self.state.set_invalid_move(reason=reason)
                if terminated_by_invalid:
                    self._handle_clue_giver_invalid_move()
            else:
                self._rotate_player_by_logic(rotate_within_team=True, start_from_first_in_team=False, move_to_next_team=False)

        # Guesser's turn
        elif self.state.role_mapping[player_id] == "Guesser":
            # Guesser must provide a guess within squared brackets
            guess_pattern = re.compile(r"\[(.*?)\]")
            match = guess_pattern.search(action)
            
            if not match:
                # Invalid guess format
                reason = "Invalid guess format. Please provide your guess within squared brackets, e.g., '[apple]'."
                terminated_by_invalid = self.state.set_invalid_move(reason=reason)
                if not terminated_by_invalid:
                    # Don't rotate player on invalid move, give them another chance
                    pass
                return self.state.step()
                
            guess = match.group(1).strip().lower()
            correct_word = self.state.game_state["word_to_guess"].lower()

            if guess == correct_word:
                self._handle_correct_guess()
            else:
                self._handle_incorrect_guess()

        else:
            # unexpected
            raise ValueError(f"Unexpected role mapping: {self.state.role_mapping[player_id]}. Expected 'Clue Giver' or 'Guesser'.")
        
        # Check for game end
        if self.state.game_state["round"] > self.state.game_state["max_rounds"]:
            # End of the game - compare scores across all teams
            scores = self.state.game_state["score"]
            max_score = max(scores.values())
            winning_teams = [team_id for team_id, score in scores.items() if score == max_score]
            
            if len(winning_teams) == 1:
                # Single winning team
                winning_team = winning_teams[0]
                team_size = self.state.num_players // 2
                team_members = [i for i in range(self.state.num_players) if i // team_size == winning_team]
                self.state.set_winners(player_ids=team_members, reason=f"Team {winning_team} won with score {max_score}.")
            else:
                # Multiple teams tied for highest score
                self.state.set_draw(reason=f"The game ended in a draw with teams {winning_teams} all scoring {max_score}.")

        return self.state.step()

    def increment_score(self):
        """ Increment the score for the current team """
        current_team = self.state.game_state["current_team"]
        self.state.game_state["score"][current_team] += 1
        self.state.add_observation(message=f"Team {current_team} scored a point! Current score: {self.state.game_state['score']}", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

    def _get_current_word_for_team(self, team_id: int) -> Tuple[str, List[str]]:
        """ Get the current word for a specific team """
        current_word_idx = self.state.game_state["team_word_indices"][team_id]
        team_words = self.state.game_state["team_word_pairs"][team_id]
        
        if current_word_idx >= len(team_words):
            raise ValueError(f"Team {team_id} has no more words available.")
            
        return team_words[current_word_idx]