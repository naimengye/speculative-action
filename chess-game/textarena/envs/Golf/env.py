import random
import re
from typing import Optional, Tuple, List, Dict, Any

import textarena as ta

class GolfEnv(ta.Env):
    def __init__(self, num_cards: int = 6, num_columns: int = 3):
        """ Initializes the Golf card game environment """
        super().__init__()
        self.num_cards = num_cards
        self.num_columns = num_columns
        self.num_rows = num_cards // num_columns
        self.deck = self._create_deck()
        
    def _create_deck(self) -> List[Dict[str, Any]]:
        """ Creates a standard 52-card deck """
        suits = ['♠', '♥', '♦', '♣']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        deck = []

        if self.num_cards <= 6:
            num_decks = 1
        elif self.num_cards <= 9:
            num_decks = 2
        elif self.num_cards <= 12:
            num_decks = 3
        
        for _ in range(num_decks):
            for suit in suits:
                for rank in ranks:
                    card = {
                        'rank': rank,
                        'suit': suit,
                        'value': self._get_card_value(rank)
                    }
                    deck.append(card)
        return deck
    
    def _get_card_value(self, rank: str) -> int:
        """ Returns the point value of a card in Golf """
        if rank == 'A':
            return 1
        elif rank in ['J', 'Q']:
            return 10
        elif rank == 'K':
            return 0  # Kinself.state.game_state are worth 0 in Golf
        else:
            return int(rank)
    
    def _card_to_string(self, card: Dict[str, Any]) -> str:
        """ Converts a card to a readable string """
        return f"{card['rank']}{card['suit']}"
    
    def _find_action_token(self, message: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """ Parse action from player message using regex patterns """
        patterns = [
            ("draw", re.compile(r"\[draw\]", re.I)),
            ("take", re.compile(r"\[take\]", re.I)),
            ("swap", re.compile(r"\[swap (\d+) (\d+)\]", re.I)),
            ("discard", re.compile(r"\[discard\]", re.I)),
            ("knock", re.compile(r"\[knock\]", re.I)),
            ("peek", re.compile(r"\[peek (\d+) (\d+)\]", re.I))
        ]
        
        found = [(name, m) for name, rx in patterns if (m := rx.search(message))]
        if len(found) != 1:
            return None, None  # none or ambiguous
            
        action_name, match = found[0]
        params = {}
        
        if action_name in ("swap", "peek"):
            params['row'] = int(match.group(1))
            params['col'] = int(match.group(2))
        
        return action_name, params
    
    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """ Reset the game state """
        if num_players < 2 or num_players > 4:
            raise ValueError("Golf supports 2-4 players")
            
        if num_players == 2:
            self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        else:
            self.state = ta.FFAMultiPlayerState(num_players=num_players, seed=seed)
        
        # Initialize game state BEFORE calling state.reset()
        game_state = self._init_game_state(num_players)
        
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)
        
        # Announce turn options for first player
        self._announce_turn_options(self.state.current_player_id)
    
    def _init_game_state(self, num_players: int) -> Dict[str, Any]:
        """ Initialize and return the complete game state """
        # Shuffle and deal new deck
        deck_copy = self.deck.copy()
        random.shuffle(deck_copy)
        
        # Calculate how many cards to reveal (1/3 of total cards)
        cards_to_reveal = self.num_cards // 3
        
        # Deal cards to players
        players = {}
        for player_id in range(num_players):
            player_cards = []
            
            # Randomly sample positions to reveal for this player
            positions_to_reveal = random.sample(range(self.num_cards), cards_to_reveal)
            
            for i in range(self.num_cards):
                start_revealed = i in positions_to_reveal
                player_cards.append({
                    'card': deck_copy.pop(),
                    'revealed': start_revealed
                })
            
            players[player_id] = {
                'cards': player_cards,
                'score': 0
            }
        
        # Start discard pile
        discard_pile = [deck_copy.pop()]
        
        # Return complete game state
        return {
            'players': players,
            'deck': deck_copy,
            'discard_pile': discard_pile,
            'current_phase': 'playing',  # playing, final_round, finished
            'knocker': None,
            'rounds_after_knock': 0,
            'turn_phase': 'draw'  # draw, action_with_card
        }
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are playing Golf (Card Game) - Player {player_id}.\n"
            f"Goal: Get the lowest total score in this single round. If columns share the same value, they are summed as 0.\n"
            f"Card Values: A=1, 2-10=face value, J/Q=10, K=0\n\n"
            f"Actions (use exact brackets):\n"
            f"- '[draw]' - Draw from deck\n"
            f"- '[take]' - Take from discard pile\n"
            f"- '[swap X Y]' - Swap drawn card with position X (row) Y (column)\n"
            f"- '[discard]' - Discard the drawn card\n"
            # f"- '[knock]' - End the game (only when no card drawn)\n"
            # f"- '[peek X Y]' - Look at card at position X Y (costs a turn)\n"
        )
    
    def _render_player_hand(self, player_id: int) -> str:
        """ Renders the player's hand in a grid format """
        if player_id not in self.state.game_state['players']: return "No cards"
            
        player = self.state.game_state['players'][player_id]
        cards = player['cards']
        
        output = ["  Col:   1    2    3"]
        
        for row in range(self.num_rows):
            row_str = f"Row {row + 1}: "
            for col in range(self.num_columns):
                card_idx = row * self.num_columns + col
                if card_idx < len(cards):
                    card_info = cards[card_idx]
                    
                    if card_info['revealed']:
                        card_str = f"{self._card_to_string(card_info['card']):>4}"
                    else:
                        card_str = "  ? "
                    
                    row_str += f"{card_str} "
                else:
                    row_str += "     "  # Empty space if no card
            output.append(row_str)
        
        return "\n".join(output)
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        
        self.state.add_observation(from_id=player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        
        # Parse action
        action_name, params = self._find_action_token(action)
        
        if action_name is None:
            self.state.set_invalid_move("Use exactly ONE bracketed action: [draw], [take], [swap X Y], [discard], [knock], or [peek X Y]")
            return self.state.step()
        
        # Handle different actions based on current turn phase
        if self.state.game_state['turn_phase'] == 'draw':
            return self._handle_draw_phase(player_id, action_name, params)
        elif self.state.game_state['turn_phase'] == 'action_with_card':
            return self._handle_action_phase(player_id, action_name, params)
    
    def _handle_draw_phase(self, player_id: int, action_name: str, params: Dict) -> Tuple[bool, ta.Info]:
        """ Handle actions when player needs to draw or take a card """
        if action_name == 'draw':
            if not self.state.game_state['deck']:
                # Deck is empty - trigger immediate game end
                self.state.add_observation(message="Deck is empty! Game ends immediately.", observation_type=ta.ObservationType.GAME_MESSAGE)
                self._end_game()
                return self.state.step(rotate_player=False)
            
            drawn_card = self.state.game_state['deck'].pop()
            self.state.game_state['drawn_card'] = drawn_card
            self.state.game_state['turn_phase'] = 'action_with_card'
            
            self.state.add_observation(to_id=player_id, message=f"You drew: {self._card_to_string(drawn_card)}\nNow you must [swap X Y] or [discard] it.", observation_type=ta.ObservationType.GAME_MESSAGE)
            
            # Manually announce the new turn options since we're not rotating players
            self._announce_turn_options(player_id)
            
            return self.state.step(rotate_player=False)
            
        elif action_name == 'take':
            if not self.state.game_state['discard_pile']: 
                self.state.set_invalid_move("The discard pile is empty!")
                return self.state.step()
            
            drawn_card = self.state.game_state['discard_pile'].pop()
            self.state.game_state['drawn_card'] = drawn_card
            self.state.game_state['turn_phase'] = 'action_with_card'
            self.state.game_state['took_from_discard'] = True  # Set this flag
            
            self.state.add_observation(to_id=player_id, message=f"You took: {self._card_to_string(drawn_card)} from discard pile\nYou must [swap X Y] with it (cannot discard).", observation_type=ta.ObservationType.GAME_MESSAGE)
            
            # Manually announce the new turn options since we're not rotating players
            self._announce_turn_options(player_id)
            
            return self.state.step(rotate_player=False)
            
        else:
            self.state.set_invalid_move("You must first [draw], [take], [knock], or [peek X Y]")
            return self.state.step()
    
    def _handle_action_phase(self, player_id: int, action_name: str, params: Dict) -> Tuple[bool, ta.Info]:
        """ Handle actions when player has a drawn card """        
        if action_name == 'swap':
            return self._handle_swap(player_id, params)
        elif action_name == 'discard':
            # Can only discard if card was drawn from deck (not discard pile)
            if self.state.game_state.get('took_from_discard', False):
                self.state.set_invalid_move("You cannot discard a card taken from the discard pile. You must swap it.")
                return self.state.step()
            return self._handle_discard_drawn(player_id)
        else:
            self.state.set_invalid_move("You have a drawn card. You must [swap X Y] or [discard] it.")
            return self.state.step()
    
    def _handle_swap(self, player_id: int, params: Dict) -> Tuple[bool, ta.Info]:
        """ Handle swapping drawn card with a position """        
        if 'drawn_card' not in self.state.game_state:
            self.state.set_invalid_move("You need a drawn card to swap.")
            return self.state.step()
        
        row, col = params['row'], params['col']
        
        if row < 1 or row > self.num_rows or col < 1 or col > self.num_columns:
            self.state.set_invalid_move(f"Position out of bounds. Use row 1-{self.num_rows}, column 1-{self.num_columns}")
            return self.state.step()
        
        # Convert to 0-based indexing
        card_idx = (row - 1) * self.num_columns + (col - 1)
        player = self.state.game_state['players'][player_id]
        
        # Perform the swap
        old_card = player['cards'][card_idx]['card']
        player['cards'][card_idx]['card'] = self.state.game_state['drawn_card']
        player['cards'][card_idx]['revealed'] = True
        
        # Put old card on discard pile
        self.state.game_state['discard_pile'].append(old_card)
        self.state.add_observation(message=f"Player {player_id} swapped {self._card_to_string(self.state.game_state['drawn_card'])} with {self._card_to_string(old_card)} at position ({row},{col})", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
        
        # Clean up and move to next player
        del self.state.game_state['drawn_card']
        self.state.game_state['took_from_discard'] = False
        
        # Check if this player now has all cards revealed (auto-end condition)
        if self._player_has_all_cards_revealed(player_id): return self._trigger_final_round(player_id, "has all cards revealed")
        
        return self._next_turn()
    
    def _handle_discard_drawn(self, player_id: int) -> Tuple[bool, ta.Info]:
        """ Handle discarding the drawn card """
        if 'drawn_card' not in self.state.game_state:
            self.state.set_invalid_move("You need a drawn card to discard.")
            return self.state.step()
        
        discarded_card = self.state.game_state['drawn_card']
        self.state.game_state['discard_pile'].append(discarded_card)
        
        self.state.add_observation(message=f"Player {player_id} discarded {self._card_to_string(discarded_card)}", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
        
        # Clean up and move to next player
        del self.state.game_state['drawn_card']
        self.state.game_state['took_from_discard'] = False
        
        # Check if this player now has all cards revealed (auto-end condition)
        if self._player_has_all_cards_revealed(player_id): return self._trigger_final_round(player_id, "has all cards revealed")
        
        return self._next_turn()
    
    def _handle_knock(self, player_id: int) -> Tuple[bool, ta.Info]:
        """ Handle a player knocking to end the round """
        if 'drawn_card' in self.state.game_state: self.state.set_invalid_move("You cannot knock with a drawn card. Complete your turn first."); return self.state.step()
        if self.state.game_state['knocker'] is not None: self.state.set_invalid_move("Someone has already knocked this round."); return self.state.step()
        self.state.game_state['knocker'] = player_id
        self.state.game_state['current_phase'] = 'final_round'
        self.state.add_observation(message=f"Player {player_id} knocked! Each other player gets one more turn.", observation_type=ta.ObservationType.GAME_MESSAGE)
        return self._next_turn()
    
    def _handle_peek(self, player_id: int, params: Dict) -> Tuple[bool, ta.Info]:
        """ Handle peeking at a face-down card """        
        row, col = params['row'], params['col']
        
        if row < 1 or row > self.num_rows or col < 1 or col > self.num_columns: self.state.set_invalid_move(f"Position out of bounds. Use row 1-{self.num_rows}, column 1-{self.num_columns}"); return self.state.step()
        
        card_idx = (row - 1) * self.num_columns + (col - 1)
        player = self.state.game_state['players'][player_id]
        
        if player['cards'][card_idx]['revealed']: self.state.set_invalid_move("That card is already revealed!"); return self.state.step()
        
        # Show the card to the player
        peeked_card = player['cards'][card_idx]['card']
        self.state.add_observation(to_id=player_id, message=f"You peeked at position ({row},{col}): {self._card_to_string(peeked_card)}", observation_type=ta.ObservationType.GAME_MESSAGE)
        return self._next_turn()
    
    def _player_has_all_cards_revealed(self, player_id: int) -> bool:
        """ Check if a player has all their cards revealed """
        if player_id not in self.state.game_state['players']: return False
        player = self.state.game_state['players'][player_id]
        return all(card_info['revealed'] for card_info in player['cards'])
    
    def _trigger_final_round(self, triggering_player: int, reason: str) -> Tuple[bool, ta.Info]:
        """ Trigger the final round when a condition is met """        
        if self.state.game_state['current_phase'] == 'final_round':
            # Already in final round, just continue
            return self._next_turn()
        
        self.state.game_state['current_phase'] = 'final_round'
        self.state.game_state['triggering_player'] = triggering_player
        self.state.game_state['rounds_after_trigger'] = 0
        self.state.add_observation(message=f"Player {triggering_player} {reason}! Each other player gets one more turn.", observation_type=ta.ObservationType.GAME_MESSAGE)
        return self._next_turn()

    def _next_turn(self) -> Tuple[bool, ta.Info]:
        """ Move to the next player and check for end conditions """        
        # Reset turn phase for next player
        self.state.game_state['turn_phase'] = 'draw'
        
        # Check if final round is complete
        if self.state.game_state['current_phase'] == 'final_round':
            self.state.game_state['rounds_after_knock'] += 1
            
            # If we've completed the final round (everyone except the trigger player gets one turn)
            final_round_complete = False
            if self.state.game_state.get('knocker') is not None:
                # Knock-triggered final round: everyone gets one turn
                final_round_complete = self.state.game_state['rounds_after_knock'] >= self.state.num_players
            elif self.state.game_state.get('triggering_player') is not None:
                # All-cards-revealed triggered: everyone except triggering player gets one turn  
                final_round_complete = self.state.game_state['rounds_after_knock'] >= (self.state.num_players - 1)
            
            if final_round_complete:
                self._end_game()
                return self.state.step(rotate_player=False)
        
        # Move to next player
        next_player = (self.state.current_player_id + 1) % self.state.num_players
        self.state.manually_set_current_player_id(next_player)
        
        result = self.state.step(rotate_player=False)
        self._announce_turn_options(next_player)
        return result
    
    def _announce_turn_options(self, player_id: int):
        """ Announce available actions to the current player """        
        hand_str = self._render_player_hand(player_id)
        discard_top = self._card_to_string(self.state.game_state['discard_pile'][-1]) if self.state.game_state['discard_pile'] else "None"
        
        if self.state.game_state['turn_phase'] == 'draw':
            if self.state.game_state['knocker'] is not None:
                options = "Final turns! Your options: [draw], [take], [peek X Y]"
            else:
                options = "Your options: [draw], [take]"#, [knock], [peek X Y]"
        else:  # action_with_card phase
            if self.state.game_state.get('took_from_discard', False):
                options = "You must: [swap X Y] (cannot discard cards from discard pile)"
            else:
                options = "You must: [swap X Y] or [discard]"
        
        self.state.add_observation(to_id=player_id, message=f"Your hand:\n{hand_str}\nDiscard pile: {discard_top}\n{options}", observation_type=ta.ObservationType.GAME_BOARD)
    
    def _end_game(self):
        """ End the game and determine winner """        
        # Reveal all cards and calculate final scores with column matching
        for player_id, player in self.state.game_state['players'].items():
            # First reveal all cards
            for card_info in player['cards']:
                card_info['revealed'] = True
            
            # Calculate score with column matching rule
            total_score = 0
            
            # Process each column
            for col in range(self.num_columns):
                column_values = []
                
                # Collect all cards in this column
                for row in range(self.num_rows):
                    card_idx = row * self.num_columns + col
                    if card_idx < len(player['cards']):
                        card = player['cards'][card_idx]['card']
                        column_values.append(card['value'])
                
                # Check if all cards in column have the same value
                if len(set(column_values)) == 1 and len(column_values) > 1:
                    # All cards match - column contributes 0 to score
                    column_score = 0
                else:
                    # Cards don't match - sum normally
                    column_score = sum(column_values)
                
                total_score += column_score
            
            player['score'] = total_score
        
        # Find winner (lowest score)
        winner_id = min(self.state.game_state['players'].keys(), key=lambda pid: self.state.game_state['players'][pid]['score'])
        winner_score = self.state.game_state['players'][winner_id]['score']
        
        # Create final summary
        summary = f"Game Over! Player {winner_id} wins with a score of {winner_score} points!\n\nFinal Scores:\n"
        sorted_players = sorted(self.state.game_state['players'].items(), key=lambda x: x[1]['score'])
        
        for i, (player_id, player) in enumerate(sorted_players):
            summary += f"Player {player_id}: {player['score']} points\n"
        
        # Set the winner and end the game
        if self.state.num_players == 2: self.state.set_winner(winner_id, summary)
        else: self.state.set_outcome(reward=1 if self.state.current_player_id == winner_id else 0, reason=summary)
    
    def get_board_str(self) -> str:
        """ Get a string representation of the current game state """
        if not self.state.game_state:
            return "Game not started"
        output = []
        output.append("=== GOLF GAME ===")
        if self.state.game_state['discard_pile']: output.append(f"Discard pile: {self._card_to_string(self.state.game_state['discard_pile'][-1])}")
        if self.state.game_state['knocker'] is not None: output.append(f"Player {self.state.game_state['knocker']} has knocked!")
        output.append("")
        for player_id, player in self.state.game_state['players'].items():
            output.append(f"Player {player_id} (Score: {player['score']}):")
            output.append(self._render_player_hand(player_id))
            output.append("")
        return "\n".join(output)