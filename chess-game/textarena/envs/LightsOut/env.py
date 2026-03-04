import random
import re
from typing import Dict, Any, Optional, Tuple, List

import textarena as ta


class LightsOutEnv(ta.Env):

    def __init__(self, size: int = 5, max_turns: int = 50):
        self.size = size
        self.max_turns = max_turns
        # Action format: [row col] where row and col are 0-indexed
        self.action_space = re.compile(r'\[(\d+)\s+(\d+)\]')

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.SinglePlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)
        grid = [[False for _ in range(self.size)] for _ in range(self.size)]
        num_scramble_moves = random.randint(5, 15)
        for _ in range(num_scramble_moves):
            row = random.randint(0, self.size - 1)
            col = random.randint(0, self.size - 1)
            self._toggle_lights(grid, row, col)
        self._initial_on = sum(light for row in grid for light in row)
        self.state.reset(
            game_state=dict(grid=grid, moves_made=0, solved=False), 
            player_prompt_function=self._prompt
        )
        self._show_current_state()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"Welcome to Lights Out! You have a {self.size}x{self.size} grid of lights.\n"
            "Your goal is to turn ALL lights OFF (represented by '.')\n"
            "When you press a light, it toggles itself AND its adjacent neighbors (up/down/left/right).\n"
            f"Type [row col] to press a light (0-indexed, so valid range is 0-{self.size-1}).\n"
            f"You have up to {self.max_turns} moves to solve the puzzle.\n"
            "Legend: 'O' = light ON, '.' = light OFF"
        )
    
    def _toggle_lights(self, grid: List[List[bool]], row: int, col: int):
        """Toggle the light at (row, col) and its orthogonal neighbors"""
        directions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # self, up, down, left, right
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                grid[new_row][new_col] = not grid[new_row][new_col]
    
    def _grid_to_string(self, grid: List[List[bool]]) -> str:
        """Convert grid to a readable string representation"""
        result = []
        # Add column headers
        result.append("   " + " ".join(str(i) for i in range(self.size)))
        
        for i, row in enumerate(grid):
            row_str = f"{i}: " + " ".join("O" if light else "." for light in row)
            result.append(row_str)
        return "\n".join(result)
    
    def _is_solved(self, grid: List[List[bool]]) -> bool:
        """Check if all lights are off (solved state)"""
        return all(not light for row in grid for light in row)
    
    def _get_percentage_completion(self) -> float:
        """Calculate completion percentage based on lights turned off"""
        grid = self.state.game_state['grid']
        on_now = sum(light for row in grid for light in row)
        raw_progress = (self._initial_on - on_now) / self._initial_on
        return float(min(1.0, raw_progress))
    
    def _show_current_state(self):
        grid_str = self._grid_to_string(self.state.game_state['grid'])
        moves_made = self.state.game_state['moves_made']
        moves_left = self.max_turns - moves_made
        completion = self._get_percentage_completion()
        
        message = f"Current grid state (Move {moves_made}, {moves_left} moves remaining, {completion:.1f}% complete):\n{grid_str}"
        self.state.add_observation(message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        # Validate action format
        m = self.action_space.fullmatch(action.strip())
        if m is None:
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"Action must be in format [row col] where row and col are integers from 0 to {self.size-1}.")
            return self.state.step()

        row, col = int(m.group(1)), int(m.group(2))
        
        # Validate coordinates
        if not (0 <= row < self.size and 0 <= col < self.size):
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"Coordinates must be between 0 and {self.size-1}. You entered [{row},{col}].")
            return self.state.step()

        # Apply the move
        self._toggle_lights(self.state.game_state['grid'], row, col)
        self.state.game_state['moves_made'] += 1
        
        # Check if puzzle is solved
        if self._is_solved(self.state.game_state['grid']):
            self._resolve_win()
        elif self.state.game_state['moves_made'] >= self.max_turns:
            self._resolve_loss()
        else:
            self._show_current_state()
        
        return self.state.step()

    def _resolve_win(self):
        moves_used = self.state.game_state['moves_made']
        message = f"Congratulations! You solved the puzzle in {moves_used} moves!"
        self.state.set_outcome(reward=1.0, reason=message)

    def _resolve_loss(self):
        completion = self._get_percentage_completion()
        message = f"Game over! You used all {self.max_turns} moves without solving the puzzle. Final completion: {completion:.1f}%"
        final_grid = self._grid_to_string(self.state.game_state['grid'])
        self.state.set_outcome(reward=self._get_percentage_completion(), reason=f"{message}\nFinal state:\n{final_grid}")

