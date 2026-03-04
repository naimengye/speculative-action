# Minesweeper Environment Documentation

## Overview
**Minesweeper** is a classic single-player puzzle game where the objective is to clear a rectangular board containing hidden mines without detonating any of them. The board is divided into cells, some of which contain mines. Cells adjacent to mines contain numbers indicating the total number of neighboring mines, and these clues help the player avoid mines. The player uses logic and probability to determine which cells are safe to reveal. This environment includes features for revealing cells, placing flags on suspected mine locations, and ensures that the first move is always safe.

## Action Space

- **Format:** Actions are strings representing either revealing a cell or placing/removing a flag, in the format `[action row column]`, where action is either "reveal" or "flag", and row and column indicate the cell's coordinates.
- **Examples:**
  - Reveal the cell at row 3, column 2: `[reveal 3 2]`
  - Place or remove a flag at row 5, column 6: `[flag 5 6]`
- **Notes:** Players can include additional text in their replies, but must provide their action in the correct format with square brackets.

## Observation Space

**Reset Observations**
On reset, the player receives a prompt containing the game instructions and the initial board state. For example:

```plaintext
You are Player 0. You are playing the Minesweeper game.
The objective of the game is to reveal all cells that do not contain mines.
To make a move, you can either reveal a cell or place a flag on a suspected mine location using one of the following commands:
- 'reveal': Reveal the contents of a specific cell.
- 'flag': Place or remove a flag on a specific cell to mark it as a potential mine.
To submit your move, type the command followed by the row and column in square brackets.
For example:
- [reveal 3 2] to reveal the cell in Row 3, Column 2.
- [flag 5 6] to place or remove a flag on the cell in Row 5, Column 6.
On your first move, you will reveal an area around the cell you choose to ensure a safe start.
The current board layout is shown below. Cells that are unrevealed are represented by a dot ('.'), revealed numbers show the count of adjacent mines, and flagged cells are marked with an 'F'.
Use logic and deduction to avoid revealing cells with mines!
Be mindful not to choose revealed or flagged cells.
Here is the current board layout:

   0  1  2  3  4  5  6  7
 0  .  .  .  .  .  .  .  .
 1  .  .  .  .  .  .  .  .
 2  .  .  .  .  .  .  .  .
 3  .  .  .  .  .  .  .  .
 4  .  .  .  .  .  .  .  .
 5  .  .  .  .  .  .  .  .
 6  .  .  .  .  .  .  .  .
 7  .  .  .  .  .  .  .  .
```

**Step Observations**
After each move, the player receives an updated view of the board. For example:

```plaintext
[Player 0] I'll make my first move to reveal the cell at [reveal 4 4].
[GAME] Game Board:
   0  1  2  3  4  5  6  7
 0  .  .  .  .  .  .  .  .
 1  .  .  .  .  .  .  .  .
 2  .  .  .  1  1  1  .  .
 3  .  .  1  1  0  1  1  .
 4  .  .  1  0  0  0  1  .
 5  .  .  1  0  0  0  1  .
 6  .  .  1  1  1  1  1  .
 7  .  .  .  .  .  .  .  .

[Player 0] Now I'll flag a potential mine location at [flag 1 1].
[GAME] Game Board:
   0  1  2  3  4  5  6  7
 0  .  .  .  .  .  .  .  .
 1  .  F  .  .  .  .  .  .
 2  .  .  .  1  1  1  .  .
 3  .  .  1  1  0  1  1  .
 4  .  .  1  0  0  0  1  .
 5  .  .  1  0  0  0  1  .
 6  .  .  1  1  1  1  1  .
 7  .  .  .  .  .  .  .  .
```

## Gameplay

- **Players:** 1 player (single-player game)
- **Initial Setup:** A rectangular grid with hidden mines is created
- **Turns:** The player takes turns revealing cells or placing flags
- **Objective:** Reveal all cells that do not contain mines
- **Maximum Turns:** Configurable, default is 100 turns

## Key Rules

1. **Board Generation:**
   - The game board is a rectangular grid (default: 8×8) containing a specified number of randomly placed mines (default: 10)
   - The first move is guaranteed to be safe, with no mines in the 3×3 area around the first revealed cell

2. **Cell Revealing:**
   - When a cell is revealed, it shows either a number (indicating the count of adjacent mines) or remains empty (if no adjacent mines)
   - If a cell with no adjacent mines is revealed, all neighboring cells are automatically revealed in a cascade
   - Revealing a cell containing a mine results in immediate game over

3. **Flagging:**
   - Players can place a flag on a cell to mark it as a suspected mine location
   - Flagged cells cannot be revealed until the flag is removed
   - Placing a flag on an already flagged cell removes the flag

4. **Valid Moves:**
   - Players can only reveal or flag cells that are within the grid bounds
   - Players cannot reveal cells that are already revealed or flagged
   - Players can flag or unflag any unrevealed cell

5. **Winning Conditions:**
   - **Win:** The player reveals all safe cells (cells without mines) or correctly flags all mines
   - **Loss:** The player reveals a cell containing a mine

6. **Game Termination:**
   - The game concludes when either all safe cells are revealed, all mines are correctly flagged, a mine is revealed, or the maximum turn limit is reached

## Rewards

| Outcome     | Reward for Player |
|-------------|:-----------------:|
| **Win**     | `+1`              |
| **Loss**    | `self._get_percentage_completion()`              |
| **Invalid** | `self._get_percentage_completion()`              |

## Parameters

- `rows` (`int`, default: `8`):
  - **Description:** Sets the number of rows in the grid
  - **Impact:** Larger grid increases difficulty by expanding the playing area

- `cols` (`int`, default: `8`):
  - **Description:** Sets the number of columns in the grid
  - **Impact:** Larger grid increases difficulty by expanding the playing area

- `num_mines` (`int`, default: `10`):
  - **Description:** Sets the number of mines in the grid
  - **Impact:** More mines increase difficulty by making it harder to find safe paths

- `max_turns` (`int`, default: `100`):
  - **Description:** Sets the maximum number of turns allowed
  - **Impact:** Fewer turns make the game more challenging by limiting attempts

## Variants

| Env-id                  | rows | cols | num_mines | max_turns |
|-------------------------|:----:|:----:|:---------:|:---------:|
| `Minesweeper-v0`        | `8`  | `8`  | `10`      | `100`     |
| 'Minesweeper-v0-small   | `5`  | `5`  | `5`       | `100`     |
| `Minesweeper-v0-medium` | `10` | `10` | `20`      | `100`     |
| `Minesweeper-v0-hard`   | `12` | `12` | `30`      | `100`     |

### Contact
If you have questions or face issues with this specific environment, please reach out directly to bobby_cheng@i2r.a-star.edu.sg