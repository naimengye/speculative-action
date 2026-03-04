# Memory Game Environment Documentation

## Overview
**Memory Game** (also known as Concentration or Matching Pairs) is a card game where players take turns flipping pairs of cards to find matching pairs. The board consists of a grid of face-down cards, with each card having a matching partner somewhere else on the board. When a player successfully matches a pair, they score a point and the cards remain face-up. If the cards don't match, they are flipped back face-down. The player who matches the most pairs by the end of the game wins. This implementation supports variable grid sizes to adjust difficulty.

## Action Space

- **Format:** Actions are strings representing the positions of two cards to flip, in the format `[R1 C1 R2 C2]`, where R1 and C1 are the row and column of the first card, and R2 and C2 are the row and column of the second card.
- **Examples:**
  - Flip card at (0,1) and card at (1,0): `[0 1 1 0]`
  - Flip card at (2,3) and card at (3,1): `[2 3 3 1]`
- **Notes:** Players can include additional text in their replies, but must provide their card selections in the correct format with square brackets.

## Observation Space

**Reset Observations**
On reset, each player receives a prompt containing their game instructions and the initial board state. For example:

```plaintext
You are Player 0. You are playing the Memory Game.
Your goal is to match more pairs of cards on the board, than your opponent.
On your turn, select two cards to flip by entering the row and column numbers of the first and second card respectively like [0 1 1 0], where the first card is in row 0 and column 1, and the second card is in row 1 and column 0.
If the two cards match, you get a point and the cards remain face up. If they do not match, the cards are flipped back face down, e.g. '.'.
The game ends when all pairs have been matched.
Here is the initial board with all cards faced down:
  0 1 2 3
0 . . . .
1 . . . .
2 . . . .
3 . . . .
```

**Step Observations**
After each move, players receive updates about the cards flipped and whether they matched. For example:

```plaintext
[Player 0] I'll try to find a matching pair. Let me flip the cards at positions [0 0 1 1].
[GAME] The cards do not match. Cards at positions [0 0] and [1 1] are B and C respectively.
[Player 1] I'm going to try to find a pair. I'll flip [2 3 0 2].
[GAME] Cards at positions [2 3] and [0 2] match!
Updated board:
  0 1 2 3
0 . . A .
1 . . . .
2 . . . A
3 . . . .
```

## Gameplay

- **Players:** 2 players
- **Initial Setup:** All cards are face-down in a grid, with each card having exactly one matching partner
- **Turns:** Players take turns flipping two cards to try to find matching pairs
- **Scoring:** A player scores 1 point for each pair they successfully match
- **Objective:** Match more pairs than the opponent by the end of the game

## Key Rules

1. **Board Setup:**
   - The game board is a grid of face-down cards (default: 4×4)
   - Each card has exactly one matching partner elsewhere on the board
   - Initially, all cards are face-down, indicated by "." in the display

2. **Card Flipping:**
   - On their turn, a player selects two face-down cards to flip
   - If the two cards have the same symbol (match), they remain face-up and the player scores a point
   - If the two cards have different symbols (no match), they are flipped back face-down
   - A player cannot select a card that is already face-up (matched)

3. **Valid Moves:**
   - Players must select two different cards
   - Both cards must be within the bounds of the grid
   - Both cards must be face-down (not previously matched)

4. **Winning Conditions:**
   - **Win:** The player with the most matched pairs when all pairs have been found
   - **Draw:** If both players match the same number of pairs
   - **Loss:** The player with fewer matched pairs when all pairs have been found

5. **Game Termination:**
   - The game concludes when all pairs have been matched

## Rewards

| Outcome     | Reward for Winner | Reward for Loser |
|-------------|:-----------------:|:----------------:|
| **Win**     | `+1`              | `-1`             |
| **Draw**    | `0`               | `0`              |
| **Invalid** | `-1`              | `0`              |

## Parameters

- `grid_size` (`int`, default: `4`):
  - **Description:** Sets the size of the grid (grid_size × grid_size)
  - **Impact:** Larger grids increase difficulty by requiring more memory and creating more potential matches

## Variants

| Env-id                     | grid_size |
|----------------------------|:---------:|
| `MemoryGame-v0`            | `4`       |
| `MemoryGame-v0-medium`     | `6`       |
| `MemoryGame-v0-hard`       | `8`       |


### Contact
If you have questions or face issues with this specific environment, please reach out directly to bobby_cheng@i2r.a-star.edu.sg