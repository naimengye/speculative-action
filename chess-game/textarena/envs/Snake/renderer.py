from typing import Dict, List, Tuple, Any


def create_board_str(width: int, height: int, snakes: Dict[int, Any], apples: List[Tuple[int, int]]) -> str:
    board = [['.' for _ in range(width)] for _ in range(height)]
    for (ax, ay) in apples: board[ay][ax] = 'A' # Place apples
    for pid, snake in snakes.items(): # Place snake body and head
        if not snake.alive: continue
        for idx, (x, y) in enumerate(snake.positions):
            if idx == 0: board[y][x] = format(pid, 'X')  # Use hex digit for player ID
            else: board[y][x] = '#'
    lines = []
    content_width = width * 2 - 1
    border_line = "+" + "-" * (content_width + 2) + "+"
    lines.append(border_line)
    for row_idx in range(height - 1, -1, -1): # Draw board from top (height-1) to bottom (0)
        row_str = " ".join(board[row_idx])
        lines.append(f"| {row_str} |")
    lines.append(border_line)
    return "\n".join(lines)
