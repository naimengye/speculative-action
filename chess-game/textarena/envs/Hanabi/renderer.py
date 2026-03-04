def create_board_str(game_state: dict) -> str:
    """
    Create a board representing Hanabi.

    Args:
        game_state (dict): A dictionary containing the game state.

    Returns:
        str: A board representing Hanabi.
    """
    fireworks = game_state.get("fireworks")
    deck_size = len(game_state.get("deck", []))
    discard_pile = game_state.get('discard_pile', [])
    info_tokens = game_state.get('info_tokens', 8)
    fuse_tokens = game_state.get('fuse_tokens', 4)


    lines = []

    # Phase display
    lines.append(f"╭── Hanabi ──────────────────────────────────────────────────╮")
    lines.append(f"│   Deck size: {deck_size:<2}   │   Info tokens: {info_tokens:<1}   │  Fuse tokens: {fuse_tokens:<1}   │")
    lines.append(f"╰────────────────────────────────────────────────────────────╯")

    # Scoreboard
    lines.append("┌─ 🎆 Fireworks ─────────────────────────────────────────────┐")
    for key in fireworks.keys():
        firework = "D" + "=" * int(fireworks[key]) + ">" + " " *(10-int(fireworks[key]))
        lines.append(f"│ {key.value:<10}: {fireworks[key]:<2} │ {firework}                              │")
    lines.append("└────────────────────────────────────────────────────────────┘")

    # Discard pile
    lines.append("┌─ 🚮 Discard pile ──────────────────────────────────────────┐")
    for card in discard_pile:
        lines.append(f"│ {str(card):<59}│")
    lines.append("└────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)