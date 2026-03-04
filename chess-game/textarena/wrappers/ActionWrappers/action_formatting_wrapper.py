from textarena.core import ActionWrapper, Env

__all__ = ["ActionFormattingWrapper", "ActionLastLineFormattingWrapper"]


class ActionFormattingWrapper(ActionWrapper):
    """
    A wrapper that formats actions by adding brackets if they're missing.
    
    This wrapper ensures that all actions follow a consistent format by wrapping
    them in square brackets if they don't already contain brackets. This is useful
    for environments that require actions to be enclosed in brackets but where
    agents might not always follow this convention.
    
    Example:
        - Input: "move north"
        - Output: "[move north]"
        
        - Input: "[trade wheat]"
        - Output: "[trade wheat]" (unchanged)
    """

    def __init__(self, env: Env):
        """
        Initialize the ActionFormattingWrapper.
        
        Args:
            env (Env): The environment to wrap.
        """
        super().__init__(env)

    def action(self, action: str) -> str:
        """
        Format the action by adding brackets if they're missing.
        
        This method checks if the action already contains square brackets.
        If not, it wraps the entire action string in square brackets.
        
        Args:
            action (str): The action to format.
            
        Returns:
            str: The formatted action, with brackets added if necessary.
        """
        if "[" not in action and "]" not in action:
            return f"[{action}]"
        else:
            return action


class ActionLastLineFormattingWrapper(ActionWrapper):
    """
    A wrapper that formats actions by only reading the last line, and adding brackets if they're missing. Also add bracket.

    """

    def __init__(self, env: Env):
        """
        Initialize the ActionFormattingWrapper.
        
        Args:
            env (Env): The environment to wrap.
        """
        super().__init__(env)

    def action(self, action: str) -> str:
        """
        Read multi-line action, trim </?answer>, and check for action at head of first or last line.
        """
        lines=action.strip().splitlines()
        if len(lines)==0: return ""
        if len(lines)>1 and lines[-1].strip()=='</answer>': lines=lines[:-1]
        if len(lines)>1 and lines[0].strip()=='<answer>': lines=lines[1:]
        if lines[-1].strip().startswith('[') and lines[-1].strip().endswith(']'):
            return lines[-1].strip()
        if lines[0].strip().startswith('[') and lines[0].strip().endswith(']'):
            return lines[0].strip()
        action=lines[-1].strip().split()[0]
        if "[" not in action and "]" not in action:
            return f"[{action}]"
        else:
            return action