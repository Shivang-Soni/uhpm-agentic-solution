from typing import List, Dict


def extract_reflection(state: Dict, action: str) -> List[Dict]:
    """"
    Extract reflection from the data state based on the provided action.
    """
    reflections = state.get("self_reflections", "")

    return [
        r for r in reflections
        if r["action"] == action
    ]
