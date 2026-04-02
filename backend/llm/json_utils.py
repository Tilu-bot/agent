"""Robust JSON extraction from LLM free-text responses.

LLMs sometimes wrap JSON in markdown code fences, add prose before or after,
or include multiple JSON objects.  This module provides a single
``extract_json()`` helper that tries several strategies in order so callers
don't have to repeat the same fragile ``raw.find('{')`` pattern.

Strategy order
--------------
1. Direct ``json.loads`` — works when the model outputs pure JSON (e.g. with
   Ollama's ``format: "json"`` mode enabled).
2. Markdown code-fence extraction — strips ````json … ``` `` wrappers.
3. Outermost-brace extraction — finds the first ``{`` / ``[`` and the last
   matching ``}`` / ``]`` and attempts to parse that substring.

All strategies fall back gracefully; the function never raises.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Matches ```json ... ``` or ``` ... ``` (non-greedy, DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\[{].*?[}\]])\s*```", re.DOTALL)


def extract_json(text: str, default: Any = None) -> Any:
    """Return the first parseable JSON value from *text*.

    Parameters
    ----------
    text:
        Raw LLM output that should contain a JSON object or array.
    default:
        Value returned when no JSON can be extracted.  Defaults to ``None``.

    Returns
    -------
    Parsed JSON value (dict, list, etc.) or *default*.
    """
    if not text:
        return default

    stripped = text.strip()

    # 1. Direct parse — works when Ollama's format="json" is used.
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Markdown code fence.
    m = _FENCE_RE.search(stripped)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Find outermost { ... } or [ ... ].
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

    return default
