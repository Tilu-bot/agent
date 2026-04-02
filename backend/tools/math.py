"""Math calculator tool.

Uses sympy for symbolic math (exact results) with a safe fallback to
Python's ``math`` / ``cmath`` modules for numeric-only expressions when
sympy is not installed.

The expression is evaluated in a restricted namespace — no builtins, no
import, no exec/eval tricks — so arbitrary code execution is not possible.
"""
from __future__ import annotations

import math
import re
from typing import Any

from backend.tools.bus import BaseTool, ToolResult

# Safe numeric namespace for eval when sympy is unavailable.
_SAFE_MATH_NS: dict[str, Any] = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")},
}

# Banned patterns — block any attempt to escape the sandbox.
_BANNED = re.compile(
    r"(__|\bimport\b|\bexec\b|\beval\b|\bopen\b|\bcompile\b|\bglobals\b|\blocals\b)"
)


def _evaluate(expr: str) -> str:
    """Evaluate *expr* using sympy if available, else safe Python eval."""
    if _BANNED.search(expr):
        raise ValueError(f"Forbidden expression: {expr!r}")

    try:
        from sympy import sympify, simplify, latex  # type: ignore[import-untyped]
        result = simplify(sympify(expr))
        numeric = complex(result.evalf())
        if numeric.imag == 0:
            numeric_str = str(float(numeric.real))
        else:
            numeric_str = str(numeric)
        return f"{result}  ≈  {numeric_str}  (LaTeX: {latex(result)})"
    except ImportError:
        pass  # sympy not installed — fall back

    result = eval(expr, _SAFE_MATH_NS, {})  # noqa: S307
    return str(result)


class MathCalculateTool(BaseTool):
    name = "math.calculate"
    description = (
        "Evaluate a mathematical expression and return the exact and approximate result. "
        "Supports arithmetic, trigonometry, logarithms, algebra, and symbolic math (sympy). "
        "Example params: {\"expression\": \"sqrt(2) * pi + 3/7\"}"
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        expression: str = params.get("expression", "").strip()
        if not expression:
            return ToolResult.from_error(self.name, "Missing 'expression' parameter")
        try:
            answer = _evaluate(expression)
            return ToolResult.from_success(
                self.name,
                {"expression": expression, "result": answer},
            )
        except Exception as exc:
            return ToolResult.from_error(self.name, f"Evaluation error: {exc}")
