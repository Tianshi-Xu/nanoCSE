"""JSON 序列化辅助工具"""

import math
from datetime import datetime
from typing import Any


def json_safe(obj: Any) -> Any:
    """Recursively convert objects to JSON-safe values.

    Handles:
    - Non-finite floats (incl. numpy scalars) -> "NaN"/"Infinity"/"-Infinity" strings
    - numpy scalars via `.item()` to native Python types
    - Path-like objects -> str
    - datetime -> ISO string
    - dict keys coerced to str; values sanitized recursively
    - lists/tuples/sets -> lists with sanitized items
    - Falls back to `str(obj)` for unknown objects
    """
    try:
        # Basic primitives
        if obj is None or isinstance(obj, (str, int, float, bool)):
            if isinstance(obj, float):
                if math.isfinite(obj):
                    return obj
                if math.isnan(obj):
                    return "NaN"
                return "-Infinity" if obj < 0 else "Infinity"
            return obj

        # numpy/scalar-like: try to convert using item()
        item_fn = getattr(obj, "item", None)
        if callable(item_fn):
            try:
                return json_safe(item_fn())
            except Exception:
                pass

        # Path-like
        if hasattr(obj, "__fspath__"):
            try:
                return str(obj)
            except Exception:
                pass

        # datetime
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Containers
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [json_safe(v) for v in obj]

        # Fallback for other objects with __dict__
        if hasattr(obj, "__dict__"):
            try:
                return {k: json_safe(v) for k, v in obj.__dict__.items()}
            except Exception:
                pass

        # Final fallback: string representation
        return str(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"
