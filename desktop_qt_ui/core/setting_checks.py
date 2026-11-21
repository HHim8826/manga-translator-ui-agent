"""Setting values checked against ``AppSettings``, the same way for a number typed on the settings
page and for an imported config file: which key is wrong, why, and a hint in the user's language."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from core.config_models import AppSettings

_LOW_KINDS = {"greater_than_equal": "ge", "greater_than": "gt"}
_HIGH_KINDS = {"less_than_equal": "le", "less_than": "lt"}


@dataclass
class SettingProblem:
    path: str
    """The key, for example ``agent.max_context_tokens``."""
    kind: str
    """pydantic's error type, or ``compact_not_below_limit``."""
    ctx: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.path.rsplit(".", 1)[-1]


def problems(data: dict) -> List[SettingProblem]:
    """What ``AppSettings`` rejects in ``data`` (a whole config as a dict)."""
    try:
        AppSettings.model_validate(data)
    except ValidationError as error:
        return [SettingProblem(".".join(str(part) for part in e["loc"]), e["type"], dict(e.get("ctx") or {}))
                for e in error.errors()]
    return []


def _field(path: str):
    model, info = AppSettings, None
    for part in path.split("."):
        info = (getattr(model, "model_fields", None) or {}).get(part)
        if info is None:
            return None
        model = info.annotation
    return info


def limits(path: str) -> Tuple[Optional[Tuple[str, Any]], Optional[Tuple[str, Any]]]:
    """The field's lower and upper limit, each ``(ge|gt|le|lt, value)`` or None."""
    low = high = None
    for meta in getattr(_field(path), "metadata", None) or ():
        for name in ("ge", "gt"):
            if getattr(meta, name, None) is not None:
                low = (name, getattr(meta, name))
        for name in ("le", "lt"):
            if getattr(meta, name, None) is not None:
                high = (name, getattr(meta, name))
    return low, high


def default(path: str) -> Any:
    node: Any = AppSettings().model_dump()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check_change(config: AppSettings, path: str, value: Any) -> Tuple[Optional[SettingProblem], Any]:
    """Setting ``path`` to ``value`` in ``config``: the problem it causes, if any, and the value to
    store. An empty field (None) where None is not allowed means the default. Only problems of this
    key or ones the change causes count, never an older one elsewhere."""
    data = config.model_dump()
    before = {(p.path, p.kind) for p in problems(data)}
    *parents, last = path.split(".")
    node = data
    for part in parents:
        node = node[part]

    def caused(candidate):
        node[last] = candidate
        found = [p for p in problems(data) if p.path == path or (p.path, p.kind) not in before]
        found.sort(key=lambda p: p.path != path)
        return found[0] if found else None

    problem = caused(value)
    if problem is not None and value is None:
        fallback = default(path)
        if fallback is not None:
            value, problem = fallback, caused(fallback)
    if problem is None:
        try:  # the value as the model stores it ("0.5" typed into a float field is 0.5)
            stored: Any = AppSettings.model_validate(data)
            for part in path.split("."):
                stored = getattr(stored, part)
            value = stored
        except (ValidationError, AttributeError):
            pass
    return problem, value


def _number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def describe(t: Callable[..., str], problem: SettingProblem, label: str) -> str:
    """The hint for one problem; ``t`` translates (``key, **kwargs``), ``label`` names the setting."""
    if problem.kind == "compact_not_below_limit":
        return t("Setting Hint Below Limit", name=label, limit=_number(problem.ctx.get("limit")))
    if problem.kind in _LOW_KINDS or problem.kind in _HIGH_KINDS:
        low, high = limits(problem.path)
        if problem.kind in _LOW_KINDS and low is None:
            low = (_LOW_KINDS[problem.kind], problem.ctx.get(_LOW_KINDS[problem.kind]))
        if problem.kind in _HIGH_KINDS and high is None:
            high = (_HIGH_KINDS[problem.kind], problem.ctx.get(_HIGH_KINDS[problem.kind]))
        if low and high and low[0] == "ge" and high[0] == "le":
            return t("Setting Hint Range", name=label, low=_number(low[1]), high=_number(high[1]))
        if problem.kind in _LOW_KINDS:
            hint = "Setting Hint At Least" if low[0] == "ge" else "Setting Hint Above"
            return t(hint, name=label, low=_number(low[1]))
        hint = "Setting Hint At Most" if high[0] == "le" else "Setting Hint Below"
        return t(hint, name=label, high=_number(high[1]))
    if problem.kind.startswith("int_"):
        return t("Setting Hint Integer", name=label)
    if problem.kind.startswith(("float_", "decimal_")):
        return t("Setting Hint Number", name=label)
    return t("Setting Hint Invalid", name=label)
