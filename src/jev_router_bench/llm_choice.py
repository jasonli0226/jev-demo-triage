"""Parse the LLM router's JSON answer. Anything unexpected raises; nothing is guessed."""

import json
import re
from typing import Any

from jev_router_bench.pool import TIERS, Tier


class LlmRouteError(RuntimeError):
    """The router model's answer could not be turned into a tier."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise LlmRouteError("duplicate keys in a JSON object of the answer")
    return dict(pairs)


def _json_objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder(object_pairs_hook=_no_duplicate_keys)
    found: list[dict] = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    return found


def _probabilities(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise LlmRouteError(f"probabilities must be a non-empty object: {value!r}")
    out: dict[str, float] = {}
    for tier, p in value.items():
        if tier not in TIERS:
            raise LlmRouteError(f"unknown tier in probabilities: {tier!r}")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            raise LlmRouteError(f"probability for {tier!r} is not in [0, 1]: {p!r}")
        out[tier] = float(p)
    return out


def _parse_object(obj: dict) -> tuple[Tier, dict[str, float] | None]:
    choice = obj["choice"]
    if not isinstance(choice, str) or choice not in TIERS:
        raise LlmRouteError(f"unknown choice: {choice!r}")
    if obj.get("probabilities") is None:
        return choice, None
    probabilities = _probabilities(obj["probabilities"])
    if probabilities.get(choice, -1.0) < max(probabilities.values()):
        raise LlmRouteError(f"choice {choice!r} is not the most probable tier: {probabilities}")
    return choice, probabilities


def parse_choice(text: str) -> tuple[Tier, dict[str, float] | None]:
    """Tier and optional probabilities from JSON objects with a `choice` key.

    Every such object must be valid and all must agree; prose around them is ignored.
    """
    parsed = [_parse_object(o) for o in _json_objects(text) if "choice" in o]
    if not parsed:
        raise LlmRouteError(f"no JSON object with a 'choice' key in answer: {text[:80]!r}")
    first = parsed[0]
    if any(p != first for p in parsed[1:]):
        raise LlmRouteError(f"disagreeing answers: {parsed}")
    return first
