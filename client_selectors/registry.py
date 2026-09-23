"""
client_selectors/registry.py
Name -> constructor registry, so configs/trial runners can reference a
selector by string name. Added in Phase 2 alongside the LLM selectors.
"""

from __future__ import annotations

from typing import Any, Callable

from client_selectors.base import BaseSelector
from client_selectors.random_selector import RandomSelector
from client_selectors.power_of_choice import PowerOfChoiceSelector
from client_selectors.oort import OortSelector

_REGISTRY = {
    "random": RandomSelector,
    "power_of_choice": PowerOfChoiceSelector,
    "oort": OortSelector,
}


def register_selector(name, constructor):
    _REGISTRY[name] = constructor


def build_selector(name, **kwargs):
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown selector '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def available_selectors():
    return sorted(_REGISTRY.keys())


def register_llm_description_only(tokenizer, model, device):
    """Needs a loaded model injected, so can't be built from name+kwargs
    alone like the numeric selectors. Call once after the model loads."""
    from client_selectors.llm_description_only import LLMDescriptionOnlySelector
    register_selector(
        "llm_description_only",
        lambda client_profiles, **kw: LLMDescriptionOnlySelector(
            client_profiles=client_profiles, tokenizer=tokenizer, model=model, device=device, **kw
        ),
    )
