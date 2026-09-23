"""Base interface every client selector (numeric or LLM-based) must implement.

Named `client_selectors`, not `selectors` — the latter collides with Python's
standard-library `selectors` module (I/O multiplexing) and gets shadowed by it.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ClientProfile:
    """What a selector is allowed to see about a client when making a decision.
    Numeric selectors use `stats`; future LLM selectors will additionally read
    `metadata_text` (Phase 3's synthetic client descriptions)."""
    client_id: int
    num_samples: int
    stats: dict            # numeric signals: e.g. {'last_loss': ..., 'avg_bandwidth': ...}
    metadata_text: str = ""  # populated starting Phase 3; unused by numeric selectors


@dataclass
class SelectionResult:
    """What every selector returns — kept uniform so downstream logging/analysis
    doesn't need to special-case numeric vs. LLM-based selectors."""
    selected_ids: List[int]
    method: str             # e.g. 'random', 'power_of_choice', 'oort', 'llm_cot'
    extra: dict = None      # selector-specific info (e.g. LLM reasoning trace later)


class BaseSelector:
    """All selectors implement `select(k) -> list[int]` (client ids chosen this round)."""

    def __init__(self, client_profiles: List[ClientProfile], seed: int = 42):
        self.client_profiles = client_profiles
        self.seed = seed

    def select(self, k: int) -> List[int]:
        raise NotImplementedError

    def update_stats(self, client_id: int, **kwargs):
        """Optional hook: selectors like Oort update per-client stats after each
        round (e.g. observed loss) to inform future selections."""
        pass
