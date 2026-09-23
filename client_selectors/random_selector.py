"""Uniform random client selection — the naive baseline."""

import random
from client_selectors.base import BaseSelector


class RandomSelector(BaseSelector):
    def __init__(self, client_profiles, seed: int = 42):
        super().__init__(client_profiles, seed)
        self._rng = random.Random(seed)

    def select(self, k: int):
        ids = [c.client_id for c in self.client_profiles]
        return sorted(self._rng.sample(ids, k))
