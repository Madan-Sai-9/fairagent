"""Power-of-Choice selector (Jee Cho et al., 2022): sample a larger candidate
pool, then keep the k candidates with the highest known local loss."""

import random
from client_selectors.base import BaseSelector


class PowerOfChoiceSelector(BaseSelector):
    def __init__(self, client_profiles, seed: int = 42, d_factor: int = 3):
        super().__init__(client_profiles, seed)
        self._rng = random.Random(seed)
        self.d_factor = d_factor  # candidate pool size = k * d_factor
        # last observed loss per client; unseen clients default to a high
        # loss so they get explored early rather than never selected
        self._last_loss = {c.client_id: float("inf") for c in client_profiles}

    def select(self, k: int):
        ids = [c.client_id for c in self.client_profiles]
        pool_size = min(len(ids), k * self.d_factor)
        candidates = self._rng.sample(ids, pool_size)
        candidates.sort(key=lambda cid: self._last_loss[cid], reverse=True)
        return sorted(candidates[:k])

    def update_stats(self, client_id: int, loss: float = None, **kwargs):
        if loss is not None:
            self._last_loss[client_id] = loss
