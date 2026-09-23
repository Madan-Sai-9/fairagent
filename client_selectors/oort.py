"""Simplified Oort selector (Lai et al., 2021, OSDI): utility = loss-based
signal + an exploration bonus for clients not picked recently. This is a
simplified version of Oort's full statistical-utility formula, sufficient for
Phase 1's sanity check."""

import math
from client_selectors.base import BaseSelector


class OortSelector(BaseSelector):
    def __init__(self, client_profiles, seed: int = 42, exploration_factor: float = 0.3):
        super().__init__(client_profiles, seed)
        self.exploration_factor = exploration_factor
        self._last_loss = {c.client_id: 1.0 for c in client_profiles}  # neutral prior
        self._rounds_since_selected = {c.client_id: 0 for c in client_profiles}
        self._round_num = 0

    def select(self, k: int):
        self._round_num += 1
        ids = [c.client_id for c in self.client_profiles]

        def utility(cid):
            staleness_bonus = math.sqrt(
                math.log(self._round_num + 1) / (self._rounds_since_selected[cid] + 1)
            )
            return self._last_loss[cid] + self.exploration_factor * staleness_bonus

        ranked = sorted(ids, key=utility, reverse=True)
        selected = sorted(ranked[:k])

        for cid in ids:
            self._rounds_since_selected[cid] = (
                0 if cid in selected else self._rounds_since_selected[cid] + 1
            )

        return selected

    def update_stats(self, client_id: int, loss: float = None, **kwargs):
        if loss is not None:
            self._last_loss[client_id] = loss
