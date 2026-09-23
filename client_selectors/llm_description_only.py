"""
client_selectors/llm_description_only.py
Description-Only prompting strategy: list client facts plainly, ask for
a selection, no worked examples, no explicit reasoning instruction.
"""

from __future__ import annotations

from typing import List

from client_selectors.base import ClientProfile
from client_selectors.llm_base import LLMSelectorBase, _render_client_line


class LLMDescriptionOnlySelector(LLMSelectorBase):
    name = "llm_description_only"

    def build_prompt(self, client_profiles: List[ClientProfile], k: int) -> str:
        client_lines = "\n".join(
            _render_client_line(c, self._last_loss.get(c.client_id))
            for c in client_profiles
        )
        return f"""You are the orchestrator for a federated learning system. You must select {k} clients out of the available pool to participate in training this round.

Available clients:
{client_lines}

Select exactly {k} clients. Prefer clients likely to improve the global model this round (e.g. higher observed loss suggests more room to improve).

Respond with ONLY a JSON object in this exact format, and nothing else:
{{"selected_clients": ["client_0", "client_5", ...], "rationale": "one sentence explaining your choice"}}
"""
