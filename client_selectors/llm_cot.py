"""
client_selectors/llm_cot.py
Chain-of-Thought prompting strategy: explicitly instructs step-by-step
reasoning before the final JSON answer.
"""

from __future__ import annotations

import re
from typing import List

from client_selectors.base import ClientProfile
from client_selectors.llm_base import LLMSelectorBase, _render_client_line, _extract_json


class LLMCoTSelector(LLMSelectorBase):
    name = "llm_cot"

    def build_prompt(self, client_profiles: List[ClientProfile], k: int) -> str:
        client_lines = "\n".join(
            _render_client_line(c, self._last_loss.get(c.client_id))
            for c in client_profiles
        )
        return f"""You are the orchestrator for a federated learning system. You must select {k} clients out of the available pool to participate in training this round.

Available clients:
{client_lines}

Think step by step: consider each client's training sample count and observed loss, weigh which clients would most improve the global model this round, and note any client you are uncertain about. Write your reasoning first.

After your reasoning, on a new line, respond with ONLY a JSON object in this exact format:
{{"selected_clients": ["client_0", "client_5", ...], "rationale": "one sentence summary of your reasoning"}}
"""

    def _call_model(self, prompt: str) -> str:
        original_max = self.max_new_tokens
        self.max_new_tokens = max(original_max, 500)
        try:
            return super()._call_model(prompt)
        finally:
            self.max_new_tokens = original_max
