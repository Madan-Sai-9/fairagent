"""
client_selectors/llm_few_shot.py
Few-Shot prompting strategy: same client-facts format as Description-Only,
but preceded by 2-3 worked examples before the real pool is presented.
"""

from __future__ import annotations

from typing import List

from client_selectors.base import ClientProfile
from client_selectors.llm_base import LLMSelectorBase, _render_client_line


_FEW_SHOT_EXAMPLES = """Example 1:
Available clients:
- client_0: 150 local training samples, last observed local loss 1.850.
- client_1: 400 local training samples, last observed local loss 0.620.
- client_2: 220 local training samples, last observed local loss 1.410.
- client_3: 310 local training samples, last observed local loss 0.980.

Select exactly 2 clients.
{"selected_clients": ["client_0", "client_2"], "rationale": "These clients have the highest observed loss, indicating the most room for the global model to improve from training on their data."}

Example 2:
Available clients:
- client_0: 500 local training samples, last observed local loss 0.310.
- client_1: 90 local training samples, not yet measured.
- client_2: 340 local training samples, last observed local loss 0.290.
- client_3: 275 local training samples, last observed local loss 1.120.

Select exactly 2 clients.
{"selected_clients": ["client_1", "client_3"], "rationale": "client_1 has not yet participated and its utility is unknown, worth exploring; client_3 has the highest observed loss among measured clients."}
"""


class LLMFewShotSelector(LLMSelectorBase):
    name = "llm_few_shot"

    def build_prompt(self, client_profiles: List[ClientProfile], k: int) -> str:
        client_lines = "\n".join(
            _render_client_line(c, self._last_loss.get(c.client_id))
            for c in client_profiles
        )
        return f"""You are the orchestrator for a federated learning system. You must select {k} clients out of the available pool to participate in training this round.

Here are two examples of past selections:

{_FEW_SHOT_EXAMPLES}
Now make your own selection for the real client pool below.

Available clients:
{client_lines}

Select exactly {k} clients.

Respond with ONLY a JSON object in this exact format, and nothing else:
{{"selected_clients": ["client_0", "client_5", ...], "rationale": "one sentence explaining your choice"}}
"""
