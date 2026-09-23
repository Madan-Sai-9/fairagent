"""
client_selectors/llm_base.py

Shared plumbing for every LLM-based selector (Description-Only, Few-Shot,
CoT). Each concrete strategy is a thin subclass overriding `build_prompt()`.

Matches the repo's actual BaseSelector contract:
    - __init__(client_profiles, seed=42)
    - select(k) -> list[int]
    - update_stats(client_id, **kwargs)
    - ClientProfile(client_id: int, num_samples: int, stats: dict, metadata_text: str)

Metadata rendering is deliberately neutral (num_samples + tracked stats,
no phrasing variation) — isolates "does the LLM orchestrator work" from
"does phrasing bias its output" (Phase 3's question), which must not be
conflated.
"""

from __future__ import annotations

import json
import re
from typing import List

from client_selectors.base import BaseSelector, ClientProfile, SelectionResult


def _render_client_line(client: ClientProfile, last_loss) -> str:
    loss_str = f"{last_loss:.3f}" if last_loss is not None else "not yet measured"
    extra_stats = {k: v for k, v in (client.stats or {}).items() if k != "last_loss"}
    extra_str = f", additional stats: {extra_stats}" if extra_stats else ""
    return (
        f"- client_{client.client_id}: {client.num_samples} local training samples, "
        f"last observed local loss {loss_str}{extra_str}."
    )


def _extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_client_id(raw_id):
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str):
        match = re.search(r"(\d+)", raw_id)
        if match:
            return int(match.group(1))
    return None


class LLMSelectorBase(BaseSelector):
    name = "llm_base"

    def __init__(self, client_profiles, tokenizer, model, device, seed=42,
                 max_retries=3, max_new_tokens=300):
        super().__init__(client_profiles, seed)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_retries = max_retries
        self.max_new_tokens = max_new_tokens
        self._last_loss = {}
        self.last_raw_output = None
        self.last_retry_count = 0
        self.last_rationale = ""

    def build_prompt(self, client_profiles, k):
        raise NotImplementedError("Subclasses must implement build_prompt()")

    def update_stats(self, client_id, loss=None, **kwargs):
        if loss is not None:
            self._last_loss[client_id] = loss

    def _call_model(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def select(self, k):
        valid_ids = {c.client_id for c in self.client_profiles}
        k_eff = min(k, len(self.client_profiles))
        prompt = self.build_prompt(self.client_profiles, k_eff)
        raw_output = ""
        for attempt in range(self.max_retries):
            raw_output = self._call_model(prompt)
            self.last_raw_output = raw_output
            parsed = _extract_json(raw_output)
            if parsed is not None and "selected_clients" in parsed:
                candidate_ids = []
                for raw_id in parsed["selected_clients"]:
                    cid = _parse_client_id(raw_id)
                    if cid is not None and cid in valid_ids:
                        candidate_ids.append(cid)
                seen = set()
                candidate_ids = [c for c in candidate_ids if not (c in seen or seen.add(c))]
                if len(candidate_ids) >= k_eff:
                    self.last_retry_count = attempt
                    self.last_rationale = parsed.get("rationale", f"LLM selection ({self.name})")
                    return sorted(candidate_ids[:k_eff])
                if candidate_ids:
                    remaining = [c.client_id for c in self.client_profiles if c.client_id not in candidate_ids]
                    padded = candidate_ids + remaining[: k_eff - len(candidate_ids)]
                    self.last_retry_count = attempt
                    self.last_rationale = parsed.get("rationale", "") + " [padded: fewer than k valid ids]"
                    return sorted(padded[:k_eff])
        self.last_retry_count = self.max_retries
        self.last_rationale = f"FALLBACK: exhausted {self.max_retries} retries. Raw: {raw_output[:200]!r}"
        return sorted(c.client_id for c in self.client_profiles)[:k_eff]
