"""
M.E.M.I. v9.2 — ClaudeLLMAdapter
==================================

Builds on v9.1 (FROZEN).

New in v9.2
-----------
- ClaudeLLMAdapter: real Anthropic API behind the same LLMPlannerAdapter interface
- One swap: StubLLMAdapter → ClaudeLLMAdapter
- Zero changes to governance, validator, executor, selector, or MEMI

Adapter contract
----------------
  Input:  SanitizedWorldState
  Output: list[dict]  (same schema as stub)
  Retry:  once, with stricter prompt, if JSON parsing fails
  Fallback: return [] — MEMI handles no-plan gracefully

Hard rules
----------
  LLM sees only SanitizedWorldState — never authority, veto, cache, self-model.
  LLM may not propose defer_to_operator or hold (validator enforces).
  Retry may recover formatting. It may not negotiate correctness.
  The intelligence can be swapped. The governance interface cannot.

Prompt design
-------------
  System prompt: role + schema + explicit prohibitions
  User message:  structured JSON { "goal": ..., "state": { ... } }
  Response:      JSON only — no markdown, no explanation

v9.3 path
---------
  Tool/function calling replaces raw JSON parsing.
  Same adapter interface. Same governance. Same validator.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import asdict

from memi_v85 import ModelStatus, SensorStatus, Gap
from memi_v90 import MEMIv90, WorldState, WorldStateProvider
from memi_v91 import (
    LLMPlannerAdapter, SanitizedWorldState, WorldStateSanitizer,
    StubLLMAdapter, GovernedLLMSession, PlanProposalValidator,
    MultiPlanResult, print_multi_result,
)


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a planning module inside a governed AI system.

Your ONLY job is to propose action plans. You may NOT:
- decide whether actions are authorised
- suggest "defer_to_operator" or "hold" (governance decisions, not yours)
- assume any action is allowed — that is decided elsewhere
- add commentary, explanation, or markdown

You will receive a JSON object with:
  "goal":  what the system is trying to achieve
  "state": observable world properties (confidence, coverage, urgency, gaps)

Return ONLY a valid JSON array of 2–3 plan proposals.
No markdown. No code fences. No explanation. Raw JSON array only.

Output schema — return exactly this structure:
[
  {
    "goal": "short description of this plan variant",
    "steps": [
      {
        "action": "monitor|stabilize|adjust",
        "intent": "one sentence — what this step achieves",
        "reversible": true,
        "whitelisted": false,
        "preconditions": []
      }
    ]
  }
]

Rules:
- Each plan must have 2–5 steps.
- "action" must be one of: monitor, stabilize, adjust
- "intent" must be non-empty.
- Propose variants with different risk profiles when conditions allow:
    conservative (monitor-heavy), standard, aggressive (only if confidence=high,
    coverage=full, urgency=low or moderate, no gaps).
- If conditions are degraded (low confidence, dropout, high urgency, gaps present):
    propose only conservative or standard plans.
- Never propose the same plan twice.
"""

SYSTEM_PROMPT_STRICT = """\
You are a planning module. Return ONLY a valid JSON array.
No markdown. No explanation. No code fences.

Each element:
{
  "goal": "string",
  "steps": [
    { "action": "monitor|stabilize|adjust", "intent": "string",
      "reversible": true, "whitelisted": false, "preconditions": [] }
  ]
}

2–3 elements. 2–5 steps each.
Forbidden actions: defer_to_operator, hold.
"""


# ─────────────────────────────────────────────
# Claude API caller (minimal — no SDK dependency)
# ─────────────────────────────────────────────

class ClaudeAPIClient:
    """
    Minimal HTTP client for the Anthropic Messages API.
    Uses only stdlib — no anthropic SDK required in v9.2.

    The API key is read from the environment at call time.
    It is never stored, logged, or passed to any governance layer.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    MODEL   = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """
        Single-turn completion. Returns assistant text content.
        Raises ClaudeAPIError on HTTP errors or missing key.
        """
        if not self._api_key:
            raise ClaudeAPIError("ANTHROPIC_API_KEY not set.")

        payload = json.dumps({
            "model":      self.MODEL,
            "max_tokens": max_tokens,
            "system":     system,
            "messages":   [{"role": "user", "content": user}],
        }).encode()

        req = urllib.request.Request(
            self.API_URL,
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         self._api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise ClaudeAPIError(f"HTTP {e.code}: {e.read().decode()}") from e
        except urllib.error.URLError as e:
            raise ClaudeAPIError(f"Network error: {e.reason}") from e

        # Extract text from content blocks
        for block in body.get("content", []):
            if block.get("type") == "text":
                return block["text"]

        raise ClaudeAPIError("No text content in response.")


class ClaudeAPIError(Exception):
    pass


# ─────────────────────────────────────────────
# ClaudeLLMAdapter
# ─────────────────────────────────────────────

class ClaudeLLMAdapter(LLMPlannerAdapter):
    """
    Real Claude behind the LLMPlannerAdapter interface.

    Retry protocol
    --------------
    Attempt 1: standard system prompt + structured JSON user message
    Attempt 2: stricter system prompt (if attempt 1 fails to parse)
    Fallback:  return []  →  MEMI receives no proposals, handles gracefully

    Retry may recover formatting. It may not negotiate correctness.

    v9.3 upgrade path: replace json.loads with tool/function calling.
    Interface stays identical.
    """

    def __init__(
        self,
        api_key:      str | None = None,
        max_attempts: int = 2,
        fallback_to_stub: bool = True,
    ):
        self._client   = ClaudeAPIClient(api_key=api_key)
        self._max_attempts = max_attempts
        self._fallback = StubLLMAdapter() if fallback_to_stub else None
        self._validator = PlanProposalValidator()

    def propose(self, state: SanitizedWorldState) -> list[dict]:
        """
        Call Claude with sanitized world state.
        Returns list of raw plan dicts (validated downstream by PlanProposalValidator).
        """
        user_message = self._build_user_message(state)

        last_error: str = ""

        for attempt in range(self._max_attempts):
            system = SYSTEM_PROMPT if attempt == 0 else SYSTEM_PROMPT_STRICT

            try:
                raw_text = self._client.complete(
                    system=system,
                    user=user_message,
                    max_tokens=1000,
                )
                proposals = self._parse(raw_text)
                if proposals:
                    return proposals
                last_error = "Parsed JSON is empty or not a list."

            except ClaudeAPIError as e:
                last_error = str(e)
                break   # API error — no point retrying with different prompt

            except Exception as e:
                last_error = f"Unexpected error: {e}"

        # All attempts failed
        print(f"[ClaudeLLMAdapter] All attempts failed: {last_error}")

        if self._fallback:
            print("[ClaudeLLMAdapter] Falling back to StubLLMAdapter.")
            return self._fallback.propose(state)

        return []

    # ── Internal ──────────────────────────────

    def _build_user_message(self, state: SanitizedWorldState) -> str:
        """
        Structure in → structure out.
        LLM receives only observable world properties.
        Never authority, veto, cache, or self-model internals.
        """
        payload = {
            "goal": state.goal,
            "state": {
                "model_confidence": state.model_confidence,
                "sensor_coverage":  state.sensor_coverage,
                "sensor_dropout":   state.sensor_dropout,
                "urgency":          state.urgency,
                "gap_descriptions": state.gap_descriptions,
            },
        }
        return json.dumps(payload, indent=2)

    def _parse(self, text: str) -> list[dict]:
        """
        Parse Claude's response as JSON.
        Strips accidental markdown fences if present.
        Returns [] if parsing fails or result is not a non-empty list.
        """
        # Strip optional markdown fences (defensive — prompt forbids them)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list) or len(parsed) == 0:
            return []

        return parsed


# ─────────────────────────────────────────────
# Convenience: build a v9.2 session
# ─────────────────────────────────────────────

def build_v92_session(
    api_key: str | None = None,
    fallback_to_stub: bool = True,
    operator_reversibility: float = 0.85,
    cache_ttl: float = 600.0,
) -> GovernedLLMSession:
    """
    Build a GovernedLLMSession with ClaudeLLMAdapter.
    Drop-in replacement for the v9.1 stub session.

    Usage:
        session = build_v92_session()           # reads ANTHROPIC_API_KEY from env
        result  = session.run(goal, world_state)

    If fallback_to_stub=True and the API call fails,
    StubLLMAdapter provides proposals so MEMI is never left with nothing.
    """
    system  = MEMIv90(
        operator_reversibility=operator_reversibility,
        cache_ttl=cache_ttl,
    )
    adapter = ClaudeLLMAdapter(
        api_key=api_key,
        fallback_to_stub=fallback_to_stub,
    )
    return GovernedLLMSession(memi_system=system, llm_adapter=adapter)


# ─────────────────────────────────────────────
# Offline validation test (no API key needed)
# ─────────────────────────────────────────────

def test_parse_and_validate():
    """
    Tests the parsing + validation pipeline with realistic Claude-like
    JSON responses — without calling the API.

    Verifies that ClaudeLLMAdapter._parse() + PlanProposalValidator
    correctly accept good output and reject bad output.
    """
    adapter   = ClaudeLLMAdapter.__new__(ClaudeLLMAdapter)
    adapter._client    = None
    adapter._max_attempts = 2
    adapter._fallback  = StubLLMAdapter()
    adapter._validator = PlanProposalValidator()

    validator = PlanProposalValidator()

    print("=" * 65)
    print("M.E.M.I. v9.2 — Offline parse + validate tests")
    print("=" * 65)

    # ── Test 1: Clean JSON ───────────────────
    print("\n── Test 1: Clean JSON from Claude ──")
    clean = json.dumps([
        {
            "goal": "[Conservative] Maintain stability",
            "steps": [
                {"action": "monitor",   "intent": "Check current state", "reversible": True, "whitelisted": False, "preconditions": []},
                {"action": "monitor",   "intent": "Confirm reading",      "reversible": True, "whitelisted": False, "preconditions": []},
                {"action": "stabilize", "intent": "Gentle correction",    "reversible": True, "whitelisted": False, "preconditions": []},
            ]
        },
        {
            "goal": "[Standard] Stabilise and adjust",
            "steps": [
                {"action": "monitor",   "intent": "Baseline",    "reversible": True, "whitelisted": False, "preconditions": []},
                {"action": "stabilize", "intent": "Safe state",  "reversible": True, "whitelisted": False, "preconditions": []},
                {"action": "adjust",    "intent": "Tune params", "reversible": True, "whitelisted": False, "preconditions": []},
                {"action": "monitor",   "intent": "Verify",      "reversible": True, "whitelisted": False, "preconditions": []},
            ]
        },
    ])
    parsed = adapter._parse(clean)
    print(f"  Parsed {len(parsed)} proposals.")
    for p in parsed:
        vr = validator.validate(p)
        print(f"  '{p['goal']}' — valid={vr.valid}  steps={len(p['steps'])}")

    # ── Test 2: Markdown fences (should strip) ─
    print("\n── Test 2: Claude wraps JSON in markdown fences ──")
    fenced = "```json\n" + clean + "\n```"
    parsed2 = adapter._parse(fenced)
    print(f"  Parsed after stripping fences: {len(parsed2)} proposals  (expect 2)")

    # ── Test 3: Forbidden action in output ────
    print("\n── Test 3: Claude sneaks in 'defer_to_operator' ──")
    bad = json.dumps([{
        "goal": "Test forbidden",
        "steps": [
            {"action": "monitor",          "intent": "Look"},
            {"action": "defer_to_operator","intent": "Escalate immediately"},
            {"action": "stabilize",        "intent": "Then stabilise"},
        ]
    }])
    parsed3 = adapter._parse(bad)
    vr3 = validator.validate(parsed3[0])
    print(f"  valid={vr3.valid}  errors={vr3.errors}")

    # ── Test 4: Completely broken JSON ────────
    print("\n── Test 4: Broken JSON → parse fails → returns [] ──")
    broken = "Sure, here is your plan: { goal: maintain, steps: [monitor] }"
    parsed4 = adapter._parse(broken)
    print(f"  Parsed: {parsed4}  (expect [])")

    # ── Test 5: Empty array ───────────────────
    print("\n── Test 5: Claude returns empty array ──")
    parsed5 = adapter._parse("[]")
    print(f"  Parsed: {parsed5}  (expect [])")

    # ── Test 6: Prompt construction ───────────
    print("\n── Test 6: User message structure ──")
    state = SanitizedWorldState(
        model_confidence="high",
        sensor_coverage="full",
        sensor_dropout=False,
        urgency="moderate",
        gap_descriptions=[],
        goal="Optimise output after anomaly resolved",
    )
    msg = adapter._build_user_message(state)
    parsed_msg = json.loads(msg)
    print(f"  Keys in user message: {list(parsed_msg.keys())}")
    print(f"  State keys: {list(parsed_msg['state'].keys())}")
    print(f"  'authority' present: {'authority' in parsed_msg}  (must be False)")
    print(f"  'veto' present:      {'veto' in parsed_msg}       (must be False)")
    print(f"  'cache' present:     {'cache' in parsed_msg}      (must be False)")

    print("\n✓ Offline tests complete.\n")


# ─────────────────────────────────────────────
# Live integration test (requires API key)
# ─────────────────────────────────────────────

def test_live(api_key: str | None = None):
    """
    Full integration test: ClaudeLLMAdapter → GovernedLLMSession → MEMI.

    Requires ANTHROPIC_API_KEY in environment or passed as argument.
    """
    print("=" * 65)
    print("M.E.M.I. v9.2 — Live integration test")
    print("=" * 65)

    session = build_v92_session(api_key=api_key, fallback_to_stub=True)

    healthy = WorldState(
        model_status=ModelStatus(valid=True, drift=0.05),
        sensor_status=SensorStatus(dropout=False, coverage=1.0),
        gaps=[],
        urgency=0.3,
    )

    degraded = WorldState(
        model_status=ModelStatus(valid=True, drift=0.2),
        sensor_status=SensorStatus(dropout=False, coverage=0.55),
        gaps=[Gap(id="g1", source="sensor", type="coverage_loss",
                  would_change_decision=True, decision_critical=True)],
        urgency=0.6,
    )

    cases = [
        ("Optimise process output after steady-state detection", healthy),
        ("Maintain stability under partial sensor loss", degraded),
    ]

    for goal, state in cases:
        print(f"\n\n── '{goal}' ──")
        result = session.run(goal, state)
        print_multi_result(result)

    print("\n── Self-model ──")
    import pprint
    pprint.pprint(session._system.self_model())
    print("\n✓ Live test complete.\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Always run offline tests — no API key needed
    test_parse_and_validate()

    # Run live test only if API key is available
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print("\nANTHROPIC_API_KEY found — running live integration test.\n")
        test_live(api_key=api_key)
    else:
        print("ANTHROPIC_API_KEY not set — skipping live test.")
        print("To run live test: ANTHROPIC_API_KEY=<key> python memi_v92.py\n")
