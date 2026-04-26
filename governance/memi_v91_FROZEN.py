"""
M.E.M.I. v9.1 — LLM Plan Proposal Interface
=============================================

Builds on v9.0 (Governed Multi-Step Planning).

New in v9.1
-----------
- LLMPlannerAdapter: abstract interface — the only contract LLM touches
- StubLLMAdapter: deterministic implementation of that contract
- WorldStateSanitizer: strips internal governance state before LLM sees it
- PlanProposalValidator: validates LLM-produced JSON before it enters MEMI
- MultiPlanExecutor: runs 2–3 LLM alternatives through MEMI, returns all results
- PlanSelector: chooses the best PlanResult after MEMI evaluation

Architectural rules (never violated)
--------------------------------------
  LLM may only produce plan proposals.
  LLM may never own execution, authority, cache, or veto.
  LLM sees only sanitized WorldState — never raw governance internals.
  LLM output is always validated before entering MEMI.

Bærende sætning
---------------
  The LLM may imagine the plan.
  M.E.M.I. decides whether the plan is allowed to become action.

Upgrade path
------------
  v9.1: StubLLMAdapter  — deterministic, no API
  v9.2: ClaudeLLMAdapter — real LLM behind same interface (one swap)
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable

from memi_v85 import (
    ActionType, AuthorityLevel, VetoType,
    ModelStatus, SensorStatus, Gap,
)
from memi_v90 import (
    MEMI, MEMIv90, Plan, PlannedStep, PlanResult, PlanStatus,
    Planner, PlanExecutor, WorldState, WorldStateProvider,
    StepExecutionRecord, StepStatus,
)


# ─────────────────────────────────────────────
# Sanitized WorldState
# ─────────────────────────────────────────────

@dataclass
class SanitizedWorldState:
    """
    What the LLM is allowed to see.

    Internal governance state (authority scores, cache entries,
    self-model internals, veto logic) is never exposed to the LLM.
    The LLM sees only observable world properties.

    World-model may never declare itself valid.
    → validity is described in prose, not as a boolean the LLM can act on.
    """
    model_confidence:  str    # "high" | "moderate" | "low" — never raw score
    sensor_coverage:   str    # "full" | "partial" | "degraded"
    sensor_dropout:    bool
    urgency:           str    # "low" | "moderate" | "high" | "critical"
    gap_descriptions:  list[str]   # human-readable gap summaries — no gap IDs
    goal:              str


class WorldStateSanitizer:
    """
    Translates internal WorldState into SanitizedWorldState.

    This is the only crossing point from governance internals to LLM input.
    Nothing passes this boundary except what is explicitly mapped here.
    """

    @staticmethod
    def sanitize(state: WorldState, goal: str) -> SanitizedWorldState:

        # Model confidence — prose only, never raw score
        drift = state.model_status.drift
        if not state.model_status.valid:
            confidence = "low"
        elif drift > 0.5 or state.model_status.critical_drift:
            confidence = "low"
        elif drift > 0.2:
            confidence = "moderate"
        else:
            confidence = "high"

        # Sensor coverage
        cov = state.sensor_status.coverage
        if state.sensor_status.dropout:
            coverage = "degraded"
        elif cov >= 0.9:
            coverage = "full"
        elif cov >= 0.6:
            coverage = "partial"
        else:
            coverage = "degraded"

        # Urgency
        u = state.urgency
        if u >= 0.85:
            urgency = "critical"
        elif u >= 0.6:
            urgency = "high"
        elif u >= 0.35:
            urgency = "moderate"
        else:
            urgency = "low"

        # Gap descriptions — strip IDs and internal fields
        gap_descriptions = [
            f"{g.source}: {g.type}"
            for g in state.gaps
            if g.decision_critical
        ]

        return SanitizedWorldState(
            model_confidence=confidence,
            sensor_coverage=coverage,
            sensor_dropout=state.sensor_status.dropout,
            urgency=urgency,
            gap_descriptions=gap_descriptions,
            goal=goal,
        )


# ─────────────────────────────────────────────
# LLM Adapter interface
# ─────────────────────────────────────────────

class LLMPlannerAdapter(ABC):
    """
    Abstract interface between MEMI and any LLM.

    Contract
    --------
    - Input:  SanitizedWorldState
    - Output: list of raw plan dicts (2–3 alternatives)
    - The adapter may never read or write MEMI internals.
    - The adapter may never call MEMI.step() or modify authority.

    v9.1: StubLLMAdapter  (deterministic)
    v9.2: ClaudeLLMAdapter (real API, same interface)
    """

    @abstractmethod
    def propose(self, state: SanitizedWorldState) -> list[dict]:
        """
        Return 2–3 plan proposals as raw dicts.

        Each dict:
          {
            "goal":  str,
            "steps": [
              {
                "action":        str,   # must be valid ActionType value
                "intent":        str,
                "reversible":    bool,  # optional, default True
                "whitelisted":   bool,  # optional, default False
                "preconditions": list[str]
              },
              ...
            ]
          }
        """
        ...


# ─────────────────────────────────────────────
# Stub LLM Adapter (v9.1)
# ─────────────────────────────────────────────

class StubLLMAdapter(LLMPlannerAdapter):
    """
    Deterministic stub that produces realistic, varied plan proposals
    based on sanitized world state — no API, no model, no randomness.

    Behaviour matrix
    ----------------
    urgency=critical, dropout       → short hold-and-monitor plan
    urgency=high, gaps present      → conservative: monitor → hold → monitor
    urgency=moderate, low gaps      → standard:     monitor → stabilize → adjust → monitor
    urgency=low, high confidence    → full:         monitor → stabilize → adjust → monitor → stabilize
    sensor=degraded                 → always leads with monitor, avoids stabilize as first action
    model_confidence=low            → avoids adjust and stabilize in early steps

    Always returns 2–3 alternatives with varying risk profiles:
      Plan A — conservative
      Plan B — standard
      Plan C — aggressive (only when conditions are good)
    """

    def propose(self, state: SanitizedWorldState) -> list[dict]:
        proposals = []

        # ── Conservative plan (always present) ──────────────────
        conservative = self._conservative(state)
        proposals.append(conservative)

        # ── Standard plan ────────────────────────────────────────
        if state.urgency not in ("critical",) and state.model_confidence != "low":
            proposals.append(self._standard(state))

        # ── Aggressive plan (good conditions only) ───────────────
        if (
            state.urgency in ("low", "moderate")
            and state.model_confidence == "high"
            and state.sensor_coverage == "full"
            and not state.gap_descriptions
        ):
            proposals.append(self._aggressive(state))

        return proposals

    # ── Plan templates ────────────────────────────────────────────

    def _conservative(self, state: SanitizedWorldState) -> dict:
        """
        Conservative plan: monitor-only or monitor + gentle stabilize.
        LLM never proposes 'hold' — holding is MEMI's authority decision,
        not something the proposal engine chooses.
        """
        steps = [
            {"action": "monitor",   "intent": "Assess current state before any action",
             "reversible": True},
            {"action": "monitor",   "intent": "Second observation to confirm reading",
             "reversible": True},
        ]
        if state.urgency in ("low",) and state.model_confidence == "high":
            steps.append(
                {"action": "stabilize", "intent": "Gentle stabilise only if state confirmed safe",
                 "reversible": True}
            )
        steps.append(
            {"action": "monitor",   "intent": "Final check after conservative action",
             "reversible": True}
        )
        return {
            "goal":  f"[Conservative] {state.goal}",
            "steps": steps,
        }

    def _standard(self, state: SanitizedWorldState) -> dict:
        steps = [
            {"action": "monitor",   "intent": "Establish baseline",            "reversible": True},
            {"action": "stabilize", "intent": "Bring system to safe state",    "reversible": True},
        ]
        if not state.gap_descriptions and state.sensor_coverage != "degraded":
            steps.append(
                {"action": "adjust",  "intent": "Tune parameters within safe range", "reversible": True}
            )
        steps.append(
            {"action": "monitor",   "intent": "Verify outcome",                "reversible": True}
        )
        return {
            "goal":  f"[Standard] {state.goal}",
            "steps": steps,
        }

    def _aggressive(self, state: SanitizedWorldState) -> dict:
        return {
            "goal":  f"[Aggressive] {state.goal}",
            "steps": [
                {"action": "stabilize", "intent": "Move directly to target state",   "reversible": True},
                {"action": "adjust",    "intent": "Optimise parameters immediately", "reversible": True},
                {"action": "stabilize", "intent": "Lock in optimised state",         "reversible": True},
                {"action": "monitor",   "intent": "Verify full optimisation",        "reversible": True},
            ],
        }


# ─────────────────────────────────────────────
# Plan Proposal Validator
# ─────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid:    bool
    plan:     Optional[dict]
    errors:   list[str] = field(default_factory=list)


class PlanProposalValidator:
    """
    Validates raw LLM-produced plan dicts before they enter MEMI.

    This is the final sanitization gate. Nothing enters the Planner
    unless it passes all checks here.

    Rules
    -----
    - Must have "goal" (non-empty string)
    - Must have "steps" (non-empty list, max 5)
    - Each step must have "action" matching a valid ActionType value
    - Each step must have "intent" (non-empty string)
    - action "defer_to_operator" and "hold" are never allowed in LLM proposals
      (deference and holding are MEMI's decisions, never the LLM's)
    - Unknown fields are stripped silently
    """

    FORBIDDEN_ACTIONS = {ActionType.DEFER.value, ActionType.HOLD.value}
    MAX_STEPS = 5

    def validate(self, raw: dict) -> ValidationResult:
        errors = []

        # Goal
        goal = raw.get("goal", "")
        if not isinstance(goal, str) or not goal.strip():
            errors.append("Missing or empty 'goal'.")

        # Steps
        steps = raw.get("steps", [])
        if not isinstance(steps, list) or len(steps) == 0:
            errors.append("'steps' must be a non-empty list.")
            return ValidationResult(valid=False, plan=None, errors=errors)

        if len(steps) > self.MAX_STEPS:
            errors.append(
                f"Plan has {len(steps)} steps — maximum is {self.MAX_STEPS}. "
                "Truncating to first 5."
            )
            steps = steps[:self.MAX_STEPS]

        valid_steps = []
        valid_action_values = {a.value for a in ActionType}

        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                errors.append(f"Step {i} is not a dict.")
                continue

            action = s.get("action", "")
            if action not in valid_action_values:
                errors.append(
                    f"Step {i}: unknown action '{action}'. "
                    f"Valid: {sorted(valid_action_values)}"
                )
                continue

            if action in self.FORBIDDEN_ACTIONS:
                errors.append(
                    f"Step {i}: action '{action}' is forbidden in LLM proposals. "
                    "Deference is MEMI's decision."
                )
                continue

            intent = s.get("intent", "")
            if not isinstance(intent, str) or not intent.strip():
                errors.append(f"Step {i}: missing or empty 'intent'.")
                continue

            # Accept only known fields — strip the rest silently
            valid_steps.append({
                "action":        action,
                "intent":        intent.strip(),
                "reversible":    bool(s.get("reversible", True)),
                "whitelisted":   bool(s.get("whitelisted", False)),
                "preconditions": [str(p) for p in s.get("preconditions", [])],
            })

        if not valid_steps:
            errors.append("No valid steps remain after validation.")
            return ValidationResult(valid=False, plan=None, errors=errors)

        cleaned = {"goal": goal.strip(), "steps": valid_steps}
        # Validation passes even with warnings (truncation, stripped fields)
        hard_failure = any(
            "unknown action" in e or "is not a dict" in e
            or "No valid steps" in e or "missing or empty 'goal'" in e
            or "forbidden" in e
            for e in errors
        )
        return ValidationResult(
            valid=not hard_failure,
            plan=cleaned if not hard_failure else None,
            errors=errors,
        )


# ─────────────────────────────────────────────
# Multi-Plan Executor
# ─────────────────────────────────────────────

@dataclass
class MultiPlanResult:
    """Results from running all LLM-proposed alternatives through MEMI."""
    goal:              str
    proposals_offered: int
    proposals_valid:   int
    results:           list[PlanResult]
    selected:          Optional[PlanResult]
    selection_reason:  str
    validation_errors: list[list[str]]


class MultiPlanExecutor:
    """
    Takes 2–3 LLM plan proposals, validates each, runs all through MEMI,
    then selects the best result.

    The LLM proposes. MEMI runs. This class only coordinates.
    """

    def __init__(self, memi_system: MEMIv90):
        self._system = memi_system
        self._validator = PlanProposalValidator()
        self._planner = Planner()

    def run_proposals(
        self,
        raw_proposals: list[dict],
        state_provider: WorldStateProvider,
        on_step: Optional[Callable[[int, StepExecutionRecord], None]] = None,
    ) -> MultiPlanResult:
        """
        Validate and execute all proposals. Return MultiPlanResult.
        on_step(proposal_index, record) called after each step if provided.
        """
        results: list[PlanResult]     = []
        validation_errors: list[list[str]] = []
        valid_count = 0

        for idx, raw in enumerate(raw_proposals):
            vr = self._validator.validate(raw)
            validation_errors.append(vr.errors)

            if not vr.valid:
                continue

            valid_count += 1
            plan = self._planner.create(
                goal=vr.plan["goal"],
                steps=vr.plan["steps"],
                plan_id=f"llm_{idx}_{str(uuid.uuid4())[:6]}",
            )

            cb = None
            if on_step:
                captured_idx = idx
                def cb(rec, i=captured_idx):
                    on_step(i, rec)

            result = self._system.execute(plan, state_provider, on_step=cb)
            results.append(result)

        selected, reason = PlanSelector.select(results)

        return MultiPlanResult(
            goal=raw_proposals[0].get("goal", "") if raw_proposals else "",
            proposals_offered=len(raw_proposals),
            proposals_valid=valid_count,
            results=results,
            selected=selected,
            selection_reason=reason,
            validation_errors=validation_errors,
        )


# ─────────────────────────────────────────────
# Plan Selector
# ─────────────────────────────────────────────

class PlanSelector:
    """
    Chooses the best PlanResult after MEMI has evaluated all alternatives.

    Selection priority
    ------------------
    1. COMPLETED plans ranked by steps_executed (most progress first)
    2. PAUSED plans (partial progress > no progress)
    3. HANDED_OFF plans
    4. ABORTED plans (last resort)

    Within same status: prefer fewer deferred steps (less escalation needed).
    """

    STATUS_ORDER = {
        PlanStatus.COMPLETED:  0,
        PlanStatus.PAUSED:     1,
        PlanStatus.HANDED_OFF: 2,
        PlanStatus.ABORTED:    3,
        PlanStatus.RUNNING:    4,
        PlanStatus.PENDING:    5,
    }

    @classmethod
    def select(cls, results: list[PlanResult]) -> tuple[Optional[PlanResult], str]:
        if not results:
            return None, "No valid proposals were executed."

        ranked = sorted(
            results,
            key=lambda r: (
                cls.STATUS_ORDER.get(r.plan_status, 99),
                -r.steps_executed,
                r.steps_deferred,
            ),
        )
        best = ranked[0]
        reason = (
            f"Selected '{best.goal}': "
            f"status={best.plan_status.value}, "
            f"executed={best.steps_executed}/{best.steps_total}, "
            f"deferred={best.steps_deferred}"
        )
        return best, reason


# ─────────────────────────────────────────────
# Governed LLM Planning Session (top-level façade)
# ─────────────────────────────────────────────

class GovernedLLMSession:
    """
    Full v9.1 execution loop:

      goal + WorldState
        → WorldStateSanitizer
        → LLMPlannerAdapter.propose()    [stub or real]
        → PlanProposalValidator (each proposal)
        → MultiPlanExecutor (all valid proposals through MEMI)
        → PlanSelector
        → MultiPlanResult

    The LLM never touches MEMI. MEMI never touches the LLM.
    The only shared data structure is Plan/PlannedStep.
    """

    def __init__(
        self,
        memi_system: MEMIv90,
        llm_adapter: LLMPlannerAdapter,
    ):
        self._system   = memi_system
        self._llm      = llm_adapter
        self._executor = MultiPlanExecutor(memi_system)
        self._sanitizer = WorldStateSanitizer()

    def run(
        self,
        goal: str,
        world_state: WorldState,
        on_step: Optional[Callable[[int, StepExecutionRecord], None]] = None,
    ) -> MultiPlanResult:
        """Execute a full governed LLM planning session."""

        # Step 1 — Sanitize
        sanitized = WorldStateSanitizer.sanitize(world_state, goal)

        # Step 2 — LLM proposes (sees only sanitized state)
        raw_proposals = self._llm.propose(sanitized)

        # Step 3 — Validate + execute all proposals through MEMI
        state_provider = WorldStateProvider(world_state)
        return self._executor.run_proposals(
            raw_proposals=raw_proposals,
            state_provider=state_provider,
            on_step=on_step,
        )


# ─────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────

def print_multi_result(mr: MultiPlanResult) -> None:
    print(f"\n  Proposals offered: {mr.proposals_offered}  "
          f"valid: {mr.proposals_valid}")
    for i, r in enumerate(mr.results):
        fast = sum(1 for rec in r.records if rec.fast_lane)
        print(f"\n  [Plan {i}] '{r.goal}'")
        print(f"    status={r.plan_status.value}  "
              f"executed={r.steps_executed}/{r.steps_total}  "
              f"fast_lane={fast}/{r.steps_total}  "
              f"deferred={r.steps_deferred}")
        for rec in r.records:
            fl = " ⚡" if rec.fast_lane else "  "
            print(f"    {fl}[{rec.step_index}] {rec.action.value:12s} "
                  f"auth={rec.authority.value:6s}  "
                  f"status={rec.status.value:10s}  "
                  f"{rec.intent}")
    if mr.selected:
        print(f"\n  ✓ Selected: {mr.selection_reason}")
    else:
        print(f"\n  ✗ No plan selected. {mr.selection_reason}")


def demo():
    print("=" * 65)
    print("M.E.M.I. v9.1 — LLM Plan Proposal Interface (Stub)")
    print("=" * 65)

    system  = MEMIv90(operator_reversibility=0.85, cache_ttl=600.0)
    stub    = StubLLMAdapter()
    session = GovernedLLMSession(memi_system=system, llm_adapter=stub)

    healthy = WorldState(
        model_status=ModelStatus(valid=True, drift=0.05),
        sensor_status=SensorStatus(dropout=False, coverage=1.0),
        gaps=[],
        urgency=0.3,
    )

    # ── Case 1: Healthy conditions — all proposals should complete ───
    print("\n\n── Case 1: Healthy conditions ──")
    print("   (expect 3 proposals: conservative, standard, aggressive)")
    r1 = session.run("Optimise process output after steady-state detection", healthy)
    print_multi_result(r1)

    # ── Case 2: Degraded sensor — conservative + standard only ──────
    print("\n\n── Case 2: Degraded sensor coverage ──")
    print("   (expect 2 proposals: conservative + standard)")
    degraded = WorldState(
        model_status=ModelStatus(valid=True, drift=0.1),
        sensor_status=SensorStatus(dropout=False, coverage=0.55),
        gaps=[Gap(id="g1", source="sensor", type="coverage_loss",
                  would_change_decision=True, decision_critical=True)],
        urgency=0.5,
    )
    r2 = session.run("Maintain stability under partial sensor loss", degraded)
    print_multi_result(r2)

    # ── Case 3: Critical urgency + dropout — VETO_1 fires ──────────
    print("\n\n── Case 3: Critical urgency + sensor dropout ──")
    print("   (expect 1 proposal: conservative, VETO_1 fires on stabilize → plan paused)")
    critical = WorldState(
        model_status=ModelStatus(valid=True, drift=0.05),
        sensor_status=SensorStatus(dropout=True, coverage=0.2),
        gaps=[],
        urgency=0.92,
    )
    r3 = session.run("Emergency stabilise under sensor dropout", critical)
    print_multi_result(r3)

    # ── Case 4: Low confidence model — no aggressive plan ───────────
    print("\n\n── Case 4: Low model confidence ──")
    print("   (expect 2 proposals, no aggressive plan)")
    low_conf = WorldState(
        model_status=ModelStatus(valid=True, drift=0.65, critical_drift=False),
        sensor_status=SensorStatus(dropout=False, coverage=0.9),
        gaps=[],
        urgency=0.4,
    )
    r4 = session.run("Stabilise after model drift detected", low_conf)
    print_multi_result(r4)

    # ── Case 5: Validator rejects forbidden actions in proposals ────
    print("\n\n── Case 5: Validator rejects 'defer_to_operator' and 'hold' in proposals ──")
    validator = PlanProposalValidator()
    bad_proposal = {
        "goal": "Test forbidden actions",
        "steps": [
            {"action": "monitor",          "intent": "Look around"},
            {"action": "defer_to_operator","intent": "LLM trying to own deference"},
            {"action": "hold",             "intent": "LLM trying to own holding"},
            {"action": "stabilize",        "intent": "Then stabilise"},
        ]
    }
    vr = validator.validate(bad_proposal)
    print(f"  valid={vr.valid}  (remaining steps after stripping forbidden: "
          f"{len(vr.plan['steps']) if vr.plan else 0})")
    for e in vr.errors:
        print(f"  ⚠ {e}")

    # ── Case 6: Second session — fast-lane from earlier mandates ────
    print("\n\n── Case 6: Second run (same conditions) — fast-lane activates ──")
    r6 = session.run("Routine check (known-good conditions)", healthy)
    print_multi_result(r6)

    # ── Diagnostics ─────────────────────────────────────────────────
    print("\n\n── Cache summary ──")
    import pprint
    pprint.pprint(system.cache_summary())
    print("\n── Self-model ──")
    pprint.pprint(system.self_model())
    print("\n✓ v9.1 smoke-test complete.\n")


if __name__ == "__main__":
    demo()
