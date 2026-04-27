"""
M.E.M.I. v12.2 — Context Switching
=====================================

Builds on v12.1 (Context-Scoped LearnedModel).

New in v12.2
------------
The system does not assume a context.
It decides which context it is operating in.

Context becomes a governed decision with uncertainty:
    - detected context (from state signals)
    - context confidence (how certain the detection is)
    - operator override (always possible)
    - active context (override if set, else detected)

Context uncertainty is visible in every decision output.
Context switches are logged with reason and confidence.

Two additions:
    ContextDecision       — the result of context detection/resolution
    detect_context()      — infers context from state signals
    resolve_context()     — applies override if set, else uses detected
    ContextSession        — manages active context across steps, logs switches

Design rules
------------
    Context must always be: visible, logged, explainable, overridable.
    Low context confidence does not block action.
    It qualifies admissibility indirectly (via context thresholds).

Architectural invariant (preserved)
-------------------------------------
    The system does not assume a context.
    It decides which context it is operating in.
    Context decisions are logged and overridable.

Bærende sætning
---------------
    The system does not assume a context.
    It decides which context it is operating in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from memi_v112 import (
    WorldState, ACTIONS,
    next_state, epistemic_gating,
    memi_decision, replan,
    semantic_filter,
    build_learning_proposal,
    Trajectory,
)
from memi_v113 import Verdict, govern_learning
from memi_v114 import BASE_ACTION_MODEL
from memi_v115 import BufferStore, flush_buffer
from memi_v120 import (
    Context, CONTEXTS, DEFAULT_CONTEXT,
    is_admissible, evaluate_projected,
)
from memi_v121 import (
    ContextualLearnedModel,
    simulate_contextual,
    select_action_contextual_learned,
    LEARNING_BOUNDS,
)


# ─────────────────────────────────────────────
# Context Decision (NEW in v12.2)
# ─────────────────────────────────────────────

@dataclass
class ContextDecision:
    """
    The result of context detection and resolution.

    detected:    domain inferred from state signals
    confidence:  how certain the detection is (0.0–1.0)
    override:    operator-specified domain (None if not set)
    active:      resolved active context (override if set, else detected)
    reason:      why this domain was detected
    """
    detected:    str
    confidence:  float
    override:    Optional[str]
    active:      str
    reason:      str

    def is_overridden(self) -> bool:
        return self.override is not None

    def __str__(self) -> str:
        src = f"override={self.override}" if self.is_overridden() else \
              f"detected={self.detected} (conf={self.confidence:.2f})"
        return f"context={self.active}  [{src}]  reason: {self.reason}"


# ─────────────────────────────────────────────
# detect_context (NEW in v12.2)
# ─────────────────────────────────────────────

def detect_context(state: WorldState) -> tuple[str, float, str]:
    """
    Infer context from state signals.
    Returns (domain, confidence, reason).

    Detection heuristics (v12.2 — simple, transparent):

    security:
        high risk + high urgency profile (eu moderate)
        fast-moving threat signature

    clinical:
        high reversibility requirement + low risk tolerance
        slow, careful epistemic profile

    industrial:
        balanced risk/reversibility
        default when signals are ambiguous

    These heuristics are intentionally simple and auditable.
    v12.x will extend with richer signal processing.
    """
    risk = state.risk
    rev  = state.reversibility
    eu   = state.epistemic_uncertainty

    # Security signal: high risk, moderate-to-low reversibility, epistemic pressure
    if risk > 0.6 and rev < 0.6 and eu > 0.3:
        confidence = min(0.95, 0.5 + risk * 0.4 + (1 - rev) * 0.2)
        return "security", round(confidence, 2), \
               f"high risk ({risk:.2f}) + low reversibility ({rev:.2f}) + eu pressure ({eu:.2f})"

    # Clinical signal: high reversibility, low risk, cautious epistemic profile
    if rev > 0.7 and risk < 0.4 and eu < 0.4:
        confidence = min(0.90, 0.4 + rev * 0.3 + (1 - risk) * 0.2)
        return "clinical", round(confidence, 2), \
               f"high reversibility ({rev:.2f}) + low risk ({risk:.2f}) + low eu ({eu:.2f})"

    # Security signal: very high risk regardless of other signals
    if risk > 0.85:
        return "security", 0.75, f"very high risk ({risk:.2f}) — security assumed"

    # Default: industrial (balanced, most common)
    confidence = 0.5 + max(0, 0.5 - abs(risk - 0.4) - abs(rev - 0.6)) * 0.4
    return "industrial", round(confidence, 2), \
           f"balanced signals — industrial default (risk={risk:.2f}, rev={rev:.2f})"


def resolve_context(
    state:    WorldState,
    override: Optional[str] = None,
) -> ContextDecision:
    """
    Resolve the active context for a given state.
    Override takes precedence over detection.
    """
    detected, confidence, reason = detect_context(state)

    active = override if override else detected

    return ContextDecision(
        detected=detected,
        confidence=confidence,
        override=override,
        active=active,
        reason=reason,
    )


# ─────────────────────────────────────────────
# ContextSession (NEW in v12.2)
# ─────────────────────────────────────────────

@dataclass
class ContextSwitchEvent:
    step:        int
    from_context:Optional[str]
    to_context:  str
    confidence:  float
    overridden:  bool
    reason:      str


class ContextSession:
    """
    Manages active context across steps.
    Detects context switches and logs them.
    Operator can set override at any time.
    """

    def __init__(self, initial_override: Optional[str] = None):
        self._override:   Optional[str]          = initial_override
        self._active:     Optional[str]           = None
        self._history:    list[ContextDecision]   = []
        self._switches:   list[ContextSwitchEvent] = []

    def set_override(self, domain: Optional[str]) -> None:
        """Operator sets or clears override."""
        self._override = domain

    def step(self, state: WorldState, step_index: int) -> ContextDecision:
        """Resolve context for this step and record any switch."""
        decision = resolve_context(state, self._override)

        # Detect switch
        if self._active is not None and self._active != decision.active:
            self._switches.append(ContextSwitchEvent(
                step=step_index,
                from_context=self._active,
                to_context=decision.active,
                confidence=decision.confidence,
                overridden=decision.is_overridden(),
                reason=decision.reason,
            ))

        self._active = decision.active
        self._history.append(decision)
        return decision

    def current_context(self) -> Optional[Context]:
        if self._active:
            return CONTEXTS.get(self._active)
        return None

    def switch_log(self) -> list[ContextSwitchEvent]:
        return self._switches

    def history(self) -> list[ContextDecision]:
        return self._history


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run(
    initial_state:        WorldState = None,
    operator_override:    Optional[str] = None,
    max_iter:             int = 10,
    mid_override:         Optional[tuple[int, str]] = None,  # (step, domain) — simulate operator intervention
):
    if initial_state is None:
        initial_state = WorldState(risk=0.80, reversibility=0.45, epistemic_uncertainty=0.60)

    state   = initial_state.copy()
    model   = ContextualLearnedModel(BASE_ACTION_MODEL)
    buffers = BufferStore()
    session = ContextSession(initial_override=operator_override)
    traj    = Trajectory()
    mode    = "normal"

    print(f"\n{'─' * 68}")
    print(f"Initial state: {state}")
    if operator_override:
        print(f"Operator override: {operator_override}")
    if mid_override:
        print(f"Mid-run override at step {mid_override[0]}: → {mid_override[1]}")

    for i in range(max_iter):
        # Operator mid-run override
        if mid_override and i == mid_override[0]:
            session.set_override(mid_override[1])
            print(f"\n  ── Operator sets override → {mid_override[1]} ──")

        # Resolve context
        ctx_decision = session.step(state, i)
        context      = CONTEXTS[ctx_decision.active]

        print(f"\n  Step {i+1}  |  {state}")
        print(f"    {ctx_decision}")

        if traj.constraint_streak >= 3:
            mode = "safe"
        else:
            mode = "normal"

        best, gate_reason, adm_map = select_action_contextual_learned(
            state, model, ACTIONS, context, mode=mode
        )

        if gate_reason:
            print(f"    ⚠ GATE: {gate_reason}")

        for a, r in adm_map.items():
            if not r.admissible:
                print(f"    ✗ {a:22s}  {r.reason}")

        decision, authority = memi_decision(state, best)
        effective = replan(decision, best)

        print(f"    ✓ {best:22s}  [{decision:20s}  auth={authority}]")

        if effective is None:
            print("    → Stop: operator required")
            break
        if effective != best:
            print(f"    → Replan: {best} → {effective}")

        state_before = state.copy()
        state        = next_state(state, effective)

        # Learning — scoped to active domain
        proposal = build_learning_proposal(
            action=effective,
            state_before=state_before,
            state_after=state,
            action_model=BASE_ACTION_MODEL,
        )
        if proposal:
            buffers.add(proposal)
            buf = buffers.get(effective)
            flush_ok, _ = buf.should_flush()
            if flush_ok:
                wa    = buf.weighted_adjustment()
                avg_c = sum(p.confidence for p in buf.proposals) / buf.n()
                mag   = max(abs(v) for v in wa.values())
                import uuid as _uuid
                from memi_v112 import LearningProposal
                synth = LearningProposal(
                    proposal_id=f"sw_{_uuid.uuid4().hex[:6]}",
                    action=effective,
                    expected_effect={}, observed_effect={},
                    delta=wa, confidence=round(avg_c, 3),
                    proposed_adjustment=wa,
                    adjustment_magnitude=round(mag, 4),
                    within_bounds=all(abs(v) <= LEARNING_BOUNDS.get(d, 0.05) for d, v in wa.items()),
                    requires_approval=(mag >= 0.05 or avg_c < 0.7),
                    justification=f"context={ctx_decision.active} n={buf.n()}",
                )
                buf.clear()
                verdict = govern_learning(synth)
                if verdict.verdict == Verdict.ACCEPT:
                    entry = model.apply(synth, ctx_decision.active)
                    print(f"    ✎ LEARN [{ctx_decision.active}]  "
                          f"adj={ {k:round(v,4) for k,v in entry.adjustment.items() if abs(v)>0.001} }")

        traj.add({"step": i, "action": effective, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty,
                  "context": ctx_decision.active, "mode": mode})

    # ── Context switch log ─────────────────────────
    switches = session.switch_log()
    if switches:
        print(f"\n  Context switches ({len(switches)}):")
        for sw in switches:
            src = "override" if sw.overridden else f"detected (conf={sw.confidence:.2f})"
            print(f"    Step {sw.step+1}: {sw.from_context} → {sw.to_context}  [{src}]  {sw.reason}")
    else:
        print(f"\n  Context switches: 0 (stable context)")

    print(f"  Final state: {state}")
    print(f"  Domains used: { {s['context'] for s in traj.steps} }")


def run_all():
    print("\n" + "=" * 72)
    print("M.E.M.I. v12.2 — Context Switching")
    print("=" * 72)

    # ── Run 1: Auto-detection, context evolves with state ────────
    print("\n\n── Run 1: Auto-detection — context follows state ──")
    run(
        initial_state=WorldState(risk=0.80, reversibility=0.45, epistemic_uncertainty=0.60),
        max_iter=8,
    )

    # ── Run 2: Operator sets override from the start ─────────────
    print("\n\n── Run 2: Operator override — clinical from start ──")
    run(
        initial_state=WorldState(risk=0.80, reversibility=0.45, epistemic_uncertainty=0.60),
        operator_override="clinical",
        max_iter=6,
    )

    # ── Run 3: Mid-run override — operator intervenes ────────────
    print("\n\n── Run 3: Auto-detection, operator overrides to clinical at step 4 ──")
    run(
        initial_state=WorldState(risk=0.80, reversibility=0.45, epistemic_uncertainty=0.60),
        max_iter=8,
        mid_override=(3, "clinical"),
    )


if __name__ == "__main__":
    run_all()
