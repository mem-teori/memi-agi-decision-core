"""
M.E.M.I. v11.3 — Govern Learning
==================================

Builds on v11.2 (Observe Outcome + Build Learning Proposal).

New in v11.3
------------
Learning proposals pass through a governance layer.
Every proposal receives a verdict: ACCEPT / QUEUE / REJECT.

One addition:
    govern_learning()     — evaluates a LearningProposal against
                            constitutional limits and confidence thresholds
                            and returns a GoverningVerdict

One new dataclass:
    GoverningVerdict      — the result of the governance check

Still unchanged:
    ACTION_MODEL          — not modified
    memi_decision()       — not modified
    epistemic_gating()    — not modified
    VETO conditions       — not modified

The model is still not updated.
v11.4 will apply approved proposals.

Verdict logic
-------------
    REJECT  — proposal violates constitutional limits (touches veto,
               gating principle, or uncertainty_travels) or is out of bounds
    QUEUE   — within bounds but confidence < 0.7 or magnitude >= 0.05;
               held for operator review or multi-observation averaging (v11.4)
    ACCEPT  — within bounds, confidence >= 0.7, magnitude < 0.05,
               does not touch constitutional dimensions

Constitutional dimensions (never adjustable)
---------------------------------------------
    memi_decision()       — not a learnable parameter
    VETO_1 / VETO_2       — not learnable
    epistemic_gating      — principle not learnable (threshold may adjust marginally in v11.5)
    uncertainty_travels   — invariant

Bærende sætning
---------------
    The system does not learn yet.
    It decides what it is allowed to learn.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Re-use v11.2 primitives
from memi_v112 import (
    WorldState, ACTION_MODEL, ACTIONS, EPISTEMIC_GATE,
    UNCERTAINTY_PENALTY_WEIGHT, LEARNING_BOUNDS,
    LearningProposal,
    simulate_from_model, next_state, epistemic_gating,
    memi_decision, replan, evaluate_projected,
    semantic_filter, best_single_step, select_action_2step,
    observe_outcome, build_learning_proposal,
    Trajectory,
)


# ─────────────────────────────────────────────
# Governing Verdict (NEW in v11.3)
# ─────────────────────────────────────────────

class Verdict(str, Enum):
    ACCEPT = "ACCEPT"   # auto-apply in v11.4
    QUEUE  = "QUEUE"    # hold for operator review or more observations
    REJECT = "REJECT"   # never apply — constitutional violation or out of bounds


@dataclass
class GoverningVerdict:
    """
    The result of governing a LearningProposal.

    verdict:        ACCEPT / QUEUE / REJECT
    proposal_id:    links back to the original proposal
    reasons:        why this verdict was reached
    blocked_dims:   dimensions that were constitutionally blocked
    queued_for:     what must happen before QUEUE becomes ACCEPT
    """
    verdict:      Verdict
    proposal_id:  str
    action:       str
    reasons:      list[str]
    blocked_dims: list[str] = field(default_factory=list)
    queued_for:   Optional[str] = None

    def summary(self) -> str:
        return (
            f"{self.verdict.value:6s}  [{self.proposal_id}]  {self.action:22s}  "
            + (f"queued_for: {self.queued_for}" if self.queued_for else
               f"blocked: {self.blocked_dims}" if self.blocked_dims else
               "; ".join(self.reasons))
        )


# ─────────────────────────────────────────────
# Constitutional guard
# ─────────────────────────────────────────────

# Dimensions that learning may NEVER adjust
CONSTITUTIONAL_DIMS: set[str] = set()
# Note: all three dimensions (risk, reversibility, epistemic_uncertainty)
# are learnable in principle — but only within bounds and only via
# ACTION_MODEL[action]["learned_effects"].
# What is constitutional is the PRINCIPLE, not the dimension:
#   - memi_decision() thresholds: not in ACTION_MODEL, never touched
#   - VETO conditions: not in ACTION_MODEL, never touched
#   - epistemic_gating principle: checked via action type
#   - uncertainty_travels: not in ACTION_MODEL, never touched

# Actions whose governance-related effects must never be learned away
PROTECTED_ACTION_TYPES = {"governance"}   # none exist yet; future guard


def is_constitutional_violation(proposal: LearningProposal) -> tuple[bool, list[str]]:
    """
    Check whether a proposal would violate constitutional limits.

    In v11.3, violations are:
    1. Proposal is out of bounds (magnitude exceeds LEARNING_BOUNDS)
    2. Proposed adjustment would effectively zero out an epistemic action's
       uncertainty reduction (would disable epistemic gating in practice)
    3. Action type is protected
    """
    violations = []

    # Check 1: out of bounds
    if not proposal.within_bounds:
        violations.append(
            f"adjustment magnitude {proposal.adjustment_magnitude:.3f} "
            f"exceeds bounds {LEARNING_BOUNDS}"
        )

    # Check 2: would disable epistemic uncertainty reduction for epistemic actions
    if proposal.action in ACTION_MODEL:
        atype = ACTION_MODEL[proposal.action].get("type")
        if atype == "epistemic":
            eu_adj = proposal.proposed_adjustment.get("epistemic_uncertainty", 0.0)
            current_eu_effect = ACTION_MODEL[proposal.action]["effects"].get("epistemic_uncertainty", 0.0)
            # Would the adjustment flip the sign of the eu effect?
            if current_eu_effect < 0 and (current_eu_effect + eu_adj) > 0:
                violations.append(
                    f"proposed adjustment would flip epistemic_uncertainty effect "
                    f"from {current_eu_effect:+.3f} to {current_eu_effect + eu_adj:+.3f} "
                    f"— would disable epistemic function"
                )

    # Check 3: protected action type
    if proposal.action in ACTION_MODEL:
        atype = ACTION_MODEL[proposal.action].get("type")
        if atype in PROTECTED_ACTION_TYPES:
            violations.append(f"action type '{atype}' is constitutionally protected")

    return bool(violations), violations


# ─────────────────────────────────────────────
# govern_learning (NEW in v11.3)
# ─────────────────────────────────────────────

def govern_learning(proposal: LearningProposal) -> GoverningVerdict:
    """
    Evaluate a LearningProposal and return a GoverningVerdict.

    Decision tree:
        1. Constitutional check → REJECT if violated
        2. Out-of-bounds check  → REJECT if violated
        3. Confidence check     → QUEUE if confidence < 0.7
        4. Magnitude check      → QUEUE if magnitude >= 0.05
        5. All pass             → ACCEPT
    """
    reasons: list[str]      = []
    blocked_dims: list[str] = []

    # ── Step 1: Constitutional check ────────────
    violated, violation_reasons = is_constitutional_violation(proposal)
    if violated:
        return GoverningVerdict(
            verdict=Verdict.REJECT,
            proposal_id=proposal.proposal_id,
            action=proposal.action,
            reasons=violation_reasons,
            blocked_dims=[d for d in proposal.proposed_adjustment if abs(proposal.proposed_adjustment[d]) > 0],
        )

    # ── Step 2: Bounds check ─────────────────────
    if not proposal.within_bounds:
        return GoverningVerdict(
            verdict=Verdict.REJECT,
            proposal_id=proposal.proposal_id,
            action=proposal.action,
            reasons=[f"out of bounds: magnitude={proposal.adjustment_magnitude:.3f}"],
        )

    # ── Step 3: Confidence check ─────────────────
    if proposal.confidence < 0.7:
        return GoverningVerdict(
            verdict=Verdict.QUEUE,
            proposal_id=proposal.proposal_id,
            action=proposal.action,
            reasons=[f"confidence={proposal.confidence:.2f} < 0.7"],
            queued_for="await higher-confidence observation (v11.4: multi-observation average)",
        )

    # ── Step 4: Magnitude check ──────────────────
    if proposal.adjustment_magnitude >= 0.05:
        return GoverningVerdict(
            verdict=Verdict.QUEUE,
            proposal_id=proposal.proposal_id,
            action=proposal.action,
            reasons=[f"magnitude={proposal.adjustment_magnitude:.3f} >= 0.05 — requires operator review"],
            queued_for="operator approval",
        )

    # ── Step 5: All checks passed → ACCEPT ───────
    reasons = [
        f"within bounds",
        f"confidence={proposal.confidence:.2f} >= 0.7",
        f"magnitude={proposal.adjustment_magnitude:.3f} < 0.05",
    ]
    return GoverningVerdict(
        verdict=Verdict.ACCEPT,
        proposal_id=proposal.proposal_id,
        action=proposal.action,
        reasons=reasons,
    )


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run(initial_uncertainty: float = 0.6):
    state = WorldState(risk=0.75, reversibility=0.5,
                       epistemic_uncertainty=initial_uncertainty)
    traj  = Trajectory()
    MAX_ITER, mode = 8, "normal"

    proposals: list[LearningProposal]  = []
    verdicts:  list[GoverningVerdict]  = []

    print("\n" + "=" * 72)
    print("M.E.M.I. v11.3 — Govern Learning")
    print("=" * 72)
    print(f"\nInitial state: {state}\n")

    for i in range(MAX_ITER):
        print(f"\n{'─' * 68}")
        print(f"Step {i+1}  |  {state}")

        if traj.constraint_streak >= 3:
            mode = "safe"
        else:
            mode = "normal"

        best, gate_reason = select_action_2step(state, ACTIONS, mode=mode)
        if gate_reason:
            print(f"  ⚠ GATE: {gate_reason}")

        decision, authority = memi_decision(state, best)
        effective = replan(decision, best)

        print(f"  proposed={best:22s}  decision={decision:25s}  auth={authority}")

        if effective is None:
            print("  → Stop: operator required")
            break

        if effective != best:
            print(f"  → Replan: {best} → {effective}")

        state_before = state.copy()
        state        = next_state(state, effective)

        # ── Observe + propose (v11.2) ────────────
        proposal = build_learning_proposal(
            action=effective,
            state_before=state_before,
            state_after=state,
            action_model=ACTION_MODEL,
        )

        if proposal:
            proposals.append(proposal)

            # ── Govern (v11.3) ───────────────────
            verdict = govern_learning(proposal)
            verdicts.append(verdict)

            # Print compact verdict
            v_color = {"ACCEPT": "✓", "QUEUE": "⏸", "REJECT": "✗"}[verdict.verdict.value]
            print(f"\n  {v_color} LEARNING VERDICT  [{proposal.proposal_id}]  "
                  f"{verdict.verdict.value}")
            print(f"    action:     {proposal.action}")
            print(f"    magnitude:  {proposal.adjustment_magnitude:.3f}  "
                  f"confidence: {proposal.confidence:.2f}")
            print(f"    reasons:    {'; '.join(verdict.reasons)}")
            if verdict.queued_for:
                print(f"    queued_for: {verdict.queued_for}")
            if verdict.blocked_dims:
                print(f"    blocked:    {verdict.blocked_dims}")
            print(f"    STATUS:     NOT APPLIED — v11.4 will apply ACCEPT verdicts")

        traj.add({"step": i, "action": effective, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})

    # ── Governance summary ────────────────────────
    print(f"\n{'=' * 72}")
    print("Governance summary")
    print("=" * 72)

    counts = {v: 0 for v in Verdict}
    for v in verdicts:
        counts[v.verdict] += 1

    print(f"  Proposals:  {len(proposals)}")
    print(f"  ACCEPT:     {counts[Verdict.ACCEPT]}   (eligible for model update in v11.4)")
    print(f"  QUEUE:      {counts[Verdict.QUEUE]}   (held — confidence or magnitude)")
    print(f"  REJECT:     {counts[Verdict.REJECT]}   (constitutional violation or out of bounds)")
    print()

    for v in verdicts:
        print(f"  {v.summary()}")

    print(f"\n  None applied. Model update is v11.4.")
    print(f"  Final state: {state}")


if __name__ == "__main__":
    run(initial_uncertainty=0.6)
