"""
M.E.M.I. v11.2 — Observe Outcome + Build Learning Proposal
============================================================

Builds on v11.1 (Decision Payload / Uncertainty Propagation).

New in v11.2
------------
The system measures what actually happened versus what it predicted.
It builds a structured learning proposal from that comparison.

It does NOT update the model yet.
It does NOT apply any adjustment.
It only observes and proposes.

Two additions:
    observe_outcome()          — measures actual effect of an action
    build_learning_proposal()  — constructs a LearningProposal from the delta

One new dataclass:
    LearningProposal           — structured proposal (not yet applied)

Unchanged:
    ACTION_MODEL               — not modified
    memi_decision()            — not modified
    epistemic_gating()         — not modified
    WorldState                 — not modified
    Trajectory, evaluate_trajectory

Architectural invariant (preserved)
-------------------------------------
    Foresight selects the proposal.
    Effect model explains why.
    Epistemic state qualifies confidence.
    Governance decides whether it is permitted.
    Decision payload carries uncertainty beyond the system.
    Learning proposals are governed decisions — not automatic updates.

Bærende sætning
---------------
    The system does not learn yet.
    It learns how to propose learning.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# State (unchanged from v11.1)
# ─────────────────────────────────────────────

@dataclass
class WorldState:
    risk:                  float
    reversibility:         float
    epistemic_uncertainty: float = 0.5

    def copy(self) -> "WorldState":
        return WorldState(
            risk=self.risk,
            reversibility=self.reversibility,
            epistemic_uncertainty=self.epistemic_uncertainty,
        )

    def certainty(self) -> str:
        u = self.epistemic_uncertainty
        if u <= 0.2:  return "high"
        if u <= 0.5:  return "moderate"
        if u <= 0.8:  return "low"
        return "unknown"

    def __str__(self) -> str:
        return (f"risk={self.risk:.3f}  "
                f"rev={self.reversibility:.3f}  "
                f"eu={self.epistemic_uncertainty:.3f} [{self.certainty()}]")

    def as_effects(self) -> dict:
        return {
            "risk":                  self.risk,
            "reversibility":         self.reversibility,
            "epistemic_uncertainty": self.epistemic_uncertainty,
        }


EPISTEMIC_GATE             = 0.7
UNCERTAINTY_PENALTY_WEIGHT = 0.4

# Learning bounds — no single proposal may exceed these
LEARNING_BOUNDS = {
    "risk":                  0.08,   # max adjustment to risk effect
    "reversibility":         0.06,
    "epistemic_uncertainty": 0.05,
}

# Minimum observations before a proposal is considered reliable
MIN_OBSERVATIONS_FOR_PROPOSAL = 1   # v11.2: single observation; v11.4 will require N


# ─────────────────────────────────────────────
# LearningProposal (NEW in v11.2)
# ─────────────────────────────────────────────

@dataclass
class LearningProposal:
    """
    A structured proposal to adjust the effect model for one action.

    This is NOT an update. It is a proposal.
    Governance decides whether it is applied (v11.3).

    Fields
    ------
    proposal_id           : unique identifier for audit
    action                : which action this concerns
    expected_effect       : what ACTION_MODEL predicted
    observed_effect       : what actually happened (from state delta)
    delta                 : observed - expected (per dimension)
    confidence            : how reliable is this single observation?
    proposed_adjustment   : the bounded adjustment being proposed
    adjustment_magnitude  : max(|delta|) — size of the proposed change
    within_bounds         : whether all proposed adjustments are within LEARNING_BOUNDS
    requires_approval     : True if magnitude >= 0.05 or confidence < 0.7
    justification         : human-readable explanation of the proposal
    applied               : False until governance approves and applies
    """
    proposal_id:          str
    action:               str
    expected_effect:      dict
    observed_effect:      dict
    delta:                dict
    confidence:           float
    proposed_adjustment:  dict
    adjustment_magnitude: float
    within_bounds:        bool
    requires_approval:    bool
    justification:        str
    applied:              bool = False

    def summary(self) -> str:
        status = "within bounds" if self.within_bounds else "OUT OF BOUNDS"
        approval = "auto-apply candidate" if not self.requires_approval else "requires approval"
        return (
            f"LearningProposal({self.action})  "
            f"magnitude={self.adjustment_magnitude:.3f}  "
            f"confidence={self.confidence:.2f}  "
            f"[{status}]  [{approval}]"
        )


# ─────────────────────────────────────────────
# observe_outcome (NEW in v11.2)
# ─────────────────────────────────────────────

def observe_outcome(
    state_before: WorldState,
    state_after:  WorldState,
    action:       str,
) -> dict:
    """
    Measure the actual effect of an action by comparing state_before
    and state_after on all three dimensions.

    Returns the observed delta per dimension.

    Note: state_after includes stochastic drift from next_state().
    The observed effect is therefore noisy — a single observation
    is not sufficient for a high-confidence proposal.
    v11.4 will introduce multi-observation averaging.
    """
    return {
        "risk": round(
            state_after.risk - state_before.risk, 4
        ),
        "reversibility": round(
            state_after.reversibility - state_before.reversibility, 4
        ),
        "epistemic_uncertainty": round(
            state_after.epistemic_uncertainty - state_before.epistemic_uncertainty, 4
        ),
    }


# ─────────────────────────────────────────────
# build_learning_proposal (NEW in v11.2)
# ─────────────────────────────────────────────

def build_learning_proposal(
    action:          str,
    state_before:    WorldState,
    state_after:     WorldState,
    action_model:    dict,
) -> Optional[LearningProposal]:
    """
    Build a LearningProposal by comparing expected vs observed effects.

    Steps:
        1. Get expected effect from ACTION_MODEL
        2. Observe actual effect from state transition
        3. Compute delta per dimension
        4. Propose bounded adjustment (clipped to LEARNING_BOUNDS)
        5. Compute confidence (inverse of state uncertainty at time of action)
        6. Determine whether proposal is within bounds and requires approval

    Returns None if the action is not in the model or delta is negligible.
    """
    if action not in action_model:
        return None

    expected = action_model[action]["effects"]
    observed = observe_outcome(state_before, state_after, action)

    # Delta: how far off was the prediction?
    delta = {
        dim: round(observed.get(dim, 0.0) - expected.get(dim, 0.0), 4)
        for dim in ("risk", "reversibility", "epistemic_uncertainty")
    }

    # Skip negligible deltas (noise floor)
    max_delta = max(abs(v) for v in delta.values())
    if max_delta < 0.01:
        return None

    # Proposed adjustment: clip to LEARNING_BOUNDS
    proposed = {}
    within_bounds = True
    for dim, d in delta.items():
        bound = LEARNING_BOUNDS.get(dim, 0.05)
        clipped = max(-bound, min(bound, d * 0.5))   # propose half the delta
        proposed[dim] = round(clipped, 4)
        if abs(d * 0.5) > bound:
            within_bounds = False

    adjustment_magnitude = max(abs(v) for v in proposed.values())

    # Confidence: higher when system was more certain at time of action
    # Low epistemic_uncertainty at decision time = more reliable observation
    confidence = round(
        max(0.1, 1.0 - state_before.epistemic_uncertainty * 0.8), 3
    )

    requires_approval = (
        adjustment_magnitude >= 0.05
        or confidence < 0.7
        or not within_bounds
    )

    # Justification
    changed_dims = [d for d, v in delta.items() if abs(v) >= 0.01]
    justification = (
        f"Observed {action}: "
        + ", ".join(
            f"{d} expected {expected.get(d, 0):+.3f} got {observed.get(d, 0):+.3f} "
            f"(Δ{delta[d]:+.3f})"
            for d in changed_dims
        )
        + f". Confidence={confidence:.2f} (eu={state_before.epistemic_uncertainty:.2f} at decision)."
    )

    return LearningProposal(
        proposal_id=str(uuid.uuid4())[:8],
        action=action,
        expected_effect={k: expected.get(k, 0.0) for k in ("risk", "reversibility", "epistemic_uncertainty")},
        observed_effect={k: observed.get(k, 0.0) for k in ("risk", "reversibility", "epistemic_uncertainty")},
        delta=delta,
        confidence=confidence,
        proposed_adjustment=proposed,
        adjustment_magnitude=round(adjustment_magnitude, 4),
        within_bounds=within_bounds,
        requires_approval=requires_approval,
        justification=justification,
    )


# ─────────────────────────────────────────────
# ACTION MODEL (unchanged from v11.1)
# ─────────────────────────────────────────────

ACTION_MODEL: dict[str, dict] = {
    "collect_telemetry": {
        "type":             "epistemic",
        "primary_effect":   "reduce_uncertainty",
        "effects": {
            "risk":                  -0.12,
            "reversibility":         +0.12,
            "epistemic_uncertainty": -0.15,
        },
        "effective_when":           lambda s: s.risk > 0.3 or s.epistemic_uncertainty > 0.4,
        "diminishing_returns_when": lambda s: s.reversibility >= 0.95 and s.epistemic_uncertainty < 0.2,
        "avoid_when":               lambda s: False,
    },
    "monitor": {
        "type":             "epistemic",
        "primary_effect":   "reduce_uncertainty",
        "effects": {
            "risk":                  -0.04,
            "reversibility":         +0.04,
            "epistemic_uncertainty": -0.05,
        },
        "effective_when":           lambda s: True,
        "diminishing_returns_when": lambda s: s.risk < 0.1 and s.epistemic_uncertainty < 0.15,
        "avoid_when":               lambda s: False,
    },
    "reduce_load": {
        "type":             "operative",
        "primary_effect":   "reduce_risk",
        "effects": {
            "risk":                  -0.08,
            "reversibility":          0.00,
            "epistemic_uncertainty":  0.00,
        },
        "effective_when":           lambda s: s.risk > 0.2 and s.epistemic_uncertainty < EPISTEMIC_GATE,
        "diminishing_returns_when": lambda s: s.risk < 0.1,
        "avoid_when":               lambda s: s.epistemic_uncertainty > EPISTEMIC_GATE,
    },
    "isolate_process": {
        "type":             "irreversible",
        "primary_effect":   "reduce_risk",
        "effects": {
            "risk":                  -0.15,
            "reversibility":         -0.10,
            "epistemic_uncertainty":  0.00,
        },
        "effective_when":           lambda s: s.risk > 0.5 and s.reversibility > 0.4 and s.epistemic_uncertainty < 0.5,
        "diminishing_returns_when": lambda s: s.risk < 0.2,
        "avoid_when":               lambda s: s.risk < 0.3 or s.epistemic_uncertainty > 0.5,
    },
}

ACTIONS = list(ACTION_MODEL.keys())


# ─────────────────────────────────────────────
# Core execution (unchanged from v11.1)
# ─────────────────────────────────────────────

def simulate_from_model(state: WorldState, action: str) -> WorldState:
    if action not in ACTION_MODEL:
        return state.copy()
    effects = ACTION_MODEL[action]["effects"]
    s = state.copy()
    s.risk                  = max(0.0, min(1.0, s.risk                  + effects.get("risk", 0.0)))
    s.reversibility         = max(0.0, min(1.0, s.reversibility         + effects.get("reversibility", 0.0)))
    s.epistemic_uncertainty = max(0.0, min(1.0, s.epistemic_uncertainty + effects.get("epistemic_uncertainty", 0.0)))
    return s


def next_state(state: WorldState, action: Optional[str] = None) -> WorldState:
    s = simulate_from_model(state, action) if action else state.copy()
    s.risk                  = max(0.0, min(1.0, s.risk          + random.uniform(-0.08,  0.04)))
    s.reversibility         = max(0.0, min(1.0, s.reversibility + random.uniform(-0.04,  0.04)))
    s.epistemic_uncertainty = max(0.0, min(1.0, s.epistemic_uncertainty + random.uniform(-0.02, 0.05)))
    return s


def epistemic_gating(state, candidates):
    if state.epistemic_uncertainty <= EPISTEMIC_GATE:
        return candidates, None
    epistemic = [a for a in candidates if ACTION_MODEL.get(a, {}).get("type") == "epistemic"]
    if not epistemic:
        return candidates, None
    return epistemic, f"eu={state.epistemic_uncertainty:.2f} > gate={EPISTEMIC_GATE}"


def memi_decision(state: WorldState, action: str) -> tuple[str, str]:
    if state.risk > 0.9 and state.reversibility < 0.2: return "VETO_2", "NONE"
    if state.risk > 0.8:  return "VETO_1", "LOW"
    if state.risk > 0.6:  return "HELD",   "LOW"
    if state.risk > 0.4:  return "ALLOWED_WITH_CAUTION", "MEDIUM"
    return "ALLOWED", "HIGH"


def replan(decision: str, action: str) -> Optional[str]:
    if decision == "HELD":    return "collect_telemetry"
    if decision == "VETO_1":  return "monitor"
    if decision in ("DEFERRED", "VETO_2"): return None
    return action


def evaluate_projected(projected: WorldState, current: WorldState) -> float:
    risk_gain = (current.risk           - projected.risk)           * 1.0
    rev_gain  = (projected.reversibility - current.reversibility)   * 0.5
    rev_loss  = -(current.reversibility  - projected.reversibility) * 0.8 \
                if projected.reversibility < current.reversibility else 0.0
    eu_gain   = (current.epistemic_uncertainty - projected.epistemic_uncertainty) * UNCERTAINTY_PENALTY_WEIGHT
    return risk_gain + rev_gain + rev_loss + eu_gain


def semantic_filter(candidates, state, mode="normal"):
    results = []
    for action in candidates:
        if action not in ACTION_MODEL:
            results.append((action, 0.0, "")); continue
        m = ACTION_MODEL[action]
        penalty, reasons = 0.0, []
        if m["avoid_when"](state):            penalty += 0.25; reasons.append("avoid_when")
        if m["diminishing_returns_when"](state): penalty += 0.10; reasons.append("diminishing")
        if mode == "safe" and m["type"] == "irreversible": penalty += 0.30; reasons.append("safe_mode")
        if not m["effective_when"](state):    penalty += 0.05; reasons.append("not_effective")
        results.append((action, penalty, "; ".join(reasons)))
    return results


DISCOUNT = 0.7

def best_single_step(state, candidates, penalties):
    best, best_score = candidates[0], float("-inf")
    for action in candidates:
        score = evaluate_projected(simulate_from_model(state, action), state) - penalties.get(action, 0.0)
        if score > best_score:
            best_score, best = score, action
    return best, best_score


def select_action_2step(state, candidates, mode="normal"):
    gated, gate_reason = epistemic_gating(state, candidates)
    filter_results     = semantic_filter(gated, state, mode)
    penalties          = {a: p for a, p, _ in filter_results}
    scores = {}
    for action in gated:
        s1        = simulate_from_model(state, action)
        immediate = evaluate_projected(s1, state) - penalties.get(action, 0.0)
        followup, _ = best_single_step(s1, gated, penalties)
        s2        = simulate_from_model(s1, followup)
        future    = evaluate_projected(s2, s1)
        scores[action] = immediate + DISCOUNT * future
    best = max(scores, key=scores.get)
    return best, gate_reason


# ─────────────────────────────────────────────
# Trajectory (unchanged)
# ─────────────────────────────────────────────

@dataclass
class Trajectory:
    steps:             list  = field(default_factory=list)
    proposals:         list  = field(default_factory=list)   # NEW — learning proposals
    operator_load:     int   = 0
    continuity_cost:   float = 0.0
    constraint_streak: int   = 0

    def add(self, step: dict) -> None:
        self.steps.append(step)
        if step["decision"] == "DEFERRED":
            self.operator_load += 1
        if step["decision"] in ["HELD", "VETO_1", "VETO_2"]:
            self.continuity_cost   += 0.03 if step.get("mode") == "safe" else 0.1
            self.constraint_streak += 1
        else:
            self.constraint_streak = 0

    def add_proposal(self, proposal: LearningProposal) -> None:
        self.proposals.append(proposal)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run(initial_uncertainty: float = 0.6):
    state = WorldState(risk=0.75, reversibility=0.5,
                       epistemic_uncertainty=initial_uncertainty)
    traj  = Trajectory()
    MAX_ITER, mode = 8, "normal"

    print("\n" + "=" * 72)
    print("M.E.M.I. v11.2 — Observe Outcome + Build Learning Proposal")
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

        # ── Observe + propose learning (v11.2) ──
        proposal = build_learning_proposal(
            action=effective,
            state_before=state_before,
            state_after=state,
            action_model=ACTION_MODEL,
        )

        if proposal:
            traj.add_proposal(proposal)
            print(f"\n  LEARNING PROPOSAL  [{proposal.proposal_id}]")
            print(f"    {proposal.summary()}")
            print(f"    {proposal.justification}")
            for dim, val in proposal.proposed_adjustment.items():
                if abs(val) >= 0.001:
                    exp = proposal.expected_effect.get(dim, 0)
                    obs = proposal.observed_effect.get(dim, 0)
                    print(f"    {dim:26s}  expected={exp:+.3f}  observed={obs:+.3f}  "
                          f"proposed_adj={val:+.4f}")
            print(f"    STATUS: {'NOT APPLIED — awaiting governance (v11.3)' }")
        else:
            print(f"  (no learning proposal — delta negligible)")

        traj.add({"step": i, "action": effective, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})

    # ── Learning proposal summary ─────────────
    print(f"\n{'=' * 72}")
    print(f"Learning proposals generated: {len(traj.proposals)}")
    print("=" * 72)
    for p in traj.proposals:
        approval = "auto-apply candidate" if not p.requires_approval else "requires approval"
        print(f"  [{p.proposal_id}]  {p.action:22s}  "
              f"magnitude={p.adjustment_magnitude:.3f}  "
              f"confidence={p.confidence:.2f}  "
              f"{'within bounds' if p.within_bounds else 'OUT OF BOUNDS':15s}  "
              f"{approval}")
    print()
    print("  None applied. Governance check is v11.3.")
    print(f"\n  Final state: {state}")


if __name__ == "__main__":
    run(initial_uncertainty=0.6)
