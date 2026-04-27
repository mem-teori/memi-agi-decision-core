"""
M.E.M.I. v11.6 — Admissibility Layer
======================================

Builds on v11.5 (Multi-Observation Learning).

New in v11.6
------------
Admissibility is now a first-class condition evaluated before foresight scores.
The system only evaluates futures it is allowed to enter.

One addition:
    is_admissible(action, state)   — encodes all three admissibility conditions
                                     type condition + effect condition + governance pre-check

Changed:
    select_action_learned()        — admissibility pre-filter before scoring
    best_single_step()             — same filter for follow-up selection

Unchanged:
    memi_decision()                — still runs after selection (constitutional)
    epistemic_gating()             — preserved as additional guard
    ACTION_MODEL                   — unchanged
    all v11.2–v11.5 learning layers

Three tensions resolved
------------------------
1. Epistemic loop:
   Epistemic actions are unconditionally admissible.
   The system can always act to learn.
   Only operative/irreversible actions require epistemic sufficiency.

2. Inadmissible futures:
   score = -inf for inadmissible actions.
   Foresight evaluation space = governance permission space.

3. Governance after selection:
   is_admissible() is evaluation-time pre-filter (fast, approximate).
   memi_decision() is execution-time constitutional gate (always runs).

Architectural invariant (updated)
-----------------------------------
    Foresight selects the proposal.
    Foresight only evaluates admissible proposals.
    Effect model explains why.
    Epistemic state qualifies confidence.
    Governance decides whether it is permitted.
    Decision payload carries uncertainty beyond the system.
    Learning proposals are governed decisions — not automatic updates.

Bærende sætninger
-----------------
    The system does not need knowledge to act.
    It needs knowledge to act irreversibly.

    The system must not evaluate futures it is not allowed to enter.

    The system should not strongly prefer what it is not allowed to do.
"""

from __future__ import annotations

import math
import uuid
import random
from dataclasses import dataclass, field
from typing import Optional

from memi_v112 import (
    WorldState, ACTIONS, EPISTEMIC_GATE,
    UNCERTAINTY_PENALTY_WEIGHT, LEARNING_BOUNDS,
    LearningProposal,
    next_state, epistemic_gating,
    memi_decision, replan, evaluate_projected,
    semantic_filter,
    observe_outcome, build_learning_proposal,
    Trajectory,
)
from memi_v113 import GoverningVerdict, Verdict, govern_learning
from memi_v114 import (
    LearnedModel, LearningLogEntry, apply_learning,
    BASE_ACTION_MODEL,
    simulate_from_learned,
)
from memi_v115 import (
    LearningBuffer, BufferStore,
    flush_buffer,
    MIN_N, VAR_THRESHOLD, CONF_THRESHOLD,
)


# ─────────────────────────────────────────────
# Admissibility thresholds
# ─────────────────────────────────────────────

# Operative actions blocked above this uncertainty level
OPERATIVE_EU_LIMIT    = EPISTEMIC_GATE          # 0.7

# Irreversible actions blocked above this uncertainty level
IRREVERSIBLE_EU_LIMIT = 0.5

# Irreversible actions require minimum reversibility
IRREVERSIBLE_REV_MIN  = 0.4

# Irreversible actions require minimum risk (not justified when risk already low)
IRREVERSIBLE_RISK_MIN = 0.3

# Governance pre-check: approximate VETO_2 boundary
VETO2_RISK_LIMIT      = 0.9
VETO2_REV_LIMIT       = 0.2


# ─────────────────────────────────────────────
# is_admissible (NEW in v11.6)
# ─────────────────────────────────────────────

@dataclass
class AdmissibilityResult:
    admissible: bool
    action:     str
    reason:     str


def is_admissible(action: str, state: WorldState) -> AdmissibilityResult:
    """
    First-class admissibility check — runs before foresight scoring.

    Three conditions (all must hold):
    ─────────────────────────────────
    1. Type condition
       epistemic:    always admissible — breaks epistemic loop
       operative:    admissible if epistemic_uncertainty <= OPERATIVE_EU_LIMIT
       irreversible: admissible if eu <= IRREVERSIBLE_EU_LIMIT
                     AND reversibility > IRREVERSIBLE_REV_MIN
                     AND risk > IRREVERSIBLE_RISK_MIN

    2. Effect condition
       avoid_when(state) must be False

    3. Governance pre-check (heuristic — not the constitutional gate)
       VETO_2 would not fire: NOT (risk > 0.9 AND rev < 0.2)
       Note: memi_decision() still runs after selection regardless.

    Returns AdmissibilityResult with reason for audit/explanation.
    """
    if action not in BASE_ACTION_MODEL:
        return AdmissibilityResult(
            admissible=False, action=action,
            reason=f"unknown action '{action}'"
        )

    model = BASE_ACTION_MODEL[action]
    atype = model.get("type", "unknown")

    # ── Condition 1: Type condition ──────────────────────
    if atype == "epistemic":
        # Always admissible — epistemic actions break the knowledge loop
        pass   # no type-based block

    elif atype == "operative":
        if state.epistemic_uncertainty > OPERATIVE_EU_LIMIT:
            return AdmissibilityResult(
                admissible=False, action=action,
                reason=(
                    f"operative action blocked: "
                    f"eu={state.epistemic_uncertainty:.2f} > "
                    f"limit={OPERATIVE_EU_LIMIT} — insufficient knowledge"
                ),
            )

    elif atype == "irreversible":
        if state.epistemic_uncertainty > IRREVERSIBLE_EU_LIMIT:
            return AdmissibilityResult(
                admissible=False, action=action,
                reason=(
                    f"irreversible action blocked: "
                    f"eu={state.epistemic_uncertainty:.2f} > "
                    f"limit={IRREVERSIBLE_EU_LIMIT}"
                ),
            )
        if state.reversibility <= IRREVERSIBLE_REV_MIN:
            return AdmissibilityResult(
                admissible=False, action=action,
                reason=(
                    f"irreversible action blocked: "
                    f"reversibility={state.reversibility:.2f} <= "
                    f"min={IRREVERSIBLE_REV_MIN} — already low, further loss not acceptable"
                ),
            )
        if state.risk <= IRREVERSIBLE_RISK_MIN:
            return AdmissibilityResult(
                admissible=False, action=action,
                reason=(
                    f"irreversible action not justified: "
                    f"risk={state.risk:.2f} <= {IRREVERSIBLE_RISK_MIN} — "
                    f"cost not justified"
                ),
            )

    # ── Condition 2: Effect condition ────────────────────
    if model.get("avoid_when", lambda s: False)(state):
        return AdmissibilityResult(
            admissible=False, action=action,
            reason=f"avoid_when condition active in current state"
        )

    # ── Condition 3: Governance pre-check ────────────────
    if state.risk > VETO2_RISK_LIMIT and state.reversibility < VETO2_REV_LIMIT:
        return AdmissibilityResult(
            admissible=False, action=action,
            reason=(
                f"governance pre-check: VETO_2 would fire "
                f"(risk={state.risk:.2f}, rev={state.reversibility:.2f})"
            ),
        )

    return AdmissibilityResult(
        admissible=True, action=action, reason="admissible"
    )


# ─────────────────────────────────────────────
# Updated foresight with admissibility (CHANGED in v11.6)
# ─────────────────────────────────────────────

DISCOUNT = 0.7
NEG_INF  = float("-inf")


def best_single_step_admissible(
    state:     WorldState,
    candidates:list[str],
    penalties: dict[str, float],
    model:     LearnedModel,
) -> tuple[str, float]:
    """best_single_step with admissibility pre-filter."""
    best, best_score = candidates[0], NEG_INF
    for action in candidates:
        if not is_admissible(action, state).admissible:
            continue
        projected = simulate_from_learned(state, action, model)
        score     = evaluate_projected(projected, state) - penalties.get(action, 0.0)
        if score > best_score:
            best_score, best = score, action
    return best, best_score


def select_action_admissible(
    state:      WorldState,
    model:      LearnedModel,
    candidates: list[str],
    mode:       str = "normal",
    discount:   float = DISCOUNT,
) -> tuple[str, Optional[str], dict[str, AdmissibilityResult]]:
    """
    2-step foresight with admissibility pre-filter (v11.6).

    Returns (best_action, gate_reason, admissibility_map).

    Key change from v11.5:
        - is_admissible() runs before scoring each candidate
        - inadmissible actions receive score = -inf
        - evaluation space is now aligned with permission space
    """
    gated, gate_reason = epistemic_gating(state, candidates)
    filter_results     = semantic_filter(gated, state, mode)
    penalties          = {a: p for a, p, _ in filter_results}

    admissibility_map: dict[str, AdmissibilityResult] = {}
    scores: dict[str, float] = {}

    for action in gated:
        adm = is_admissible(action, state)
        admissibility_map[action] = adm

        if not adm.admissible:
            scores[action] = NEG_INF
            continue

        s1        = simulate_from_learned(state, action, model)
        immediate = evaluate_projected(s1, state) - penalties.get(action, 0.0)

        followup, _ = best_single_step_admissible(s1, gated, penalties, model)
        s2          = simulate_from_learned(s1, followup, model)
        future      = evaluate_projected(s2, s1)

        scores[action] = immediate + discount * future

    # Select best admissible action
    admissible_scores = {a: s for a, s in scores.items() if s > NEG_INF}
    if admissible_scores:
        best = max(admissible_scores, key=admissible_scores.get)
    else:
        # All actions inadmissible — fall back to safest epistemic action
        best = next(
            (a for a in gated if BASE_ACTION_MODEL.get(a, {}).get("type") == "epistemic"),
            gated[0]
        )

    return best, gate_reason, admissibility_map


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run(initial_uncertainty: float = 0.6, max_iter: int = 12):
    state   = WorldState(risk=0.75, reversibility=0.5,
                         epistemic_uncertainty=initial_uncertainty)
    model   = LearnedModel(BASE_ACTION_MODEL)
    buffers = BufferStore()
    traj    = Trajectory()
    mode    = "normal"

    proposals_generated = 0
    flushed_count       = 0
    applied_count       = 0
    admissibility_blocks= 0

    print("\n" + "=" * 72)
    print("M.E.M.I. v11.6 — Admissibility Layer")
    print("=" * 72)
    print(f"\nInitial state: {state}\n")

    for i in range(max_iter):
        print(f"\n{'─' * 68}")
        print(f"Step {i+1}  |  {state}")

        if traj.constraint_streak >= 3:
            mode = "safe"
        else:
            mode = "normal"

        best, gate_reason, adm_map = select_action_admissible(
            state, model, ACTIONS, mode=mode
        )

        if gate_reason:
            print(f"  ⚠ GATE: {gate_reason}")

        # Print admissibility map (only non-admissible entries + selected)
        blocked = [(a, r.reason) for a, r in adm_map.items() if not r.admissible]
        if blocked:
            admissibility_blocks += len(blocked)
            for a, reason in blocked:
                print(f"  ✗ inadmissible: {a:22s}  {reason}")

        decision, authority = memi_decision(state, best)
        effective = replan(decision, best)

        print(f"  ✓ selected:    {best:22s}  decision={decision:24s}  auth={authority}")

        if effective is None:
            print("  → Stop: operator required")
            break

        if effective != best:
            print(f"  → Replan: {best} → {effective}")

        state_before = state.copy()
        state        = next_state(state, effective)

        # ── Learning pipeline (v11.2–v11.5, unchanged) ──
        proposal = build_learning_proposal(
            action=effective,
            state_before=state_before,
            state_after=state,
            action_model=BASE_ACTION_MODEL,
        )
        if proposal:
            proposals_generated += 1
            buffers.add(proposal)
            buf      = buffers.get(effective)
            flush_ok, flush_reason = buf.should_flush()
            if flush_ok:
                entry, verdict, _ = flush_buffer(buf, model)
                flushed_count += 1
                if entry:
                    applied_count += 1
                    print(f"  ✎ LEARN [{entry.entry_id}]  "
                          f"adj={ {k:round(v,4) for k,v in entry.adjustment.items() if abs(v)>0.001} }")

        traj.add({"step": i, "action": effective, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})

    # ── Summary ──────────────────────────────────
    print(f"\n{'=' * 72}")
    print("Summary")
    print("=" * 72)
    print(f"  Admissibility blocks: {admissibility_blocks}")
    print(f"  Learning proposals:   {proposals_generated}")
    print(f"  Flushes / Applied:    {flushed_count} / {applied_count}")

    if model.learned_summary():
        print(f"\n  Learned adjustments:")
        for action, data in model.learned_summary().items():
            print(f"  {action}")
            for dim in ("risk", "reversibility", "epistemic_uncertainty"):
                base    = data["base"].get(dim, 0.0)
                learned = data["learned"].get(dim, 0.0)
                eff     = data["effective"].get(dim, 0.0)
                if abs(learned) > 0.001:
                    print(f"    {dim:26s}  base={base:+.4f}  "
                          f"learned={learned:+.4f}  effective={eff:+.4f}")

    print(f"\n  Final state: {state}")

    # ── Contrast test ─────────────────────────────
    print(f"\n{'─' * 68}")
    print("Contrast: same scenario with high initial uncertainty (eu=0.85)")
    print("Expected: irreversible action blocked from evaluation")
    print('─' * 68)

    state_hi = WorldState(risk=0.75, reversibility=0.5, epistemic_uncertainty=0.85)
    _, _, adm_hi = select_action_admissible(state_hi, LearnedModel(BASE_ACTION_MODEL), ACTIONS)
    for a, r in adm_hi.items():
        sym = "✓" if r.admissible else "✗"
        print(f"  {sym} {a:22s}  {r.reason}")


if __name__ == "__main__":
    run(initial_uncertainty=0.6)
