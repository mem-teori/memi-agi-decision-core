"""
M.E.M.I. v12.0 — Contextual Admissibility
===========================================

Builds on v11.6 (Admissibility Layer).

New in v12.0
------------
Admissibility thresholds and evaluation weights are now context-qualified.
The same epistemic uncertainty that permits action in security
may not permit it in clinical care.

Two additions:
    Context               — domain-specific thresholds, weights, semantics
    CONTEXTS              — three built-in contexts: security, clinical, industrial

Two changes:
    is_admissible()       — takes Context, uses context thresholds
    evaluate_projected()  — takes Context, uses context weights

Unchanged:
    memi_decision()       — governance not affected
    VETO_1 / VETO_2       — constitutional, context-independent
    uncertainty_travels   — invariant
    LearningBuffer        — unchanged (context-scoped learning is v12.1)
    foresight structure   — unchanged

Architectural invariant (updated)
-----------------------------------
    Foresight selects the proposal.
    Foresight only evaluates admissible proposals.
    Admissibility is qualified by context.
    Effect model explains why.
    Epistemic state qualifies confidence.
    Governance decides whether it is permitted.
    Decision payload carries uncertainty beyond the system.
    Learning proposals are governed decisions — not automatic updates.
    The system learns per context — not a single global model.

Bærende sætning
---------------
    The system does not learn a single model of the world.
    It learns how the world changes across contexts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from memi_v112 import (
    WorldState, ACTIONS, EPISTEMIC_GATE,
    UNCERTAINTY_PENALTY_WEIGHT, LEARNING_BOUNDS,
    LearningProposal,
    next_state, epistemic_gating,
    memi_decision, replan,
    semantic_filter,
    observe_outcome, build_learning_proposal,
    Trajectory,
)
from memi_v113 import GoverningVerdict, Verdict, govern_learning
from memi_v114 import (
    LearnedModel, LearningLogEntry, apply_learning,
    BASE_ACTION_MODEL, simulate_from_learned,
)
from memi_v115 import BufferStore, flush_buffer
from memi_v116 import AdmissibilityResult


# ─────────────────────────────────────────────
# Context (NEW in v12.0)
# ─────────────────────────────────────────────

@dataclass
class Context:
    """
    Domain-specific interpretation of state dimensions and admissibility.

    Context qualifies:
        admissibility thresholds    (irreversible_eu_limit, operative_eu_limit)
        evaluation weights          (risk_weight, reversibility_weight, eu_weight)
        action semantics            (what each action means in this domain)

    Context cannot touch:
        VETO_1 / VETO_2             constitutional, domain-independent
        governance authority        not contextual
        uncertainty_travels         invariant
        base learning bounds        not contextual
    """
    domain:               str
    description:          str

    # Admissibility thresholds
    risk_tolerance:       float   # 0.0 = zero tolerance, 1.0 = high tolerance
    irreversible_eu_limit:float   # max eu for irreversible actions
    operative_eu_limit:   float   # max eu for operative actions
    irreversible_rev_min: float   # min reversibility for irreversible actions
    irreversible_risk_min:float   # min risk to justify irreversible action

    # Evaluation weights
    risk_weight:          float   # how much risk reduction matters
    reversibility_weight: float   # how much reversibility gain/loss matters
    eu_weight:            float   # how much uncertainty reduction matters

    # Semantic labels
    action_semantics:     dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Built-in contexts
# ─────────────────────────────────────────────

CONTEXTS: dict[str, Context] = {

    "security": Context(
        domain="security",
        description=(
            "Adversarial detection and response. "
            "Act before confirmed breach, but avoid irreversible isolation without certainty."
        ),
        risk_tolerance=0.40,
        irreversible_eu_limit=0.40,   # tighter than global default (0.50)
        operative_eu_limit=0.65,
        irreversible_rev_min=0.35,
        irreversible_risk_min=0.35,
        risk_weight=1.2,              # risk reduction slightly more valuable
        reversibility_weight=0.55,
        eu_weight=0.35,
        action_semantics={
            "collect_telemetry": "gather network/process evidence — always safe",
            "monitor":           "observe without intervention",
            "reduce_load":       "throttle suspicious process — reversible",
            "isolate_process":   "terminate and quarantine — hard to undo in production",
        },
    ),

    "clinical": Context(
        domain="clinical",
        description=(
            "Patient safety is primary. "
            "Irreversible interventions require high certainty. "
            "Reversibility is near-absolute requirement."
        ),
        risk_tolerance=0.20,          # very low — patient safety
        irreversible_eu_limit=0.20,   # very tight
        operative_eu_limit=0.45,
        irreversible_rev_min=0.55,    # high — must be confident before irreversible act
        irreversible_risk_min=0.40,   # higher threshold — more risk must be present
        risk_weight=1.8,              # risk reduction highly valuable
        reversibility_weight=1.2,     # reversibility very important
        eu_weight=0.50,
        action_semantics={
            "collect_telemetry": "diagnostic data collection — always safe",
            "monitor":           "observation without intervention",
            "reduce_load":       "reduce stimulus / lower dose — generally reversible",
            "isolate_process":   "intervention — may not be undoable",
        },
    ),

    "industrial": Context(
        domain="industrial",
        description=(
            "Process control. "
            "Disruption is costly but usually recoverable. "
            "Balanced risk and reversibility."
        ),
        risk_tolerance=0.50,
        irreversible_eu_limit=0.45,
        operative_eu_limit=0.65,
        irreversible_rev_min=0.40,
        irreversible_risk_min=0.30,
        risk_weight=1.0,
        reversibility_weight=0.70,
        eu_weight=0.40,
        action_semantics={
            "collect_telemetry": "sensor reading / diagnostic cycle",
            "monitor":           "observe process state",
            "reduce_load":       "reduce throughput / pressure",
            "isolate_process":   "shutdown — significant operational cost",
        },
    ),
}

DEFAULT_CONTEXT = CONTEXTS["industrial"]


# ─────────────────────────────────────────────
# Context-qualified admissibility (CHANGED in v12.0)
# ─────────────────────────────────────────────

def is_admissible(
    action:  str,
    state:   WorldState,
    context: Context = DEFAULT_CONTEXT,
) -> AdmissibilityResult:
    """
    Context-qualified admissibility check.

    Uses context thresholds instead of global constants.
    The same action may be admissible in one context and not in another.

    Constitutional checks (VETO pre-check) remain context-independent.
    """
    if action not in BASE_ACTION_MODEL:
        return AdmissibilityResult(
            admissible=False, action=action,
            reason=f"unknown action '{action}'"
        )

    model = BASE_ACTION_MODEL[action]
    atype = model.get("type", "unknown")

    # ── Type condition (context-qualified thresholds) ─
    if atype == "epistemic":
        pass   # always admissible — breaks epistemic loop in all contexts

    elif atype == "operative":
        if state.epistemic_uncertainty > context.operative_eu_limit:
            return AdmissibilityResult(
                admissible=False, action=action,
                reason=(
                    f"[{context.domain}] operative blocked: "
                    f"eu={state.epistemic_uncertainty:.2f} > "
                    f"limit={context.operative_eu_limit}"
                ),
            )

    elif atype == "irreversible":
        if state.epistemic_uncertainty > context.irreversible_eu_limit:
            return AdmissibilityResult(
                admissible=False, action=action,
                reason=(
                    f"[{context.domain}] irreversible blocked: "
                    f"eu={state.epistemic_uncertainty:.2f} > "
                    f"limit={context.irreversible_eu_limit}"
                ),
            )
        if state.reversibility <= context.irreversible_rev_min:
            return AdmissibilityResult(
                admissible=False, action=action,
                reason=(
                    f"[{context.domain}] irreversible blocked: "
                    f"rev={state.reversibility:.2f} <= "
                    f"min={context.irreversible_rev_min}"
                ),
            )
        if state.risk <= context.irreversible_risk_min:
            return AdmissibilityResult(
                admissible=False, action=action,
                reason=(
                    f"[{context.domain}] irreversible not justified: "
                    f"risk={state.risk:.2f} <= {context.irreversible_risk_min}"
                ),
            )

    # ── Effect condition (context-independent) ────────
    if model.get("avoid_when", lambda s: False)(state):
        return AdmissibilityResult(
            admissible=False, action=action,
            reason="avoid_when condition active"
        )

    # ── Governance pre-check (constitutional — context-independent) ──
    if state.risk > 0.9 and state.reversibility < 0.2:
        return AdmissibilityResult(
            admissible=False, action=action,
            reason="governance pre-check: VETO_2 would fire"
        )

    semantic = context.action_semantics.get(action, "")
    return AdmissibilityResult(
        admissible=True, action=action,
        reason=f"admissible [{context.domain}]" + (f" — {semantic}" if semantic else "")
    )


# ─────────────────────────────────────────────
# Context-weighted evaluation (CHANGED in v12.0)
# ─────────────────────────────────────────────

def evaluate_projected(
    projected: WorldState,
    current:   WorldState,
    context:   Context = DEFAULT_CONTEXT,
) -> float:
    """
    Context-weighted evaluation.

    Clinical:  risk reduction highly valued, reversibility loss very costly
    Security:  risk reduction valued, reversibility moderate
    Industrial: balanced
    """
    risk_gain  = (current.risk           - projected.risk)           * context.risk_weight
    rev_gain   = (projected.reversibility - current.reversibility)   * context.reversibility_weight
    rev_loss   = -(current.reversibility  - projected.reversibility) * context.reversibility_weight * 1.4 \
                  if projected.reversibility < current.reversibility else 0.0
    eu_gain    = (current.epistemic_uncertainty - projected.epistemic_uncertainty) * context.eu_weight

    return risk_gain + rev_gain + rev_loss + eu_gain


# ─────────────────────────────────────────────
# Foresight with context (updated from v11.6)
# ─────────────────────────────────────────────

DISCOUNT = 0.7


def best_single_step_ctx(
    state:     WorldState,
    candidates:list[str],
    penalties: dict[str, float],
    model:     LearnedModel,
    context:   Context,
) -> tuple[str, float]:
    best, best_score = candidates[0], float("-inf")
    for action in candidates:
        if not is_admissible(action, state, context).admissible:
            continue
        projected = simulate_from_learned(state, action, model)
        score     = evaluate_projected(projected, state, context) - penalties.get(action, 0.0)
        if score > best_score:
            best_score, best = score, action
    return best, best_score


def select_action_contextual(
    state:      WorldState,
    model:      LearnedModel,
    candidates: list[str],
    context:    Context,
    mode:       str = "normal",
    discount:   float = DISCOUNT,
) -> tuple[str, Optional[str], dict[str, AdmissibilityResult]]:
    """2-step foresight with context-qualified admissibility and evaluation."""
    gated, gate_reason = epistemic_gating(state, candidates)
    filter_results     = semantic_filter(gated, state, mode)
    penalties          = {a: p for a, p, _ in filter_results}

    admissibility_map: dict[str, AdmissibilityResult] = {}
    scores: dict[str, float] = {}

    for action in gated:
        adm = is_admissible(action, state, context)
        admissibility_map[action] = adm

        if not adm.admissible:
            scores[action] = float("-inf")
            continue

        s1        = simulate_from_learned(state, action, model)
        immediate = evaluate_projected(s1, state, context) - penalties.get(action, 0.0)

        followup, _ = best_single_step_ctx(s1, gated, penalties, model, context)
        s2          = simulate_from_learned(s1, followup, model)
        future      = evaluate_projected(s2, s1, context)

        scores[action] = immediate + discount * future

    admissible_scores = {a: s for a, s in scores.items() if s > float("-inf")}
    if admissible_scores:
        best = max(admissible_scores, key=admissible_scores.get)
    else:
        best = next(
            (a for a in gated if BASE_ACTION_MODEL.get(a, {}).get("type") == "epistemic"),
            gated[0]
        )

    return best, gate_reason, admissibility_map


# ─────────────────────────────────────────────
# Run one scenario in one context
# ─────────────────────────────────────────────

def run_context(
    context:              Context,
    initial_uncertainty:  float = 0.55,
    max_iter:             int   = 8,
) -> None:
    state   = WorldState(risk=0.75, reversibility=0.5,
                         epistemic_uncertainty=initial_uncertainty)
    model   = LearnedModel(BASE_ACTION_MODEL)
    buffers = BufferStore()
    traj    = Trajectory()
    mode    = "normal"
    blocks  = 0

    print(f"\n{'─' * 68}")
    print(f"Context: {context.domain.upper()}  —  {context.description}")
    print(f"  risk_tolerance={context.risk_tolerance}  "
          f"irreversible_eu_limit={context.irreversible_eu_limit}  "
          f"operative_eu_limit={context.operative_eu_limit}")
    print(f"  risk_weight={context.risk_weight}  "
          f"reversibility_weight={context.reversibility_weight}")
    print(f"\nInitial state: {state}")

    for i in range(max_iter):
        print(f"\n  Step {i+1}  |  {state}")

        if traj.constraint_streak >= 3:
            mode = "safe"

        best, gate_reason, adm_map = select_action_contextual(
            state, model, ACTIONS, context, mode=mode
        )

        if gate_reason:
            print(f"    ⚠ GATE: {gate_reason}")

        blocked = [(a, r.reason) for a, r in adm_map.items() if not r.admissible]
        for a, reason in blocked:
            blocks += 1
            print(f"    ✗ {a:22s}  {reason}")

        decision, authority = memi_decision(state, best)
        effective = replan(decision, best)

        sem = context.action_semantics.get(best, "")
        print(f"    ✓ {best:22s}  [{decision:20s}  auth={authority}]"
              + (f"  \"{sem}\"" if sem else ""))

        if effective is None:
            print("    → Stop: operator required")
            break

        if effective != best:
            print(f"    → Replan: {best} → {effective}")

        state_before = state.copy()
        state        = next_state(state, effective)

        # Learning (v11.x, unchanged)
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
                entry, verdict, _ = flush_buffer(buf, model)
                if entry:
                    print(f"    ✎ LEARN  adj="
                          f"{ {k:round(v,4) for k,v in entry.adjustment.items() if abs(v)>0.001} }")

        traj.add({"step": i, "action": effective, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})

    print(f"\n  Final: {state}  |  admissibility blocks: {blocks}")


# ─────────────────────────────────────────────
# Main — same scenario, three contexts
# ─────────────────────────────────────────────

def run():
    print("\n" + "=" * 72)
    print("M.E.M.I. v12.0 — Contextual Admissibility")
    print("Same state. Three contexts. Different admissibility.")
    print("=" * 72)

    for ctx_name in ("security", "clinical", "industrial"):
        run_context(
            context=CONTEXTS[ctx_name],
            initial_uncertainty=0.55,
            max_iter=6,
        )

    # ── Contrast table: same state, all three contexts ──
    print(f"\n\n{'=' * 72}")
    print("Contrast: is_admissible() for same state across contexts")
    print("State: risk=0.75  rev=0.50  eu=0.55")
    print("=" * 72)
    test_state = WorldState(risk=0.75, reversibility=0.5, epistemic_uncertainty=0.55)
    print(f"\n  {'Action':22s}  {'security':12s}  {'clinical':12s}  {'industrial':12s}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*12}  {'─'*12}")
    for action in ACTIONS:
        row = [action]
        for ctx_name in ("security", "clinical", "industrial"):
            r = is_admissible(action, test_state, CONTEXTS[ctx_name])
            row.append("✓" if r.admissible else "✗")
        print(f"  {row[0]:22s}  {row[1]:12s}  {row[2]:12s}  {row[3]:12s}")

    print(f"\n  Note: 'isolate_process' (irreversible, eu=0.55):")
    for ctx_name in ("security", "clinical", "industrial"):
        r = is_admissible("isolate_process", test_state, CONTEXTS[ctx_name])
        print(f"    {ctx_name:12s}  {r.reason}")


if __name__ == "__main__":
    run()
