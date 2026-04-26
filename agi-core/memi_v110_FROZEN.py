"""
M.E.M.I. v11.0 — Epistemic State / Meta-Cognition Layer
=========================================================

Builds on v10.8 (Action Effect Model / Symbolic Grounding).

New in v11.0
------------
The system adds a third dimension to world state: epistemic_uncertainty.

This is not a proxy for risk. It is a representation of how confident
the system is in its own assessment of the world.

    risk                 = what could go wrong
    epistemic_uncertainty = how certain we are that we know

They must not be conflated.

Three additions:
    WorldState.epistemic_uncertainty   — explicit third dimension
    epistemic_effect in ACTION_MODEL   — how actions affect certainty
    epistemic_gating()                 — system prioritises reduction
                                         of uncertainty when it is too
                                         high to act reliably

One extension:
    evaluate_projected()               — uncertainty penalty added
    explain_selection()                — epistemic reasoning included

Unchanged:
    memi_decision()                    — governance is not affected
    select_action_2step()              — foresight structure unchanged
    Trajectory, evaluate_trajectory    — loop logic unchanged

Architectural invariant (updated)
-----------------------------------
    Foresight selects the proposal.
    Effect model explains why.
    Epistemic state qualifies confidence.
    Governance decides whether it is permitted.

Bærende sætning
---------------
    A system is not intelligent when it can act.
    It is intelligent when it knows when it should not act.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# State (EXTENDED in v11.0)
# ─────────────────────────────────────────────

@dataclass
class WorldState:
    risk:                  float
    reversibility:         float
    epistemic_uncertainty: float = 0.5   # NEW — 0.0 = certain, 1.0 = unknown

    def copy(self) -> "WorldState":
        return WorldState(
            risk=self.risk,
            reversibility=self.reversibility,
            epistemic_uncertainty=self.epistemic_uncertainty,
        )

    def certainty(self) -> str:
        u = self.epistemic_uncertainty
        if u <= 0.2:  return "high certainty"
        if u <= 0.5:  return "moderate certainty"
        if u <= 0.8:  return "low certainty"
        return "unknown"

    def __str__(self) -> str:
        return (f"risk={self.risk:.3f}  "
                f"rev={self.reversibility:.3f}  "
                f"eu={self.epistemic_uncertainty:.3f} [{self.certainty()}]")


# ─────────────────────────────────────────────
# Epistemic gating threshold
# ─────────────────────────────────────────────

EPISTEMIC_GATE = 0.7   # above this: system must reduce uncertainty first
UNCERTAINTY_PENALTY_WEIGHT = 0.4


# ─────────────────────────────────────────────
# ACTION MODEL (EXTENDED in v11.0)
# ─────────────────────────────────────────────

ACTION_MODEL: dict[str, dict] = {
    "collect_telemetry": {
        "type":             "epistemic",
        "primary_effect":   "reduce_uncertainty",
        "effects": {
            "risk":                  -0.12,
            "reversibility":         +0.12,
            "epistemic_uncertainty": -0.15,   # NEW — primary epistemic effect
        },
        "effective_when":           lambda s: s.risk > 0.3 or s.epistemic_uncertainty > 0.4,
        "diminishing_returns_when": lambda s: s.reversibility >= 0.95 and s.epistemic_uncertainty < 0.2,
        "avoid_when":               lambda s: False,
        "description": "Reduces uncertainty and risk. Primary epistemic action. "
                       "Preferred when epistemic_uncertainty is high.",
    },
    "monitor": {
        "type":             "epistemic",
        "primary_effect":   "reduce_uncertainty",
        "effects": {
            "risk":                  -0.04,
            "reversibility":         +0.04,
            "epistemic_uncertainty": -0.05,   # NEW — weak epistemic effect
        },
        "effective_when":           lambda s: True,
        "diminishing_returns_when": lambda s: s.risk < 0.1 and s.epistemic_uncertainty < 0.15,
        "avoid_when":               lambda s: False,
        "description": "Minimal-footprint observation. Marginally reduces uncertainty.",
    },
    "reduce_load": {
        "type":             "operative",
        "primary_effect":   "reduce_risk",
        "effects": {
            "risk":                  -0.08,
            "reversibility":          0.00,
            "epistemic_uncertainty":  0.00,   # no epistemic effect
        },
        "effective_when":           lambda s: s.risk > 0.2 and s.epistemic_uncertainty < EPISTEMIC_GATE,
        "diminishing_returns_when": lambda s: s.risk < 0.1,
        "avoid_when":               lambda s: s.epistemic_uncertainty > EPISTEMIC_GATE,
        "description": "Direct risk reduction. No epistemic effect. "
                       "Avoid when uncertainty is too high to trust the assessment.",
    },
    "isolate_process": {
        "type":             "irreversible",
        "primary_effect":   "reduce_risk",
        "effects": {
            "risk":                  -0.15,
            "reversibility":         -0.10,
            "epistemic_uncertainty":  0.00,
        },
        "effective_when":           lambda s: s.risk > 0.5 and s.reversibility > 0.4
                                              and s.epistemic_uncertainty < 0.5,
        "diminishing_returns_when": lambda s: s.risk < 0.2,
        "avoid_when":               lambda s: s.risk < 0.3 or s.epistemic_uncertainty > 0.5,
        "description": "Highest risk reduction. Costs reversibility. "
                       "Avoid when risk is low or uncertainty is too high — "
                       "acting irreversibly on uncertain information is dangerous.",
    },
}

ACTIONS = list(ACTION_MODEL.keys())


# ─────────────────────────────────────────────
# simulate_from_model (EXTENDED in v11.0)
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
    # Uncertainty drifts upward without epistemic action (world changes, model ages)
    s.epistemic_uncertainty = max(0.0, min(1.0, s.epistemic_uncertainty + random.uniform(-0.02, 0.05)))
    return s


# ─────────────────────────────────────────────
# Epistemic gating (NEW in v11.0)
# ─────────────────────────────────────────────

def epistemic_gating(
    state:      WorldState,
    candidates: list[str],
) -> tuple[list[str], str | None]:
    """
    If epistemic_uncertainty is above EPISTEMIC_GATE, the system
    must prioritise epistemic actions — it does not know enough
    to act reliably on operative or irreversible proposals.

    Returns (filtered_candidates, gate_reason).
    gate_reason is None if no gating applies.

    This is meta-cognition:
        "I am uncertain about my own assessment.
         I should reduce that uncertainty before acting on it."
    """
    if state.epistemic_uncertainty <= EPISTEMIC_GATE:
        return candidates, None

    epistemic_candidates = [
        a for a in candidates
        if ACTION_MODEL.get(a, {}).get("type") == "epistemic"
    ]

    if not epistemic_candidates:
        return candidates, None

    reason = (
        f"epistemic_uncertainty={state.epistemic_uncertainty:.2f} > "
        f"gate={EPISTEMIC_GATE} — restricting to epistemic actions only"
    )
    return epistemic_candidates, reason


# ─────────────────────────────────────────────
# M.E.M.I. (unchanged from v10.8)
# ─────────────────────────────────────────────

def memi_decision(state: WorldState, action: str) -> tuple[str, str]:
    if state.risk > 0.9 and state.reversibility < 0.2:
        return "VETO_2", "NONE"
    if state.risk > 0.8:
        return "VETO_1", "LOW"
    if state.risk > 0.6:
        return "HELD", "LOW"
    if state.risk > 0.4:
        return "ALLOWED_WITH_CAUTION", "MEDIUM"
    return "ALLOWED", "HIGH"

def replan(decision: str, action: str) -> Optional[str]:
    if decision == "HELD":   return "collect_telemetry"
    if decision == "VETO_1": return "monitor"
    if decision in ["DEFERRED", "VETO_2"]: return None
    return action


# ─────────────────────────────────────────────
# Evaluation (EXTENDED in v11.0)
# ─────────────────────────────────────────────

DISCOUNT = 0.7

def evaluate_projected(projected: WorldState, current: WorldState) -> float:
    risk_gain    = (current.risk           - projected.risk)           * 1.0
    rev_gain     = (projected.reversibility - current.reversibility)   * 0.5
    rev_loss     = -(current.reversibility  - projected.reversibility) * 0.8 \
                    if projected.reversibility < current.reversibility else 0.0
    # NEW — uncertainty penalty: reward reducing uncertainty
    eu_improvement = (current.epistemic_uncertainty - projected.epistemic_uncertainty) * UNCERTAINTY_PENALTY_WEIGHT
    return risk_gain + rev_gain + rev_loss + eu_improvement


def semantic_filter(
    candidates: list[str],
    state:      WorldState,
    mode:       str = "normal",
) -> list[tuple[str, float, str]]:
    results = []
    for action in candidates:
        if action not in ACTION_MODEL:
            results.append((action, 0.0, "no model"))
            continue
        model   = ACTION_MODEL[action]
        penalty = 0.0
        reasons = []
        if model["avoid_when"](state):
            penalty += 0.25
            reasons.append(f"avoid_when triggered")
        if model["diminishing_returns_when"](state):
            penalty += 0.10
            reasons.append("diminishing returns")
        if mode == "safe" and model["type"] == "irreversible":
            penalty += 0.30
            reasons.append("safe_mode: irreversible")
        if not model["effective_when"](state):
            penalty += 0.05
            reasons.append("not in effective range")
        results.append((action, penalty, "; ".join(reasons) if reasons else ""))
    return results


def best_single_step(
    state:      WorldState,
    candidates: list[str],
    penalties:  dict[str, float],
) -> tuple[str, float]:
    best, best_score = candidates[0], float("-inf")
    for action in candidates:
        projected = simulate_from_model(state, action)
        score     = evaluate_projected(projected, state) - penalties.get(action, 0.0)
        if score > best_score:
            best_score = score
            best       = action
    return best, best_score


def select_action_2step(
    state:      WorldState,
    candidates: list[str],
    mode:       str = "normal",
    discount:   float = DISCOUNT,
) -> tuple[str, dict, str | None]:
    """
    2-step foresight with epistemic gating + semantic filtering.
    Returns (best_action, trace, gate_reason).
    """
    # Epistemic gating (NEW in v11.0)
    gated_candidates, gate_reason = epistemic_gating(state, candidates)

    # Semantic filter
    filter_results = semantic_filter(gated_candidates, state, mode)
    penalties      = {a: p for a, p, _ in filter_results}

    cumulative_scores = {}
    trace_detail      = {}

    for action in gated_candidates:
        state_1   = simulate_from_model(state, action)
        immediate = evaluate_projected(state_1, state) - penalties.get(action, 0.0)

        followup, _ = best_single_step(state_1, gated_candidates, penalties)
        state_2     = simulate_from_model(state_1, followup)
        future      = evaluate_projected(state_2, state_1)

        cumulative = immediate + discount * future
        cumulative_scores[action] = cumulative
        trace_detail[action] = {
            "type":            ACTION_MODEL[action]["type"],
            "eu_effect":       ACTION_MODEL[action]["effects"].get("epistemic_uncertainty", 0.0),
            "state_1":         {"risk": round(state_1.risk, 3), "rev": round(state_1.reversibility, 3),
                                "eu": round(state_1.epistemic_uncertainty, 3)},
            "immediate_score": round(immediate, 3),
            "penalty":         round(penalties.get(action, 0.0), 3),
            "best_followup":   followup,
            "future_score":    round(future, 3),
            "cumulative":      round(cumulative, 3),
        }

    best = max(cumulative_scores, key=cumulative_scores.get)
    return best, {"candidates": trace_detail, "mode": mode, "discount": discount}, gate_reason


# ─────────────────────────────────────────────
# Explain selection (EXTENDED in v11.0)
# ─────────────────────────────────────────────

def explain_selection(
    action:       str,
    state:        WorldState,
    score:        float,
    gate_reason:  Optional[str] = None,
    penalty:      float = 0.0,
    filter_reason:str = "",
) -> str:
    if action not in ACTION_MODEL:
        return f"Selected '{action}' (score={score:+.3f})"

    model      = ACTION_MODEL[action]
    atype      = model["type"]
    primary    = model["primary_effect"]
    certainty  = state.certainty()

    if state.risk > 0.6:    risk_ctx = "high-risk"
    elif state.risk > 0.3:  risk_ctx = "moderate-risk"
    else:                   risk_ctx = "low-risk"

    # Epistemic context (NEW)
    if state.epistemic_uncertainty > EPISTEMIC_GATE:
        eu_ctx = f"uncertainty={state.epistemic_uncertainty:.2f} ABOVE GATE — epistemic action required"
    elif state.epistemic_uncertainty > 0.5:
        eu_ctx = f"uncertainty={state.epistemic_uncertainty:.2f} — prefer epistemic actions"
    else:
        eu_ctx = f"uncertainty={state.epistemic_uncertainty:.2f} — sufficient to act"

    if atype == "epistemic":
        reasoning = (
            f"Epistemic action in {risk_ctx} state. "
            f"{eu_ctx}. "
            f"Primary: {primary}."
        )
    elif atype == "operative":
        reasoning = (
            f"Operative action in {risk_ctx} state. "
            f"{eu_ctx}. "
            f"Primary: {primary}. Uncertainty low enough to trust assessment."
        )
    else:
        reasoning = (
            f"Irreversible action in {risk_ctx} state. "
            f"{eu_ctx}. "
            f"Primary: {primary}."
        )

    gate_note   = f" [GATED: {gate_reason}]" if gate_reason else ""
    filter_note = f" [Penalty -{penalty:.2f}: {filter_reason}]" if penalty > 0 and filter_reason else ""

    return f"→ {action} (score={score:+.3f}): {reasoning}{gate_note}{filter_note}"


# ─────────────────────────────────────────────
# Trajectory (unchanged from v10.8)
# ─────────────────────────────────────────────

@dataclass
class Trajectory:
    steps:             list  = field(default_factory=list)
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

def evaluate_trajectory(traj: Trajectory) -> dict:
    risks = [s["risk"] for s in traj.steps]
    eus   = [s.get("eu", 0.5) for s in traj.steps]
    if not risks:
        return {}
    reduction  = risks[0] - risks[-1]
    eu_reduction = eus[0] - eus[-1] if eus else 0
    score      = reduction - traj.continuity_cost - traj.operator_load * 0.2
    return {
        "risk_reduction":     round(reduction, 3),
        "eu_reduction":       round(eu_reduction, 3),
        "avg_risk":           round(sum(risks) / len(risks), 3),
        "avg_uncertainty":    round(sum(eus) / len(eus), 3),
        "continuity_cost":    round(traj.continuity_cost, 3),
        "operator_load":      traj.operator_load,
        "goal_score":         round(score, 3),
    }


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run(initial_uncertainty: float = 0.75):
    """
    Run with high initial epistemic_uncertainty to demonstrate gating.
    The system must first reduce uncertainty before operative actions
    become available.
    """
    state = WorldState(risk=0.75, reversibility=0.5, epistemic_uncertainty=initial_uncertainty)
    traj  = Trajectory()
    MAX_ITER = 12
    mode  = "normal"

    print("\n" + "=" * 72)
    print("M.E.M.I. v11.0 — Epistemic State / Meta-Cognition Layer")
    print(f"Epistemic gate: {EPISTEMIC_GATE}  |  Uncertainty weight: {UNCERTAINTY_PENALTY_WEIGHT}")
    print("=" * 72)
    print(f"\nInitial state: {state}\n")

    for i in range(MAX_ITER):
        print(f"\n{'─' * 70}")
        print(f"Step {i+1}  |  {state}")

        if traj.constraint_streak >= 3:
            mode = "safe"
            print("⚠  SAFE MODE")
        else:
            mode = "normal"

        best_action, trace, gate_reason = select_action_2step(
            state, ACTIONS, mode=mode
        )

        # Gate notification
        if gate_reason:
            print(f"\n  ⚠ EPISTEMIC GATE: {gate_reason}")

        # Foresight table
        print(f"\n  {'Action':22s}  {'Type':12s}  {'eu→':>6}  {'immed':>6}  {'followup':16s}  {'cumul':>7}")
        print(f"  {'─'*22}  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*16}  {'─'*7}")
        for action, d in trace["candidates"].items():
            marker  = " ←" if action == best_action else "  "
            eu_eff  = f"{d['eu_effect']:+.2f}" if d['eu_effect'] != 0 else "    —"
            print(f"  {action:22s}  {d['type']:12s}  {eu_eff:>6}  "
                  f"{d['immediate_score']:+.3f}  {d['best_followup']:16s}  "
                  f"{d['cumulative']:+.3f}{marker}")

        # Explanation
        best_score = trace["candidates"][best_action]["cumulative"]
        best_penalty = next((p for a, p, _ in semantic_filter(ACTIONS, state, mode) if a == best_action), 0.0)
        best_filter  = next((r for a, _, r in semantic_filter(ACTIONS, state, mode) if a == best_action), "")
        explanation  = explain_selection(best_action, state, best_score, gate_reason, best_penalty, best_filter)
        print(f"\n  {explanation}")

        # M.E.M.I.
        decision, authority = memi_decision(state, best_action)
        print(f"  M.E.M.I.: {decision:28s}  authority={authority}")

        effective = replan(decision, best_action)
        if effective is None:
            print("  → Stop: operator required")
            traj.add({"step": i, "action": best_action, "decision": "DEFERRED",
                      "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})
            break

        if effective != best_action:
            print(f"  → Replan: {best_action} → {effective}")

        traj.add({"step": i, "action": best_action, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})
        state = next_state(state, effective)

    print(f"\n{'=' * 72}")
    print("Goal evaluation")
    print("=" * 72)
    for k, v in evaluate_trajectory(traj).items():
        print(f"  {k:<24} {v}")
    print(f"\n  Steps:        {len(traj.steps)}")
    print(f"  Final state:  {state}")


if __name__ == "__main__":
    print("\n── Run 1: High initial uncertainty (eu=0.75) ──")
    run(initial_uncertainty=0.75)

    print("\n\n── Run 2: Low initial uncertainty (eu=0.2) ──")
    run(initial_uncertainty=0.20)
