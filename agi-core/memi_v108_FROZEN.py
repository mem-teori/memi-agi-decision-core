"""
M.E.M.I. v10.8 — Action Effect Model / Symbolic Grounding
===========================================================

Builds on v10.7 (2-Step Foresight).

New in v10.8
------------
Actions are no longer labels that map to scores.
They are grounded in explicit effect signatures that describe
what dimension of the world they change, how, and under what conditions.

Four additions:
    ACTION_MODEL          — effect signatures per action
    simulate_from_model() — uses ACTION_MODEL instead of hardcoded if/elif
    semantic_filter()     — deprioritises actions inappropriate for current state
    explain_selection()   — produces a readable reason for the selected action

Unchanged:
    memi_decision()       — governance is not affected
    select_action_2step() — foresight structure is not affected
    Trajectory, evaluate_trajectory, run loop

Architectural invariant (preserved)
-------------------------------------
    Foresight selects the proposal.
    Effect model explains why.
    Governance decides whether it is permitted.

Bærende sætning
---------------
    An action is grounded when the system understands
    what dimension of the world it changes —
    and can use that understanding to reason about
    whether the action is appropriate before proposing it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

@dataclass
class WorldState:
    risk:          float
    reversibility: float

    def copy(self) -> "WorldState":
        return WorldState(risk=self.risk, reversibility=self.reversibility)

    def __str__(self) -> str:
        return f"risk={self.risk:.3f}  rev={self.reversibility:.3f}"


# ─────────────────────────────────────────────
# ACTION MODEL (NEW in v10.8)
# ─────────────────────────────────────────────

ACTION_MODEL: dict[str, dict] = {
    "collect_telemetry": {
        "type":           "epistemic",
        "primary_effect": "reduce_uncertainty",
        "effects": {
            "risk":          -0.12,
            "reversibility": +0.12,
        },
        "effective_when":          lambda s: s.risk > 0.3,
        "diminishing_returns_when":lambda s: s.reversibility >= 0.95,
        "avoid_when":              lambda s: False,
        "description": "Reduces uncertainty by gathering information. "
                       "Improves reversibility as a primary effect.",
    },
    "monitor": {
        "type":           "epistemic",
        "primary_effect": "reduce_uncertainty",
        "effects": {
            "risk":          -0.04,
            "reversibility": +0.04,
        },
        "effective_when":          lambda s: True,
        "diminishing_returns_when":lambda s: s.risk < 0.1 and s.reversibility >= 0.95,
        "avoid_when":              lambda s: False,
        "description": "Minimal-footprint observation. "
                       "Always safe, always marginally useful.",
    },
    "reduce_load": {
        "type":           "operative",
        "primary_effect": "reduce_risk",
        "effects": {
            "risk":          -0.08,
            "reversibility":  0.00,
        },
        "effective_when":          lambda s: s.risk > 0.2,
        "diminishing_returns_when":lambda s: s.risk < 0.1,
        "avoid_when":              lambda s: False,
        "description": "Reduces risk directly without affecting reversibility. "
                       "Preferred operative action when risk is moderate.",
    },
    "isolate_process": {
        "type":           "irreversible",
        "primary_effect": "reduce_risk",
        "effects": {
            "risk":          -0.15,
            "reversibility": -0.10,
        },
        "effective_when":          lambda s: s.risk > 0.5 and s.reversibility > 0.4,
        "diminishing_returns_when":lambda s: s.risk < 0.2,
        "avoid_when":              lambda s: s.risk < 0.3,
        "description": "Highest risk reduction. Costs reversibility. "
                       "Avoid when risk is already low — the cost is not justified.",
    },
}

ACTIONS = list(ACTION_MODEL.keys())


# ─────────────────────────────────────────────
# simulate_from_model (NEW in v10.8)
# ─────────────────────────────────────────────

def simulate_from_model(state: WorldState, action: str) -> WorldState:
    """
    Deterministic 1-step simulation using ACTION_MODEL.
    Replaces hardcoded if/elif logic with structured effect lookup.

    This is the grounding point: the system's predictions about the
    world are derived from explicit effect signatures, not implicit code.
    """
    if action not in ACTION_MODEL:
        return state.copy()

    effects = ACTION_MODEL[action]["effects"]
    s = state.copy()
    s.risk          = max(0.0, min(1.0, s.risk          + effects.get("risk", 0.0)))
    s.reversibility = max(0.0, min(1.0, s.reversibility + effects.get("reversibility", 0.0)))
    return s


def next_state(state: WorldState, action: Optional[str] = None) -> WorldState:
    """Actual execution: model effects + stochastic drift."""
    s = simulate_from_model(state, action) if action else state.copy()
    s.risk          = max(0.0, min(1.0, s.risk          + random.uniform(-0.08, 0.04)))
    s.reversibility = max(0.0, min(1.0, s.reversibility + random.uniform(-0.04, 0.04)))
    return s


# ─────────────────────────────────────────────
# semantic_filter (NEW in v10.8)
# ─────────────────────────────────────────────

@dataclass
class FilterResult:
    action:      str
    retained:    bool
    reason:      str
    penalty:     float = 0.0


def semantic_filter(
    candidates: list[str],
    state:      WorldState,
    mode:       str = "normal",
) -> tuple[list[str], list[FilterResult]]:
    """
    Deprioritise actions that are inappropriate for the current state,
    based on their effect model conditions.

    Does NOT remove actions from the candidate list — governance may still
    permit them. Instead, returns penalty values that foresight applies
    on top of the score.

    In safe mode: irreversible actions get an additional penalty.
    """
    results  = []
    filtered = []

    for action in candidates:
        if action not in ACTION_MODEL:
            results.append(FilterResult(action=action, retained=True, reason="unknown action"))
            filtered.append(action)
            continue

        model   = ACTION_MODEL[action]
        penalty = 0.0
        reasons = []

        # Avoid condition
        if model["avoid_when"](state):
            penalty += 0.25
            reasons.append(f"avoid_when triggered (risk={state.risk:.2f})")

        # Diminishing returns
        if model["diminishing_returns_when"](state):
            penalty += 0.10
            reasons.append("diminishing returns in current state")

        # Safe mode: penalise irreversible actions
        if mode == "safe" and model["type"] == "irreversible":
            penalty += 0.30
            reasons.append("safe_mode: irreversible type penalised")

        # Ineffective (but not avoided)
        if not model["effective_when"](state):
            penalty += 0.05
            reasons.append("not in primary effective range")

        reason = "; ".join(reasons) if reasons else "no filter applied"
        results.append(FilterResult(
            action=action, retained=True, reason=reason, penalty=penalty
        ))
        filtered.append(action)

    return filtered, results


# ─────────────────────────────────────────────
# explain_selection (NEW in v10.8)
# ─────────────────────────────────────────────

def explain_selection(
    action:  str,
    state:   WorldState,
    score:   float,
    filter_result: Optional[FilterResult] = None,
) -> str:
    """
    Produce a human-readable explanation for why this action was selected.
    This is the first step toward the system being able to say:
    'I chose this action because...'
    """
    if action not in ACTION_MODEL:
        return f"Selected '{action}' (score={score:+.3f}) — no model available."

    model       = ACTION_MODEL[action]
    action_type = model["type"]
    primary     = model["primary_effect"]
    desc        = model["description"]

    # State context
    if state.risk > 0.6:
        context = "high-risk state"
    elif state.risk > 0.3:
        context = "moderate-risk state"
    else:
        context = "low-risk state"

    if state.reversibility < 0.4:
        context += ", low reversibility"
    elif state.reversibility > 0.8:
        context += ", high reversibility"

    # Type-specific reasoning
    if action_type == "epistemic":
        reasoning = (
            f"Epistemic action selected in {context}. "
            f"Primary effect: {primary}. "
            f"Reduces uncertainty before committing to operative action."
        )
    elif action_type == "operative":
        reasoning = (
            f"Operative action selected in {context}. "
            f"Primary effect: {primary}. "
            f"Direct risk reduction without reversibility cost."
        )
    elif action_type == "irreversible":
        reasoning = (
            f"Irreversible action selected in {context}. "
            f"Primary effect: {primary}. "
            f"Highest risk reduction — reversibility cost accepted."
        )
    else:
        reasoning = f"Action type: {action_type}."

    penalty_note = ""
    if filter_result and filter_result.penalty > 0:
        penalty_note = f" [Filter penalty: -{filter_result.penalty:.2f} — {filter_result.reason}]"

    return f"→ {action} (score={score:+.3f}): {reasoning}{penalty_note}"


# ─────────────────────────────────────────────
# M.E.M.I. (unchanged from v10.7)
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
# Evaluation (unchanged from v10.7)
# ─────────────────────────────────────────────

DISCOUNT = 0.7

def evaluate_projected(projected: WorldState, current: WorldState) -> float:
    risk_gain = (current.risk - projected.risk) * 1.0
    rev_gain  = (projected.reversibility - current.reversibility) * 0.5
    rev_loss  = -(current.reversibility - projected.reversibility) * 0.8 \
                if projected.reversibility < current.reversibility else 0.0
    return risk_gain + rev_gain + rev_loss

def best_single_step(
    state:      WorldState,
    candidates: list[str],
    penalties:  dict[str, float],
    mode:       str,
) -> tuple[str, float]:
    best_action = candidates[0]
    best_score  = float("-inf")
    for action in candidates:
        projected = simulate_from_model(state, action)
        score     = evaluate_projected(projected, state) - penalties.get(action, 0.0)
        if score > best_score:
            best_score  = score
            best_action = action
    return best_action, best_score


def select_action_2step(
    state:      WorldState,
    candidates: list[str],
    mode:       str = "normal",
    discount:   float = DISCOUNT,
) -> tuple[str, dict, list[FilterResult]]:
    """
    2-step foresight with semantic filtering (v10.8).
    Returns (best_action, trace, filter_results).
    """
    # Semantic filter — compute penalties
    _, filter_results = semantic_filter(candidates, state, mode)
    penalties = {fr.action: fr.penalty for fr in filter_results}

    cumulative_scores = {}
    trace_detail      = {}

    for action in candidates:
        state_1   = simulate_from_model(state, action)
        immediate = evaluate_projected(state_1, state) - penalties.get(action, 0.0)

        followup, _ = best_single_step(state_1, candidates, penalties, mode)
        state_2     = simulate_from_model(state_1, followup)
        future      = evaluate_projected(state_2, state_1)

        cumulative = immediate + discount * future
        cumulative_scores[action] = cumulative
        trace_detail[action] = {
            "type":            ACTION_MODEL[action]["type"],
            "primary_effect":  ACTION_MODEL[action]["primary_effect"],
            "state_1":         {"risk": round(state_1.risk, 3), "rev": round(state_1.reversibility, 3)},
            "immediate_score": round(immediate, 3),
            "penalty":         round(penalties.get(action, 0.0), 3),
            "best_followup":   followup,
            "state_2":         {"risk": round(state_2.risk, 3), "rev": round(state_2.reversibility, 3)},
            "future_score":    round(future, 3),
            "cumulative":      round(cumulative, 3),
        }

    best = max(cumulative_scores, key=cumulative_scores.get)
    return best, {"candidates": trace_detail, "mode": mode, "discount": discount}, filter_results


# ─────────────────────────────────────────────
# Trajectory (unchanged from v10.7)
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
    if not risks:
        return {}
    reduction = risks[0] - risks[-1]
    score     = reduction - traj.continuity_cost - traj.operator_load * 0.2
    return {
        "risk_reduction":  round(reduction, 3),
        "avg_risk":        round(sum(risks) / len(risks), 3),
        "continuity_cost": round(traj.continuity_cost, 3),
        "operator_load":   traj.operator_load,
        "goal_score":      round(score, 3),
    }


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run():
    state = WorldState(risk=0.8, reversibility=0.5)
    traj  = Trajectory()
    MAX_ITER = 10
    mode  = "normal"

    print("\n" + "=" * 72)
    print("M.E.M.I. v10.8 — Action Effect Model / Symbolic Grounding")
    print("=" * 72)
    print(f"\nInitial state: {state}\n")

    for i in range(MAX_ITER):
        print(f"\n{'─' * 68}")
        print(f"Step {i+1}  |  {state}")

        if traj.constraint_streak >= 3:
            mode = "safe"
            print("⚠  SAFE MODE")
        else:
            mode = "normal"

        # ── 2-step foresight + semantic filter ───
        best_action, trace, filter_results = select_action_2step(
            state, ACTIONS, mode=mode
        )
        penalties = {fr.action: fr.penalty for fr in filter_results}

        # ── Print foresight table ─────────────────
        print(f"\n  2-Step Foresight  (γ={trace['discount']})")
        print(f"  {'Action':22s}  {'Type':12s}  {'pen':>5}  "
              f"{'immed':>6}  {'followup':16s}  {'future':>6}  {'cumul':>7}")
        print(f"  {'─'*22}  {'─'*12}  {'─'*5}  {'─'*6}  {'─'*16}  {'─'*6}  {'─'*7}")

        best_fr = next(fr for fr in filter_results if fr.action == best_action)

        for action, d in trace["candidates"].items():
            marker = " ←" if action == best_action else "  "
            pen    = f"-{penalties.get(action,0):.2f}" if penalties.get(action,0) > 0 else "    —"
            print(f"  {action:22s}  {d['type']:12s}  {pen:>5}  "
                  f"{d['immediate_score']:+.3f}  {d['best_followup']:16s}  "
                  f"{d['future_score']:+.3f}  {d['cumulative']:+.3f}{marker}")

        # ── Explain selection ─────────────────────
        best_score = trace["candidates"][best_action]["cumulative"]
        explanation = explain_selection(best_action, state, best_score, best_fr)
        print(f"\n  {explanation}")

        # ── M.E.M.I. authority check ──────────────
        decision, authority = memi_decision(state, best_action)
        print(f"  M.E.M.I.: {decision:28s}  authority={authority}")

        # ── Replan ────────────────────────────────
        effective = replan(decision, best_action)
        if effective is None:
            print("  → Stop: operator required")
            traj.add({"step": i, "action": best_action,
                      "decision": "DEFERRED", "risk": state.risk, "mode": mode})
            break

        if effective != best_action:
            eff_type = ACTION_MODEL.get(effective, {}).get("type", "?")
            print(f"  → Replan: {best_action} ({ACTION_MODEL[best_action]['type']}) "
                  f"→ {effective} ({eff_type})")

        traj.add({"step": i, "action": best_action,
                  "decision": decision, "risk": state.risk, "mode": mode})
        state = next_state(state, effective)

    print(f"\n{'=' * 72}")
    print("Goal evaluation")
    print("=" * 72)
    for k, v in evaluate_trajectory(traj).items():
        print(f"  {k:<22} {v}")
    print(f"\n  Steps completed:  {len(traj.steps)}")
    print(f"  Final state:      {state}")


if __name__ == "__main__":
    run()
