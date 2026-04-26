"""
M.E.M.I. v10.7 — 2-Step Foresight
====================================

Builds on v10.6 (1-Step Foresight).

New in v10.7
------------
The system no longer only asks "what happens if I act?"
It asks "what action becomes possible after I act?"

2-step foresight loop:
    for each candidate action A:
        state_1 = simulate(state, A)
        best_followup, score_2 = best_action_at(state_1)
        state_2 = simulate(state_1, best_followup)
        cumulative_score(A) = score(state_1) + discount * score(state_2)

    select A with highest cumulative_score
    → M.E.M.I. authority check
    → execute or replan

Design rule (preserved from v10.6)
-----------------------------------
    Foresight selects the proposal.
    Governance decides whether the proposal is permitted.

Discount factor
---------------
    Future states count less than immediate states (γ = 0.7).
    This models that further futures are less certain —
    without introducing stochasticity.

Bærende sætning
---------------
    M.E.M.I. should not only ask "what happens if I act?"
    It should ask "what action becomes possible after I act?"
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


ACTIONS = ["isolate_process", "reduce_load", "collect_telemetry", "monitor"]
DISCOUNT = 0.7   # γ — future steps count less


# ─────────────────────────────────────────────
# World model (unchanged from v10.6)
# ─────────────────────────────────────────────

def simulate(state: WorldState, action: str) -> WorldState:
    """Deterministic 1-step projection. No random drift."""
    s = state.copy()
    if action == "collect_telemetry":
        s.risk          = max(0.0, s.risk          - 0.12)
        s.reversibility = min(1.0, s.reversibility + 0.12)
    elif action == "monitor":
        s.risk          = max(0.0, s.risk          - 0.04)
        s.reversibility = min(1.0, s.reversibility + 0.04)
    elif action == "reduce_load":
        s.risk          = max(0.0, s.risk          - 0.08)
    elif action == "isolate_process":
        s.risk          = max(0.0, s.risk          - 0.15)
        s.reversibility = max(0.0, s.reversibility - 0.10)
    return s

def next_state(state: WorldState, action: Optional[str] = None) -> WorldState:
    """Actual execution: deterministic effects + stochastic drift."""
    s = state.copy()
    s.risk          = max(0.0, min(1.0, s.risk          + random.uniform(-0.08, 0.04)))
    s.reversibility = max(0.0, min(1.0, s.reversibility + random.uniform(-0.04, 0.04)))
    if action == "collect_telemetry":
        s.risk          = max(0.0, s.risk - 0.12)
        s.reversibility = min(1.0, s.reversibility + 0.12)
    elif action == "monitor":
        s.risk          = max(0.0, s.risk - 0.04)
        s.reversibility = min(1.0, s.reversibility + 0.04)
    elif action == "reduce_load":
        s.risk          = max(0.0, s.risk - 0.08)
    elif action == "isolate_process":
        s.risk          = max(0.0, s.risk - 0.15)
        s.reversibility = max(0.0, s.reversibility - 0.10)
    return s


# ─────────────────────────────────────────────
# M.E.M.I. (unchanged from v10.6)
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
# Evaluation (unchanged from v10.6)
# ─────────────────────────────────────────────

def evaluate_projected(projected: WorldState, current: WorldState) -> float:
    risk_gain = (current.risk - projected.risk) * 1.0
    rev_gain  = (projected.reversibility - current.reversibility) * 0.5
    rev_loss  = -(current.reversibility - projected.reversibility) * 0.8 \
                if projected.reversibility < current.reversibility else 0.0
    return risk_gain + rev_gain + rev_loss


# ─────────────────────────────────────────────
# 2-Step Foresight (NEW in v10.7)
# ─────────────────────────────────────────────

def best_single_step(state: WorldState, candidates: list[str], mode: str) -> tuple[str, float]:
    """
    Find the best action at a given state using 1-step evaluation.
    Used as the inner loop of 2-step foresight.
    """
    best_action = candidates[0]
    best_score  = float("-inf")

    for action in candidates:
        projected = simulate(state, action)
        score     = evaluate_projected(projected, state)
        if mode == "safe" and projected.reversibility < state.reversibility:
            score -= 0.3
        if score > best_score:
            best_score  = score
            best_action = action

    return best_action, best_score


def select_action_2step(
    state:      WorldState,
    candidates: list[str],
    mode:       str = "normal",
    discount:   float = DISCOUNT,
) -> tuple[str, dict]:
    """
    2-step foresight:
        for each candidate action A:
            state_1        = simulate(state, A)
            immediate      = evaluate(state_1, state)
            followup, s2   = best_single_step(state_1, candidates)
            state_2        = simulate(state_1, followup)
            future         = evaluate(state_2, state_1)
            cumulative     = immediate + discount * future

        select A with highest cumulative score.

    The key question answered:
        "What action becomes possible after I act?"
    """
    cumulative_scores = {}
    trace_detail      = {}

    for action in candidates:
        # Step 1: simulate immediate consequence
        state_1   = simulate(state, action)
        immediate = evaluate_projected(state_1, state)
        if mode == "safe" and state_1.reversibility < state.reversibility:
            immediate -= 0.3

        # Step 2: find best followup from state_1
        followup, _ = best_single_step(state_1, candidates, mode)
        state_2     = simulate(state_1, followup)
        future      = evaluate_projected(state_2, state_1)

        cumulative = immediate + discount * future

        cumulative_scores[action] = cumulative
        trace_detail[action] = {
            "state_1":         {"risk": round(state_1.risk, 3), "rev": round(state_1.reversibility, 3)},
            "immediate_score": round(immediate, 3),
            "best_followup":   followup,
            "state_2":         {"risk": round(state_2.risk, 3), "rev": round(state_2.reversibility, 3)},
            "future_score":    round(future, 3),
            "cumulative":      round(cumulative, 3),
        }

    best = max(cumulative_scores, key=cumulative_scores.get)

    return best, {
        "candidates": trace_detail,
        "selected":   best,
        "mode":       mode,
        "discount":   discount,
    }


# ─────────────────────────────────────────────
# Trajectory (unchanged from v10.6)
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
    MAX_ITER = 12
    mode  = "normal"

    print("\n" + "=" * 68)
    print("M.E.M.I. v10.7 — 2-Step Foresight")
    print(f"Discount γ = {DISCOUNT}")
    print("=" * 68)
    print(f"\nInitial state: {state}\n")

    for i in range(MAX_ITER):
        print(f"\n{'─' * 60}")
        print(f"Step {i+1}  |  {state}")

        # ── Strategy adaptation ───────────────────
        if traj.constraint_streak >= 3:
            mode = "safe"
            print("⚠  SAFE MODE")
        else:
            mode = "normal"

        # ── 2-step foresight ──────────────────────
        best_action, trace = select_action_2step(state, ACTIONS, mode=mode)

        print(f"\n  2-Step Foresight  (γ={trace['discount']})")
        print(f"  {'Action':22s}  {'immed':>6}  {'followup':16s}  {'future':>6}  {'cumul':>7}")
        print(f"  {'─'*22}  {'─'*6}  {'─'*16}  {'─'*6}  {'─'*7}")
        for action, d in trace["candidates"].items():
            marker = " ←" if action == best_action else "  "
            print(f"  {action:22s}  {d['immediate_score']:+.3f}  "
                  f"{d['best_followup']:16s}  {d['future_score']:+.3f}  "
                  f"{d['cumulative']:+.3f}{marker}")

        # ── M.E.M.I. authority check ──────────────
        decision, authority = memi_decision(state, best_action)
        print(f"\n  M.E.M.I.: {decision:28s}  authority={authority}")

        # ── Replan ────────────────────────────────
        effective = replan(decision, best_action)
        if effective is None:
            print("  → Stop: operator required")
            traj.add({"step": i, "action": best_action,
                      "decision": "DEFERRED", "risk": state.risk, "mode": mode})
            break

        if effective != best_action:
            print(f"  → Replan: {best_action} → {effective}")

        traj.add({"step": i, "action": best_action,
                  "decision": decision, "risk": state.risk, "mode": mode})
        state = next_state(state, effective)

    print(f"\n{'=' * 68}")
    print("Goal evaluation")
    print("=" * 68)
    for k, v in evaluate_trajectory(traj).items():
        print(f"  {k:<22} {v}")
    print(f"\n  Steps completed:    {len(traj.steps)}")
    print(f"  Final state:        {state}")


if __name__ == "__main__":
    run()
