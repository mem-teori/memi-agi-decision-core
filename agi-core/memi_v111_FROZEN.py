"""
M.E.M.I. v11.1 — Decision Payload / Uncertainty Propagation
=============================================================

Builds on v11.0 (Epistemic State / Meta-Cognition).

New in v11.1
------------
When a decision leaves the system, its uncertainty travels with it.

v11.0: the system knows whether it knows enough to act.
v11.1: the system sends that knowledge with the decision.

One addition:
    build_epistemic_payload()   — constructs a structured uncertainty
                                  package attached to every decision output

One extension:
    DecisionResult              — output dataclass now carries the payload
    run()                       — prints payload for each step

Unchanged:
    memi_decision()             — governance not affected
    epistemic_gating()          — gating not affected
    select_action_2step()       — foresight not affected
    ACTION_MODEL                — effect model not affected
    WorldState                  — state not affected
    Trajectory, evaluate_trajectory

New architectural invariant
----------------------------
    Foresight selects the proposal.
    Effect model explains why.
    Epistemic state qualifies confidence.
    Governance decides whether it is permitted.
    Decision payload carries uncertainty beyond the system.

Bærende sætning
---------------
    A decision is not complete when action is selected.
    It is complete when its uncertainty is carried with it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────
# State (unchanged from v11.0)
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


EPISTEMIC_GATE            = 0.7
UNCERTAINTY_PENALTY_WEIGHT = 0.4


# ─────────────────────────────────────────────
# Decision Payload (NEW in v11.1)
# ─────────────────────────────────────────────

@dataclass
class EpistemicPayload:
    """
    Structured uncertainty package attached to every decision.

    This is what travels beyond the system — to the operator,
    to the audit log, to downstream systems.

    uncertainty_travels = True is a constitutional flag:
    the system cannot emit a decision without its uncertainty.
    """
    epistemic_uncertainty: float
    confidence:            str       # "high" / "moderate" / "low" / "unknown"
    risk:                  float
    reversibility:         float
    decision:              str
    action:                str
    why_uncertain:         list[str]
    missing_information:   list[str]
    recommended_followup:  Optional[str]
    gate_reason:           Optional[str]
    uncertainty_travels:   bool = True   # constitutional — never False


@dataclass
class DecisionResult:
    """Full output of one governed decision step."""
    step:            int
    action_proposed: str
    action_effective:str
    decision:        str
    authority:       str
    fast_lane:       bool
    state_before:    WorldState
    state_after:     Optional[WorldState]
    payload:         EpistemicPayload
    explanation:     str

    def summary(self) -> str:
        return (
            f"[{self.decision:25s}]  "
            f"auth={self.authority:6s}  "
            f"eu={self.payload.epistemic_uncertainty:.2f} "
            f"[{self.payload.confidence}]  "
            f"→ {self.action_effective}"
        )


# ─────────────────────────────────────────────
# build_epistemic_payload (NEW in v11.1)
# ─────────────────────────────────────────────

def build_epistemic_payload(
    state:        WorldState,
    decision:     str,
    action:       str,
    gate_reason:  Optional[str] = None,
    action_model: Optional[dict] = None,
) -> EpistemicPayload:
    """
    Construct the uncertainty package for a decision.

    why_uncertain:       reasons derived from state + gating
    missing_information: what epistemic actions would address
    recommended_followup: what the system recommends if uncertain
    """
    why_uncertain        = []
    missing_information  = []
    recommended_followup = None

    eu = state.epistemic_uncertainty

    # ── Why uncertain ────────────────────────
    if eu > EPISTEMIC_GATE:
        why_uncertain.append(
            f"epistemic_uncertainty={eu:.2f} exceeds gate={EPISTEMIC_GATE}"
        )
    if eu > 0.5:
        why_uncertain.append("model confidence below reliable threshold")
    if state.risk > 0.6 and eu > 0.4:
        why_uncertain.append("high risk combined with elevated uncertainty")
    if gate_reason:
        why_uncertain.append(f"epistemic gate triggered: {gate_reason}")
    if not why_uncertain:
        why_uncertain.append("no significant uncertainty detected")

    # ── Missing information ───────────────────
    if eu > 0.6:
        missing_information.append("direct measurement of primary risk indicator")
    if eu > 0.5:
        missing_information.append("confidence-weighted sensor coverage")
    if state.reversibility < 0.4:
        missing_information.append("reversibility confirmation before irreversible action")
    if decision in ("HELD", "VETO_1"):
        missing_information.append("sufficient data to trust risk assessment")
    if decision in ("OPERATOR_REVIEW", "DEFERRED", "VETO_2"):
        missing_information.append("operator validation of system state")

    # ── Recommended followup ──────────────────
    if eu > EPISTEMIC_GATE:
        recommended_followup = "collect_telemetry_before_intervention"
    elif eu > 0.5:
        recommended_followup = "monitor_before_operative_action"
    elif decision in ("OPERATOR_REVIEW", "DEFERRED"):
        recommended_followup = "operator_review_and_reauthorise"
    elif decision in ("HELD", "VETO_1"):
        recommended_followup = "await_data_then_resubmit"

    return EpistemicPayload(
        epistemic_uncertainty=round(eu, 3),
        confidence=state.certainty(),
        risk=round(state.risk, 3),
        reversibility=round(state.reversibility, 3),
        decision=decision,
        action=action,
        why_uncertain=why_uncertain,
        missing_information=missing_information,
        recommended_followup=recommended_followup,
        gate_reason=gate_reason,
        uncertainty_travels=True,
    )


# ─────────────────────────────────────────────
# ACTION MODEL (unchanged from v11.0)
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
        "description": "Reduces uncertainty and risk. Primary epistemic action.",
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
        "description": "Minimal-footprint observation.",
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
        "description": "Direct risk reduction. Requires sufficient certainty.",
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
        "description": "Highest risk reduction. Costs reversibility. Avoid when uncertain.",
    },
}

ACTIONS = list(ACTION_MODEL.keys())


# ─────────────────────────────────────────────
# Core functions (unchanged from v11.0)
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


def epistemic_gating(state: WorldState, candidates: list[str]) -> tuple[list[str], Optional[str]]:
    if state.epistemic_uncertainty <= EPISTEMIC_GATE:
        return candidates, None
    epistemic = [a for a in candidates if ACTION_MODEL.get(a, {}).get("type") == "epistemic"]
    if not epistemic:
        return candidates, None
    reason = (f"eu={state.epistemic_uncertainty:.2f} > gate={EPISTEMIC_GATE} "
              f"— epistemic actions only")
    return epistemic, reason


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
            results.append((action, 0.0, ""))
            continue
        m = ACTION_MODEL[action]
        penalty, reasons = 0.0, []
        if m["avoid_when"](state):          penalty += 0.25; reasons.append("avoid_when")
        if m["diminishing_returns_when"](s  := state): penalty += 0.10; reasons.append("diminishing")
        if mode == "safe" and m["type"] == "irreversible": penalty += 0.30; reasons.append("safe_mode")
        if not m["effective_when"](state):  penalty += 0.05; reasons.append("not effective")
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


def select_action_2step(state, candidates, mode="normal", discount=DISCOUNT):
    gated, gate_reason = epistemic_gating(state, candidates)
    filter_results     = semantic_filter(gated, state, mode)
    penalties          = {a: p for a, p, _ in filter_results}
    scores, trace      = {}, {}

    for action in gated:
        s1        = simulate_from_model(state, action)
        immediate = evaluate_projected(s1, state) - penalties.get(action, 0.0)
        followup, _= best_single_step(s1, gated, penalties)
        s2        = simulate_from_model(s1, followup)
        future    = evaluate_projected(s2, s1)
        cumulative= immediate + discount * future
        scores[action] = cumulative
        trace[action]  = {
            "type": ACTION_MODEL[action]["type"],
            "eu_effect": ACTION_MODEL[action]["effects"].get("epistemic_uncertainty", 0.0),
            "immediate": round(immediate, 3),
            "followup": followup,
            "future": round(future, 3),
            "cumulative": round(cumulative, 3),
            "penalty": round(penalties.get(action, 0.0), 3),
        }

    best = max(scores, key=scores.get)
    return best, trace, gate_reason


# ─────────────────────────────────────────────
# Trajectory (unchanged from v11.0)
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
    return {
        "risk_reduction":  round(risks[0] - risks[-1], 3),
        "eu_reduction":    round(eus[0]  - eus[-1],  3),
        "avg_risk":        round(sum(risks) / len(risks), 3),
        "avg_uncertainty": round(sum(eus)  / len(eus),  3),
        "continuity_cost": round(traj.continuity_cost, 3),
        "operator_load":   traj.operator_load,
        "goal_score":      round((risks[0]-risks[-1]) - traj.continuity_cost - traj.operator_load*0.2, 3),
    }


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run(initial_uncertainty: float = 0.75):
    state = WorldState(risk=0.75, reversibility=0.5,
                       epistemic_uncertainty=initial_uncertainty)
    traj  = Trajectory()
    MAX_ITER, mode = 10, "normal"

    print("\n" + "=" * 72)
    print("M.E.M.I. v11.1 — Decision Payload / Uncertainty Propagation")
    print("=" * 72)
    print(f"\nInitial state: {state}\n")

    results: list[DecisionResult] = []

    for i in range(MAX_ITER):
        print(f"\n{'─' * 68}")
        print(f"Step {i+1}  |  {state}")

        if traj.constraint_streak >= 3:
            mode = "safe"; print("⚠  SAFE MODE")
        else:
            mode = "normal"

        best, trace, gate_reason = select_action_2step(state, ACTIONS, mode=mode)

        if gate_reason:
            print(f"  ⚠ GATE: {gate_reason}")

        # Foresight table
        print(f"\n  {'Action':22s}  {'Type':12s}  {'eu→':>5}  {'immed':>6}  {'followup':16s}  {'cumul':>7}")
        print(f"  {'─'*22}  {'─'*12}  {'─'*5}  {'─'*6}  {'─'*16}  {'─'*7}")
        for action, d in trace.items():
            m = " ←" if action == best else "  "
            eu = f"{d['eu_effect']:+.2f}" if d['eu_effect'] != 0 else "   —"
            print(f"  {action:22s}  {d['type']:12s}  {eu:>5}  "
                  f"{d['immediate']:+.3f}  {d['followup']:16s}  {d['cumulative']:+.3f}{m}")

        decision, authority = memi_decision(state, best)
        effective = replan(decision, best)

        # ── Build epistemic payload (v11.1) ──────
        payload = build_epistemic_payload(
            state=state,
            decision=decision,
            action=effective or best,
            gate_reason=gate_reason,
        )

        # ── Print payload (compact) ──────────────
        print(f"\n  EPISTEMIC PAYLOAD")
        print(f"    uncertainty:  {payload.epistemic_uncertainty}  [{payload.confidence} confidence]")
        print(f"    why_uncertain: {'; '.join(payload.why_uncertain)}")
        if payload.missing_information:
            print(f"    missing:      {'; '.join(payload.missing_information)}")
        if payload.recommended_followup:
            print(f"    followup:     {payload.recommended_followup}")
        print(f"    travels:      {payload.uncertainty_travels}")

        print(f"\n  M.E.M.I.: {decision:28s}  authority={authority}")

        if effective is None:
            print("  → Stop: operator required")
            traj.add({"step": i, "action": best, "decision": "DEFERRED",
                      "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})
            result = DecisionResult(
                step=i, action_proposed=best, action_effective="escalate_to_operator",
                decision="DEFERRED", authority=authority, fast_lane=False,
                state_before=state.copy(), state_after=None,
                payload=payload, explanation="Operator required.",
            )
            results.append(result)
            break

        if effective != best:
            print(f"  → Replan: {best} → {effective}")

        state_after = next_state(state, effective)
        result = DecisionResult(
            step=i, action_proposed=best, action_effective=effective,
            decision=decision, authority=authority, fast_lane=False,
            state_before=state.copy(), state_after=state_after,
            payload=payload, explanation="",
        )
        results.append(result)
        traj.add({"step": i, "action": best, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})
        state = state_after

    print(f"\n{'=' * 72}")
    print("Decision summary (payload per step)")
    print("=" * 72)
    for r in results:
        print(f"  Step {r.step+1}  {r.summary()}")

    print(f"\n{'─' * 68}")
    print("Trajectory evaluation")
    for k, v in evaluate_trajectory(traj).items():
        print(f"  {k:<24} {v}")


if __name__ == "__main__":
    print("\n── High uncertainty (eu=0.75) ──")
    run(initial_uncertainty=0.75)
