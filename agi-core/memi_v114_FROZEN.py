"""
M.E.M.I. v11.4 — Apply Accepted Learning
==========================================

Builds on v11.3 (Govern Learning).

New in v11.4
------------
ACCEPT verdicts are applied to the action model.
QUEUE and REJECT verdicts are not applied.

The model update is explicit, bounded, and auditable:
    - Only `learned_effects` is written — base `effects` never change
    - Every application is logged with timestamp and proposal ID
    - The learning log is part of the audit trail

One addition:
    apply_learning()       — applies an ACCEPT verdict to learned_effects
    LearnedModel           — wrapper that carries base + learned adjustments

One new structure:
    LearnedModel           — ACTION_MODEL extended with learned_effects per action
                             Base effects unchanged. Learned adjustments stored separately.

Unchanged:
    memi_decision()        — not modified
    epistemic_gating()     — not modified
    VETO conditions        — not modified
    base ACTION_MODEL      — never overwritten

Architectural invariant (preserved)
-------------------------------------
    Foresight selects the proposal.
    Effect model explains why.
    Epistemic state qualifies confidence.
    Governance decides whether it is permitted.
    Decision payload carries uncertainty beyond the system.
    Learning proposals are governed decisions — not automatic updates.
    Only accepted proposals may update the model.
    Base effects are never overwritten — only learned adjustments are stored.

Bærende sætning
---------------
    The system learns now.
    Only what it was permitted to learn.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from memi_v112 import (
    WorldState, ACTIONS, EPISTEMIC_GATE,
    UNCERTAINTY_PENALTY_WEIGHT, LEARNING_BOUNDS,
    LearningProposal,
    next_state, epistemic_gating,
    memi_decision, replan, evaluate_projected,
    semantic_filter, best_single_step,
    observe_outcome, build_learning_proposal,
    Trajectory,
)
from memi_v113 import (
    GoverningVerdict, Verdict,
    govern_learning,
)

import copy


# ─────────────────────────────────────────────
# Learned Model (NEW in v11.4)
# ─────────────────────────────────────────────

@dataclass
class LearningLogEntry:
    """Audit record for one applied learning update."""
    entry_id:     str
    proposal_id:  str
    action:       str
    adjustment:   dict
    applied_at:   float = field(default_factory=time.time)
    cumulative_adjustment: dict = field(default_factory=dict)


class LearnedModel:
    """
    Wraps ACTION_MODEL with a learned_effects layer.

    Base effects (ACTION_MODEL[action]["effects"]) are NEVER modified.
    Learned adjustments are stored in learned_effects[action] and
    added on top of base effects when simulating.

    This means:
    - The original model is always recoverable
    - Every change is traceable to a specific proposal
    - The audit log shows the full history of model evolution
    """

    def __init__(self, base_model: dict):
        self._base     = copy.deepcopy(base_model)
        self._learned: dict[str, dict] = {
            action: {"risk": 0.0, "reversibility": 0.0, "epistemic_uncertainty": 0.0}
            for action in base_model
        }
        self._log: list[LearningLogEntry] = []

    def effective_effects(self, action: str) -> dict:
        """Base effects + learned adjustments = what simulation uses."""
        base    = self._base[action]["effects"]
        learned = self._learned.get(action, {})
        return {
            dim: round(base.get(dim, 0.0) + learned.get(dim, 0.0), 4)
            for dim in ("risk", "reversibility", "epistemic_uncertainty")
        }

    def apply(self, proposal: LearningProposal) -> LearningLogEntry:
        """
        Apply an ACCEPT proposal to learned_effects.
        Clamps cumulative learned adjustment to 2× LEARNING_BOUNDS
        so repeated learning cannot diverge from the base model.
        """
        action = proposal.action
        if action not in self._learned:
            self._learned[action] = {"risk": 0.0, "reversibility": 0.0, "epistemic_uncertainty": 0.0}

        for dim, adj in proposal.proposed_adjustment.items():
            bound  = LEARNING_BOUNDS.get(dim, 0.05) * 2   # cumulative cap = 2× single-step bound
            current = self._learned[action].get(dim, 0.0)
            new_val  = max(-bound, min(bound, current + adj))
            self._learned[action][dim] = round(new_val, 4)

        entry = LearningLogEntry(
            entry_id=str(uuid.uuid4())[:8],
            proposal_id=proposal.proposal_id,
            action=action,
            adjustment=dict(proposal.proposed_adjustment),
            cumulative_adjustment=dict(self._learned[action]),
        )
        self._log.append(entry)
        return entry

    def learning_log(self) -> list[LearningLogEntry]:
        return self._log

    def learned_summary(self) -> dict:
        return {
            action: {
                "base":     self._base[action]["effects"],
                "learned":  self._learned[action],
                "effective":self.effective_effects(action),
            }
            for action in self._base
            if any(abs(v) > 0.001 for v in self._learned[action].values())
        }


# ─────────────────────────────────────────────
# apply_learning (NEW in v11.4)
# ─────────────────────────────────────────────

def apply_learning(
    verdict:  GoverningVerdict,
    proposal: LearningProposal,
    model:    LearnedModel,
) -> Optional[LearningLogEntry]:
    """
    Apply a learning update if and only if verdict is ACCEPT.
    Returns LearningLogEntry on success, None otherwise.
    """
    if verdict.verdict != Verdict.ACCEPT:
        return None
    return model.apply(proposal)


# ─────────────────────────────────────────────
# simulate_from_learned_model
# ─────────────────────────────────────────────

def simulate_from_learned(state: WorldState, action: str, model: LearnedModel) -> WorldState:
    """Simulate using base + learned effects."""
    effects = model.effective_effects(action)
    s = state.copy()
    s.risk                  = max(0.0, min(1.0, s.risk                  + effects.get("risk", 0.0)))
    s.reversibility         = max(0.0, min(1.0, s.reversibility         + effects.get("reversibility", 0.0)))
    s.epistemic_uncertainty = max(0.0, min(1.0, s.epistemic_uncertainty + effects.get("epistemic_uncertainty", 0.0)))
    return s


def select_action_learned(
    state:     WorldState,
    model:     LearnedModel,
    candidates:list[str],
    mode:      str = "normal",
    discount:  float = 0.7,
) -> tuple[str, Optional[str]]:
    """2-step foresight using learned model."""
    gated, gate_reason = epistemic_gating(state, candidates)
    filter_results     = semantic_filter(gated, state, mode)
    penalties          = {a: p for a, p, _ in filter_results}

    scores = {}
    for action in gated:
        s1        = simulate_from_learned(state, action, model)
        immediate = evaluate_projected(s1, state) - penalties.get(action, 0.0)
        # best followup from learned model
        best_fu, _ = max(
            ((a, evaluate_projected(simulate_from_learned(s1, a, model), s1)
               - penalties.get(a, 0.0))
             for a in gated),
            key=lambda x: x[1],
        )
        s2     = simulate_from_learned(s1, best_fu, model)
        future = evaluate_projected(s2, s1)
        scores[action] = immediate + discount * future

    best = max(scores, key=scores.get)
    return best, gate_reason


# ─────────────────────────────────────────────
# Base ACTION_MODEL (unchanged)
# ─────────────────────────────────────────────

BASE_ACTION_MODEL: dict[str, dict] = {
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


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run(initial_uncertainty: float = 0.6):
    state  = WorldState(risk=0.75, reversibility=0.5,
                        epistemic_uncertainty=initial_uncertainty)
    model  = LearnedModel(BASE_ACTION_MODEL)
    traj   = Trajectory()
    MAX_ITER, mode = 10, "normal"

    proposals: list[LearningProposal] = []
    verdicts:  list[GoverningVerdict] = []
    applied:   list[LearningLogEntry] = []

    print("\n" + "=" * 72)
    print("M.E.M.I. v11.4 — Apply Accepted Learning")
    print("=" * 72)
    print(f"\nInitial state: {state}\n")

    for i in range(MAX_ITER):
        print(f"\n{'─' * 68}")
        print(f"Step {i+1}  |  {state}")

        if traj.constraint_streak >= 3:
            mode = "safe"
        else:
            mode = "normal"

        best, gate_reason = select_action_learned(state, model, ACTIONS, mode=mode)
        if gate_reason:
            print(f"  ⚠ GATE: {gate_reason}")

        decision, authority = memi_decision(state, best)
        effective = replan(decision, best)

        print(f"  proposed={best:22s}  decision={decision:24s}  auth={authority}")

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
            action_model=BASE_ACTION_MODEL,
        )

        if proposal:
            proposals.append(proposal)
            verdict = govern_learning(proposal)
            verdicts.append(verdict)

            # ── Apply if ACCEPT (v11.4) ──────────
            entry = apply_learning(verdict, proposal, model)

            v_sym = {"ACCEPT": "✓", "QUEUE": "⏸", "REJECT": "✗"}[verdict.verdict.value]
            print(f"  {v_sym} {verdict.verdict.value:6s}  [{proposal.proposal_id}]  "
                  f"{effective:22s}  "
                  f"magnitude={proposal.adjustment_magnitude:.3f}  "
                  f"confidence={proposal.confidence:.2f}")

            if entry:
                applied.append(entry)
                print(f"    ✎ APPLIED  [{entry.entry_id}]  "
                      f"cumulative: { {k: round(v,4) for k,v in entry.cumulative_adjustment.items() if abs(v)>0.001} }")
            elif verdict.verdict == Verdict.QUEUE:
                print(f"    ⏸ queued — {verdict.queued_for}")
            elif verdict.verdict == Verdict.REJECT:
                print(f"    ✗ rejected — {'; '.join(verdict.reasons)}")

        traj.add({"step": i, "action": effective, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})

    # ── Summary ──────────────────────────────────
    print(f"\n{'=' * 72}")
    print("Learning summary")
    print("=" * 72)
    counts = {v: sum(1 for vd in verdicts if vd.verdict == v) for v in Verdict}
    print(f"  Proposals: {len(proposals)}  "
          f"ACCEPT: {counts[Verdict.ACCEPT]}  "
          f"QUEUE: {counts[Verdict.QUEUE]}  "
          f"REJECT: {counts[Verdict.REJECT]}  "
          f"APPLIED: {len(applied)}")

    if model.learned_summary():
        print(f"\n  Learned adjustments (actions with non-zero learned effects):")
        for action, data in model.learned_summary().items():
            print(f"\n  {action}")
            for dim in ("risk", "reversibility", "epistemic_uncertainty"):
                base = data["base"].get(dim, 0.0)
                learned = data["learned"].get(dim, 0.0)
                effective_val = data["effective"].get(dim, 0.0)
                if abs(learned) > 0.001:
                    print(f"    {dim:26s}  base={base:+.4f}  "
                          f"learned={learned:+.4f}  "
                          f"effective={effective_val:+.4f}")
    else:
        print("\n  No learned adjustments this run.")

    print(f"\n  Audit log ({len(model.learning_log())} entries):")
    for entry in model.learning_log():
        print(f"    [{entry.entry_id}]  {entry.action:22s}  "
              f"proposal={entry.proposal_id}  "
              f"adj={ {k:round(v,4) for k,v in entry.adjustment.items() if abs(v)>0.001} }")

    print(f"\n  Final state:  {state}")


if __name__ == "__main__":
    run(initial_uncertainty=0.6)
