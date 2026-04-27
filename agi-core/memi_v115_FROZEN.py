"""
M.E.M.I. v11.5 — Multi-Observation Learning
=============================================

Builds on v11.4 (Apply Accepted Learning).

New in v11.5
------------
The system does not trust a single experience.
It trusts consistency across experiences.

v11.4: 1 ACCEPT observation → apply immediately
v11.5: N observations → check variance + confidence → apply only if consistent

One addition:
    LearningBuffer        — collects proposals per action before flushing
    should_flush()        — checks min_n + variance + avg_confidence
    flush_buffer()        — computes weighted average adjustment, applies via v11.4

Flush conditions (all three must hold):
    1. len(buffer) >= MIN_N          (enough observations)
    2. variance(adjustments) < VAR_THRESHOLD   (consistent direction)
    3. avg_confidence >= CONF_THRESHOLD        (high-quality observations)

When flushing:
    weighted_adjustment = sum(adj * conf) / sum(conf)   per dimension
    → passes through govern_learning() before apply_learning()

Unchanged:
    memi_decision()        — not modified
    epistemic_gating()     — not modified
    VETO conditions        — not modified
    base ACTION_MODEL      — never overwritten
    LearnedModel           — unchanged (still receives apply calls)

Architectural invariant (preserved)
-------------------------------------
    Foresight selects the proposal.
    Effect model explains why.
    Epistemic state qualifies confidence.
    Governance decides whether it is permitted.
    Decision payload carries uncertainty beyond the system.
    Learning proposals are governed decisions — not automatic updates.
    Only accepted proposals may update the model.
    The system does not trust a single experience.
    It trusts consistency across experiences.

Bærende sætning
---------------
    The system does not trust a single experience.
    It learns from consistency across experiences.
"""

from __future__ import annotations

import math
import uuid
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
    BASE_ACTION_MODEL, select_action_learned,
)


# ─────────────────────────────────────────────
# Buffer thresholds
# ─────────────────────────────────────────────

MIN_N          = 3      # minimum observations before flush is considered
VAR_THRESHOLD  = 0.002  # max variance of adjustments per dimension
CONF_THRESHOLD = 0.70   # min average confidence across buffered proposals


# ─────────────────────────────────────────────
# LearningBuffer (NEW in v11.5)
# ─────────────────────────────────────────────

@dataclass
class LearningBuffer:
    """
    Accumulates LearningProposals for one action.
    Flush only when all three conditions hold:
        1. len >= MIN_N
        2. variance(adjustments) < VAR_THRESHOLD per dimension
        3. avg(confidence) >= CONF_THRESHOLD
    """
    action:    str
    proposals: list[LearningProposal] = field(default_factory=list)

    def add(self, proposal: LearningProposal) -> None:
        self.proposals.append(proposal)

    def n(self) -> int:
        return len(self.proposals)

    def _dim_adjustments(self, dim: str) -> list[float]:
        return [p.proposed_adjustment.get(dim, 0.0) for p in self.proposals]

    def _variance(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def should_flush(self) -> tuple[bool, str]:
        """
        Check all three flush conditions.
        Returns (should_flush, reason).
        """
        if self.n() < MIN_N:
            return False, f"n={self.n()} < min_n={MIN_N}"

        dims = ("risk", "reversibility", "epistemic_uncertainty")
        for dim in dims:
            adjs = self._dim_adjustments(dim)
            var  = self._variance(adjs)
            if var > VAR_THRESHOLD:
                return False, (
                    f"variance({dim})={var:.5f} > threshold={VAR_THRESHOLD} — "
                    f"observations inconsistent"
                )

        avg_conf = sum(p.confidence for p in self.proposals) / self.n()
        if avg_conf < CONF_THRESHOLD:
            return False, f"avg_confidence={avg_conf:.3f} < threshold={CONF_THRESHOLD}"

        return True, (
            f"n={self.n()}>={MIN_N}, "
            f"variance<{VAR_THRESHOLD}, "
            f"avg_confidence={avg_conf:.3f}>={CONF_THRESHOLD}"
        )

    def weighted_adjustment(self) -> dict:
        """
        Compute confidence-weighted average adjustment per dimension.
        Uses only the consistent, high-confidence observations.
        """
        dims          = ("risk", "reversibility", "epistemic_uncertainty")
        total_conf    = sum(p.confidence for p in self.proposals)
        if total_conf == 0:
            return {d: 0.0 for d in dims}

        return {
            dim: round(
                sum(
                    p.proposed_adjustment.get(dim, 0.0) * p.confidence
                    for p in self.proposals
                ) / total_conf,
                4,
            )
            for dim in dims
        }

    def variance_summary(self) -> dict:
        dims = ("risk", "reversibility", "epistemic_uncertainty")
        return {
            dim: round(self._variance(self._dim_adjustments(dim)), 6)
            for dim in dims
        }

    def clear(self) -> None:
        self.proposals = []


# ─────────────────────────────────────────────
# flush_buffer (NEW in v11.5)
# ─────────────────────────────────────────────

def flush_buffer(
    buffer: LearningBuffer,
    model:  LearnedModel,
) -> tuple[Optional[LearningLogEntry], Optional[GoverningVerdict], str]:
    """
    Flush the buffer by:
        1. Computing the weighted average adjustment
        2. Building a synthetic LearningProposal from the average
        3. Running it through govern_learning()
        4. Applying if ACCEPT

    Returns (log_entry, verdict, reason).
    """
    flush_ok, flush_reason = buffer.should_flush()
    if not flush_ok:
        return None, None, flush_reason

    weighted_adj  = buffer.weighted_adjustment()
    avg_confidence = sum(p.confidence for p in buffer.proposals) / buffer.n()
    magnitude      = max(abs(v) for v in weighted_adj.values())
    within_bounds  = all(
        abs(v) <= LEARNING_BOUNDS.get(dim, 0.05)
        for dim, v in weighted_adj.items()
    )

    # Build synthetic proposal from average
    synthetic = LearningProposal(
        proposal_id=f"buf_{str(uuid.uuid4())[:6]}",
        action=buffer.action,
        expected_effect={},    # not used in governance check
        observed_effect={},
        delta=weighted_adj,
        confidence=round(avg_confidence, 3),
        proposed_adjustment=weighted_adj,
        adjustment_magnitude=round(magnitude, 4),
        within_bounds=within_bounds,
        requires_approval=(magnitude >= 0.05 or avg_confidence < CONF_THRESHOLD),
        justification=(
            f"Multi-observation flush: n={buffer.n()}, "
            f"avg_confidence={avg_confidence:.3f}, "
            f"variance={buffer.variance_summary()}"
        ),
    )

    verdict = govern_learning(synthetic)
    entry   = apply_learning(verdict, synthetic, model)
    buffer.clear()

    return entry, verdict, flush_reason


# ─────────────────────────────────────────────
# BufferStore — one buffer per action
# ─────────────────────────────────────────────

class BufferStore:
    def __init__(self):
        self._buffers: dict[str, LearningBuffer] = {}

    def add(self, proposal: LearningProposal) -> None:
        if proposal.action not in self._buffers:
            self._buffers[proposal.action] = LearningBuffer(action=proposal.action)
        self._buffers[proposal.action].add(proposal)

    def get(self, action: str) -> Optional[LearningBuffer]:
        return self._buffers.get(action)

    def all_actions(self) -> list[str]:
        return list(self._buffers.keys())

    def summary(self) -> dict:
        return {
            action: {
                "n": buf.n(),
                "should_flush": buf.should_flush()[0],
                "variance": buf.variance_summary(),
                "avg_confidence": round(
                    sum(p.confidence for p in buf.proposals) / buf.n(), 3
                ) if buf.n() > 0 else 0.0,
            }
            for action, buf in self._buffers.items()
        }


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run(initial_uncertainty: float = 0.6, max_iter: int = 15):
    state   = WorldState(risk=0.75, reversibility=0.5,
                         epistemic_uncertainty=initial_uncertainty)
    model   = LearnedModel(BASE_ACTION_MODEL)
    buffers = BufferStore()
    traj    = Trajectory()
    mode    = "normal"

    proposals_generated = 0
    verdicts_by_type    = {v: 0 for v in Verdict}
    applied_count       = 0
    flushed_count       = 0

    print("\n" + "=" * 72)
    print("M.E.M.I. v11.5 — Multi-Observation Learning")
    print(f"min_n={MIN_N}  var_threshold={VAR_THRESHOLD}  conf_threshold={CONF_THRESHOLD}")
    print("=" * 72)
    print(f"\nInitial state: {state}\n")

    for i in range(max_iter):
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

        # ── Observe + propose ────────────────────
        proposal = build_learning_proposal(
            action=effective,
            state_before=state_before,
            state_after=state,
            action_model=BASE_ACTION_MODEL,
        )

        if proposal:
            proposals_generated += 1
            buffers.add(proposal)
            buf = buffers.get(effective)

            flush_ok, flush_reason = buf.should_flush()
            print(f"  buffer[{effective}]: n={buf.n()}  "
                  f"{'→ FLUSH' if flush_ok else f'hold ({flush_reason})'}")

            if flush_ok:
                entry, verdict, _ = flush_buffer(buf, model)
                flushed_count += 1
                if verdict:
                    verdicts_by_type[verdict.verdict] += 1
                v_sym = {"ACCEPT": "✓", "QUEUE": "⏸", "REJECT": "✗"}.get(
                    verdict.verdict.value if verdict else "?", "?"
                )
                if entry:
                    applied_count += 1
                    print(f"    {v_sym} FLUSH {verdict.verdict.value}  [{entry.entry_id}]  "
                          f"weighted_adj={ {k:round(v,4) for k,v in entry.adjustment.items() if abs(v)>0.001} }")
                else:
                    reason = "; ".join(verdict.reasons) if verdict else "unknown"
                    print(f"    {v_sym} FLUSH {verdict.verdict.value if verdict else '?'}  — {reason}")

        traj.add({"step": i, "action": effective, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})

    # ── Summary ──────────────────────────────────
    print(f"\n{'=' * 72}")
    print("Multi-observation learning summary")
    print("=" * 72)
    print(f"  Proposals generated:  {proposals_generated}")
    print(f"  Buffer flushes:       {flushed_count}")
    print(f"  Applied:              {applied_count}")
    print(f"  ACCEPT/QUEUE/REJECT:  "
          f"{verdicts_by_type[Verdict.ACCEPT]} / "
          f"{verdicts_by_type[Verdict.QUEUE]} / "
          f"{verdicts_by_type[Verdict.REJECT]}")

    print(f"\n  Buffer state at end:")
    for action, info in buffers.summary().items():
        print(f"    {action:22s}  n={info['n']}  "
              f"avg_conf={info['avg_confidence']:.2f}  "
              f"{'ready' if info['should_flush'] else 'waiting'}")

    if model.learned_summary():
        print(f"\n  Learned adjustments (effective vs base):")
        for action, data in model.learned_summary().items():
            print(f"\n  {action}")
            for dim in ("risk", "reversibility", "epistemic_uncertainty"):
                base    = data["base"].get(dim, 0.0)
                learned = data["learned"].get(dim, 0.0)
                eff     = data["effective"].get(dim, 0.0)
                if abs(learned) > 0.001:
                    print(f"    {dim:26s}  base={base:+.4f}  "
                          f"learned={learned:+.4f}  effective={eff:+.4f}")
    else:
        print("\n  No adjustments applied this run.")

    print(f"\n  Audit log ({len(model.learning_log())} entries):")
    for e in model.learning_log():
        print(f"    [{e.entry_id}]  {e.action:22s}  "
              f"adj={ {k:round(v,4) for k,v in e.adjustment.items() if abs(v)>0.001} }")

    print(f"\n  Final state: {state}")


if __name__ == "__main__":
    run(initial_uncertainty=0.6, max_iter=15)
