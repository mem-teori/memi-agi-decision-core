"""
M.E.M.I. v12.1 — Context-Scoped LearnedModel
==============================================

Builds on v12.0 (Contextual Admissibility).

New in v12.1
------------
Learned adjustments are stored per domain, not globally.

What `collect_telemetry` does in security may differ from what it does
in clinical care. The system now learns these separately.

One addition:
    ContextualLearnedModel    — extends LearnedModel with per-domain learned_effects
                                base effects unchanged
                                learned adjustments keyed by (domain, action, dimension)

One change:
    run_context()             — uses ContextualLearnedModel
                                passes context.domain when storing/retrieving learned effects

Unchanged:
    Context                   — unchanged
    is_admissible()           — unchanged
    evaluate_projected()      — unchanged
    memi_decision()           — unchanged
    VETO conditions           — unchanged
    LearningBuffer / flush logic — unchanged

Cross-context generalisation (v12.3):
    Transfer of learned adjustments between domains is a governed decision.
    It does not happen automatically.

Architectural invariant (preserved)
-------------------------------------
    The system learns per context — not a single global model.
    Cross-context transfer is a governed decision, not automatic.

Bærende sætning
---------------
    What the system learns in one domain
    does not automatically apply in another.
"""

from __future__ import annotations

import copy
import time
import uuid
import random
from dataclasses import dataclass, field
from typing import Optional

from memi_v112 import (
    WorldState, ACTIONS, LEARNING_BOUNDS,
    LearningProposal,
    next_state, epistemic_gating,
    memi_decision, replan,
    semantic_filter,
    build_learning_proposal,
    Trajectory,
)
from memi_v113 import Verdict, govern_learning
from memi_v114 import (
    LearningLogEntry, apply_learning,
    BASE_ACTION_MODEL, simulate_from_learned,
)
from memi_v115 import BufferStore, flush_buffer
from memi_v120 import (
    Context, CONTEXTS, DEFAULT_CONTEXT,
    is_admissible, evaluate_projected,
    select_action_contextual,
)


# ─────────────────────────────────────────────
# ContextualLearnedModel (NEW in v12.1)
# ─────────────────────────────────────────────

class ContextualLearnedModel:
    """
    Extends LearnedModel with per-domain learned adjustments.

    Structure:
        _learned[domain][action][dimension] = adjustment

    Base effects (BASE_ACTION_MODEL[action]["effects"]) are NEVER modified.
    Learned adjustments for each domain are stored separately.

    What is learned in security does not transfer to clinical.
    Cross-domain transfer is a governed decision (v12.3).
    """

    def __init__(self, base_model: dict):
        self._base    = copy.deepcopy(base_model)
        self._learned: dict[str, dict[str, dict[str, float]]] = {}   # domain → action → dim → adj
        self._log:     list[LearningLogEntry] = []

    def _ensure_domain(self, domain: str, action: str) -> None:
        if domain not in self._learned:
            self._learned[domain] = {}
        if action not in self._learned[domain]:
            self._learned[domain][action] = {
                "risk": 0.0, "reversibility": 0.0, "epistemic_uncertainty": 0.0
            }

    def effective_effects(self, action: str, domain: str) -> dict:
        """Base effects + domain-specific learned adjustment."""
        base    = self._base[action]["effects"]
        learned = self._learned.get(domain, {}).get(action, {})
        return {
            dim: round(base.get(dim, 0.0) + learned.get(dim, 0.0), 4)
            for dim in ("risk", "reversibility", "epistemic_uncertainty")
        }

    def apply(self, proposal: LearningProposal, domain: str) -> LearningLogEntry:
        """Apply an ACCEPT proposal to the domain-specific learned adjustments."""
        action = proposal.action
        self._ensure_domain(domain, action)

        for dim, adj in proposal.proposed_adjustment.items():
            bound   = LEARNING_BOUNDS.get(dim, 0.05) * 2
            current = self._learned[domain][action].get(dim, 0.0)
            self._learned[domain][action][dim] = round(
                max(-bound, min(bound, current + adj)), 4
            )

        entry = LearningLogEntry(
            entry_id=str(uuid.uuid4())[:8],
            proposal_id=proposal.proposal_id,
            action=action,
            adjustment=dict(proposal.proposed_adjustment),
            cumulative_adjustment=dict(self._learned[domain][action]),
        )
        # Tag entry with domain for audit
        entry.__dict__["domain"] = domain
        self._log.append(entry)
        return entry

    def domain_summary(self, domain: str) -> dict:
        """Learned adjustments for one domain."""
        result = {}
        for action, adjs in self._learned.get(domain, {}).items():
            if any(abs(v) > 0.001 for v in adjs.values()):
                base    = self._base[action]["effects"]
                result[action] = {
                    "base":     {d: base.get(d, 0.0) for d in adjs},
                    "learned":  adjs,
                    "effective": self.effective_effects(action, domain),
                }
        return result

    def all_domains(self) -> list[str]:
        return [d for d in self._learned if self._learned[d]]

    def learning_log(self) -> list[LearningLogEntry]:
        return self._log


# ─────────────────────────────────────────────
# simulate using contextual model
# ─────────────────────────────────────────────

def simulate_contextual(
    state:   WorldState,
    action:  str,
    model:   ContextualLearnedModel,
    domain:  str,
) -> WorldState:
    effects = model.effective_effects(action, domain)
    s = state.copy()
    s.risk                  = max(0.0, min(1.0, s.risk                  + effects.get("risk", 0.0)))
    s.reversibility         = max(0.0, min(1.0, s.reversibility         + effects.get("reversibility", 0.0)))
    s.epistemic_uncertainty = max(0.0, min(1.0, s.epistemic_uncertainty + effects.get("epistemic_uncertainty", 0.0)))
    return s


def best_single_step_ctx_learned(
    state:     WorldState,
    candidates:list[str],
    penalties: dict[str, float],
    model:     ContextualLearnedModel,
    context:   Context,
) -> tuple[str, float]:
    best, best_score = candidates[0], float("-inf")
    for action in candidates:
        if not is_admissible(action, state, context).admissible:
            continue
        projected = simulate_contextual(state, action, model, context.domain)
        score     = evaluate_projected(projected, state, context) - penalties.get(action, 0.0)
        if score > best_score:
            best_score, best = score, action
    return best, best_score


def select_action_contextual_learned(
    state:      WorldState,
    model:      ContextualLearnedModel,
    candidates: list[str],
    context:    Context,
    mode:       str = "normal",
    discount:   float = 0.7,
) -> tuple[str, Optional[str], dict]:
    gated, gate_reason = epistemic_gating(state, candidates)
    filter_results     = semantic_filter(gated, state, mode)
    penalties          = {a: p for a, p, _ in filter_results}
    adm_map            = {}
    scores             = {}

    for action in gated:
        adm = is_admissible(action, state, context)
        adm_map[action] = adm
        if not adm.admissible:
            scores[action] = float("-inf")
            continue
        s1        = simulate_contextual(state, action, model, context.domain)
        immediate = evaluate_projected(s1, state, context) - penalties.get(action, 0.0)
        followup, _ = best_single_step_ctx_learned(s1, gated, penalties, model, context)
        s2        = simulate_contextual(s1, followup, model, context.domain)
        future    = evaluate_projected(s2, s1, context)
        scores[action] = immediate + discount * future

    admissible = {a: s for a, s in scores.items() if s > float("-inf")}
    best = max(admissible, key=admissible.get) if admissible else \
           next((a for a in gated if BASE_ACTION_MODEL.get(a, {}).get("type") == "epistemic"), gated[0])
    return best, gate_reason, adm_map


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

def run_context(
    context:             Context,
    shared_model:        ContextualLearnedModel,
    initial_uncertainty: float = 0.55,
    max_iter:            int   = 8,
) -> None:
    state   = WorldState(risk=0.75, reversibility=0.5,
                         epistemic_uncertainty=initial_uncertainty)
    buffers = BufferStore()
    traj    = Trajectory()
    mode    = "normal"
    applied = 0

    print(f"\n{'─' * 68}")
    print(f"Context: {context.domain.upper()}")

    for i in range(max_iter):
        print(f"\n  Step {i+1}  |  {state}")

        if traj.constraint_streak >= 3:
            mode = "safe"

        best, gate_reason, adm_map = select_action_contextual_learned(
            state, shared_model, ACTIONS, context, mode=mode
        )

        if gate_reason:
            print(f"    ⚠ GATE: {gate_reason}")

        for a, r in adm_map.items():
            if not r.admissible:
                print(f"    ✗ {a:22s}  {r.reason}")

        decision, authority = memi_decision(state, best)
        effective = replan(decision, best)

        # Show effective effects for selected action in this domain
        eff = shared_model.effective_effects(best, context.domain)
        learned_tag = ""
        domain_learned = shared_model._learned.get(context.domain, {}).get(best, {})
        if any(abs(v) > 0.001 for v in domain_learned.values()):
            learned_tag = " [learned model]"

        print(f"    ✓ {best:22s}  [{decision:20s}  auth={authority}]{learned_tag}")

        if effective is None:
            print("    → Stop: operator required")
            break
        if effective != best:
            print(f"    → Replan: {best} → {effective}")

        state_before = state.copy()
        state        = next_state(state, effective)

        # Learning — stored in context.domain scope
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
                # Flush into contextual model with domain scope
                from memi_v115 import flush_buffer as _flush
                # Build synthetic proposal from buffer
                wa    = buf.weighted_adjustment()
                avg_c = sum(p.confidence for p in buf.proposals) / buf.n()
                mag   = max(abs(v) for v in wa.values())
                import uuid as _uuid
                synth = LearningProposal(
                    proposal_id=f"ctx_{_uuid.uuid4().hex[:6]}",
                    action=effective,
                    expected_effect={}, observed_effect={},
                    delta=wa, confidence=round(avg_c, 3),
                    proposed_adjustment=wa,
                    adjustment_magnitude=round(mag, 4),
                    within_bounds=all(abs(v) <= LEARNING_BOUNDS.get(d, 0.05)
                                      for d, v in wa.items()),
                    requires_approval=(mag >= 0.05 or avg_c < 0.7),
                    justification=f"context={context.domain} n={buf.n()} avg_conf={avg_c:.3f}",
                )
                buf.clear()
                verdict = govern_learning(synth)
                if verdict.verdict == Verdict.ACCEPT:
                    entry = shared_model.apply(synth, context.domain)
                    applied += 1
                    print(f"    ✎ LEARN [{context.domain}] [{entry.entry_id}]  "
                          f"adj={ {k:round(v,4) for k,v in entry.adjustment.items() if abs(v)>0.001} }")

        traj.add({"step": i, "action": effective, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty, "mode": mode})

    print(f"\n  Final: {state}  |  applied: {applied}")

    # Show what was learned in this domain
    ds = shared_model.domain_summary(context.domain)
    if ds:
        print(f"  Learned in [{context.domain}]:")
        for action, data in ds.items():
            for dim in ("risk", "reversibility", "epistemic_uncertainty"):
                if abs(data["learned"].get(dim, 0)) > 0.001:
                    print(f"    {action:22s}  {dim:26s}  "
                          f"base={data['base'].get(dim,0):+.4f}  "
                          f"learned={data['learned'].get(dim,0):+.4f}  "
                          f"effective={data['effective'].get(dim,0):+.4f}")


def run():
    print("\n" + "=" * 72)
    print("M.E.M.I. v12.1 — Context-Scoped LearnedModel")
    print("One shared model. Learning stored per domain.")
    print("=" * 72)

    # One shared model — but learning is domain-scoped
    shared_model = ContextualLearnedModel(BASE_ACTION_MODEL)

    for ctx_name in ("security", "clinical", "industrial"):
        run_context(CONTEXTS[ctx_name], shared_model, max_iter=8)

    # Show separation: each domain learned independently
    print(f"\n{'=' * 72}")
    print("Learning separation: what each domain learned")
    print("=" * 72)
    for domain in shared_model.all_domains():
        ds = shared_model.domain_summary(domain)
        if ds:
            print(f"\n  [{domain}]")
            for action, data in ds.items():
                for dim in ("risk", "reversibility", "epistemic_uncertainty"):
                    if abs(data["learned"].get(dim, 0)) > 0.001:
                        print(f"    {action:22s}  {dim:22s}  "
                              f"learned={data['learned'].get(dim,0):+.4f}  "
                              f"effective={data['effective'].get(dim,0):+.4f}")
        else:
            print(f"\n  [{domain}]  no adjustments this run")

    print(f"\n  Domains with learned adjustments: {shared_model.all_domains()}")
    print(f"  Total audit entries: {len(shared_model.learning_log())}")
    print()
    print("  Cross-domain transfer: not automatic — governed decision in v12.3")


if __name__ == "__main__":
    run()
