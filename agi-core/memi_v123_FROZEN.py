"""
M.E.M.I. v12.3 — Governed Cross-Context Transfer
==================================================

Builds on v12.2 (Context Switching).

New in v12.3
------------
What the system learned in one domain may be proposed for transfer
to another — but only as a governed decision, never automatically.

Transfer is a proposal, not an update.
It follows the same governance chain as learning:
    observe → propose → govern → apply (if accepted)

Two additions:
    TransferProposal      — structured proposal to transfer learned adjustments
                            from source domain to target domain
    propose_transfer()    — builds a TransferProposal from ContextualLearnedModel
    govern_transfer()     — checks transfer against constitutional limits
                            and domain compatibility
    apply_transfer()      — applies an ACCEPT verdict to target domain

Transfer rules
--------------
    Transfers are only proposed when:
    1. Source domain has sufficient accumulated learning (min_entries)
    2. Target domain has insufficient learning (below threshold)
    3. Domain compatibility is plausible (configurable)

    Transfers are REJECTED when:
    1. The adjustment would violate LEARNING_BOUNDS in target domain
    2. The adjustment would flip the sign of an epistemic action's eu effect
    3. The source and target domains are considered incompatible

    Transfers are QUEUED when:
    1. Source confidence is below threshold
    2. Magnitude is above auto-apply limit

Constitutional limits (unchanged)
-----------------------------------
    memi_decision()         not affected
    VETO conditions         not affected
    uncertainty_travels     not affected
    base effects            never overwritten

Bærende sætning
---------------
    What the system learned in one domain
    may be proposed for transfer to another.
    The proposal must be governed.
    The transfer is never automatic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from memi_v112 import (
    WorldState, ACTIONS, LEARNING_BOUNDS,
    next_state, memi_decision, replan,
    build_learning_proposal, Trajectory,
)
from memi_v113 import Verdict, govern_learning
from memi_v114 import BASE_ACTION_MODEL, LearningLogEntry
from memi_v115 import BufferStore
from memi_v120 import Context, CONTEXTS
from memi_v121 import ContextualLearnedModel
from memi_v122 import ContextSession, resolve_context, select_action_contextual_learned


# ─────────────────────────────────────────────
# Domain compatibility map
# ─────────────────────────────────────────────

# Transfer is only proposed between compatible domain pairs.
# Incompatible = too different in risk tolerance / reversibility meaning.
DOMAIN_COMPATIBILITY: dict[str, list[str]] = {
    "security":   ["industrial"],          # moderate overlap
    "industrial": ["security"],            # moderate overlap
    "clinical":   [],                      # clinical is isolated — no auto-transfer
}

# Minimum audit log entries in source before transfer is considered
MIN_SOURCE_ENTRIES = 1

# Minimum confidence required for transfer proposal
TRANSFER_CONF_THRESHOLD = 0.65

# Max magnitude for auto-apply transfer (lower than learning — transfers are riskier)
TRANSFER_AUTO_MAGNITUDE = 0.03


# ─────────────────────────────────────────────
# TransferProposal (NEW in v12.3)
# ─────────────────────────────────────────────

@dataclass
class TransferProposal:
    """
    A proposal to transfer learned adjustments from source to target domain.

    This is NOT an update. It is a proposal.
    Governance decides whether it is applied.

    Fields
    ------
    proposal_id:      unique identifier
    source_domain:    where the knowledge was learned
    target_domain:    where it would be applied
    action:           which action this concerns
    adjustment:       the proposed adjustment (from source learned_effects)
    confidence:       confidence in the transfer (based on source evidence)
    magnitude:        max(|adjustment|)
    within_bounds:    whether adjustment is within LEARNING_BOUNDS
    requires_approval: True if magnitude >= TRANSFER_AUTO_MAGNITUDE
    justification:    human-readable reason
    applied:          False until governance approves
    """
    proposal_id:      str
    source_domain:    str
    target_domain:    str
    action:           str
    adjustment:       dict
    confidence:       float
    magnitude:        float
    within_bounds:    bool
    requires_approval:bool
    justification:    str
    applied:          bool = False

    def summary(self) -> str:
        status   = "within bounds" if self.within_bounds else "OUT OF BOUNDS"
        approval = "auto-apply candidate" if not self.requires_approval else "requires approval"
        return (
            f"Transfer [{self.source_domain} → {self.target_domain}]  "
            f"{self.action:22s}  "
            f"magnitude={self.magnitude:.4f}  "
            f"conf={self.confidence:.2f}  "
            f"[{status}]  [{approval}]"
        )


# ─────────────────────────────────────────────
# propose_transfer (NEW in v12.3)
# ─────────────────────────────────────────────

def propose_transfer(
    model:         ContextualLearnedModel,
    source_domain: str,
    target_domain: str,
    action:        str,
) -> Optional[TransferProposal]:
    """
    Build a TransferProposal if conditions are met.

    Returns None if:
    - domains are incompatible
    - source has insufficient learning entries
    - source has no non-trivial adjustment for this action
    - target already has similar adjustment (would be redundant)
    """
    # Compatibility check
    compatible = DOMAIN_COMPATIBILITY.get(source_domain, [])
    if target_domain not in compatible:
        return None

    # Source must have sufficient entries
    source_entries = [
        e for e in model.learning_log()
        if getattr(e, "domain", None) == source_domain and e.action == action
    ]
    if len(source_entries) < MIN_SOURCE_ENTRIES:
        return None

    # Source must have non-trivial adjustment
    source_adj = model._learned.get(source_domain, {}).get(action, {})
    if not any(abs(v) > 0.001 for v in source_adj.values()):
        return None

    # Target should have less learning (otherwise transfer is redundant)
    target_adj = model._learned.get(target_domain, {}).get(action, {})
    if sum(abs(v) for v in target_adj.values()) >= sum(abs(v) for v in source_adj.values()) * 0.8:
        return None   # target already learned similar amount

    # Build proposal: use source adjustment scaled by 0.7 (conservative transfer)
    scale = 0.7
    proposed = {
        dim: round(val * scale, 4)
        for dim, val in source_adj.items()
        if abs(val) > 0.001
    }
    if not proposed:
        return None

    magnitude = max(abs(v) for v in proposed.values())
    within_bounds = all(
        abs(v) <= LEARNING_BOUNDS.get(d, 0.05)
        for d, v in proposed.items()
    )

    # Confidence: based on number of source entries and their consistency
    confidence = min(0.90, TRANSFER_CONF_THRESHOLD + len(source_entries) * 0.05)

    requires_approval = (
        magnitude >= TRANSFER_AUTO_MAGNITUDE
        or confidence < TRANSFER_CONF_THRESHOLD
        or not within_bounds
    )

    justification = (
        f"Transfer from [{source_domain}] to [{target_domain}]: "
        f"{action}, based on {len(source_entries)} source entries. "
        f"Source adj: { {k:round(v,4) for k,v in source_adj.items() if abs(v)>0.001} }. "
        f"Proposed (×{scale}): { {k:round(v,4) for k,v in proposed.items()} }."
    )

    return TransferProposal(
        proposal_id=f"xfr_{uuid.uuid4().hex[:6]}",
        source_domain=source_domain,
        target_domain=target_domain,
        action=action,
        adjustment=proposed,
        confidence=round(confidence, 3),
        magnitude=round(magnitude, 4),
        within_bounds=within_bounds,
        requires_approval=requires_approval,
        justification=justification,
    )


# ─────────────────────────────────────────────
# govern_transfer (NEW in v12.3)
# ─────────────────────────────────────────────

@dataclass
class TransferVerdict:
    verdict:      Verdict
    proposal_id:  str
    reasons:      list[str]
    queued_for:   Optional[str] = None


def govern_transfer(proposal: TransferProposal) -> TransferVerdict:
    """
    Governance check for a TransferProposal.

    REJECT if:
    - out of bounds
    - domains are incompatible
    - would flip sign of epistemic action's eu effect

    QUEUE if:
    - confidence < TRANSFER_CONF_THRESHOLD
    - magnitude >= TRANSFER_AUTO_MAGNITUDE

    ACCEPT if all pass.
    """
    reasons = []

    # Compatibility (re-check at governance time)
    compatible = DOMAIN_COMPATIBILITY.get(proposal.source_domain, [])
    if proposal.target_domain not in compatible:
        return TransferVerdict(
            verdict=Verdict.REJECT,
            proposal_id=proposal.proposal_id,
            reasons=[f"domains {proposal.source_domain} → {proposal.target_domain} not compatible"],
        )

    # Bounds
    if not proposal.within_bounds:
        return TransferVerdict(
            verdict=Verdict.REJECT,
            proposal_id=proposal.proposal_id,
            reasons=[f"out of bounds: magnitude={proposal.magnitude:.4f}"],
        )

    # Epistemic sign flip guard
    if proposal.action in BASE_ACTION_MODEL:
        atype = BASE_ACTION_MODEL[proposal.action].get("type")
        if atype == "epistemic":
            eu_adj  = proposal.adjustment.get("epistemic_uncertainty", 0.0)
            base_eu = BASE_ACTION_MODEL[proposal.action]["effects"].get("epistemic_uncertainty", 0.0)
            if base_eu < 0 and (base_eu + eu_adj) > 0:
                return TransferVerdict(
                    verdict=Verdict.REJECT,
                    proposal_id=proposal.proposal_id,
                    reasons=[
                        f"would flip epistemic_uncertainty effect "
                        f"({base_eu:+.3f} + {eu_adj:+.3f} = {base_eu+eu_adj:+.3f})"
                    ],
                )

    # Confidence
    if proposal.confidence < TRANSFER_CONF_THRESHOLD:
        return TransferVerdict(
            verdict=Verdict.QUEUE,
            proposal_id=proposal.proposal_id,
            reasons=[f"confidence={proposal.confidence:.2f} < threshold={TRANSFER_CONF_THRESHOLD}"],
            queued_for="await more source evidence",
        )

    # Magnitude
    if proposal.magnitude >= TRANSFER_AUTO_MAGNITUDE:
        return TransferVerdict(
            verdict=Verdict.QUEUE,
            proposal_id=proposal.proposal_id,
            reasons=[f"magnitude={proposal.magnitude:.4f} >= auto_limit={TRANSFER_AUTO_MAGNITUDE}"],
            queued_for="operator approval",
        )

    reasons = [
        f"compatible domains",
        f"within bounds",
        f"confidence={proposal.confidence:.2f}",
        f"magnitude={proposal.magnitude:.4f} < {TRANSFER_AUTO_MAGNITUDE}",
    ]
    return TransferVerdict(
        verdict=Verdict.ACCEPT,
        proposal_id=proposal.proposal_id,
        reasons=reasons,
    )


# ─────────────────────────────────────────────
# apply_transfer (NEW in v12.3)
# ─────────────────────────────────────────────

def apply_transfer(
    verdict:  TransferVerdict,
    proposal: TransferProposal,
    model:    ContextualLearnedModel,
) -> Optional[LearningLogEntry]:
    """Apply a transfer only if verdict is ACCEPT."""
    if verdict.verdict != Verdict.ACCEPT:
        return None

    from memi_v112 import LearningProposal
    synth = LearningProposal(
        proposal_id=proposal.proposal_id,
        action=proposal.action,
        expected_effect={}, observed_effect={},
        delta=proposal.adjustment, confidence=proposal.confidence,
        proposed_adjustment=proposal.adjustment,
        adjustment_magnitude=proposal.magnitude,
        within_bounds=proposal.within_bounds,
        requires_approval=proposal.requires_approval,
        justification=proposal.justification,
    )
    return model.apply(synth, proposal.target_domain)


# ─────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────

def run():
    import random
    print("\n" + "=" * 72)
    print("M.E.M.I. v12.3 — Governed Cross-Context Transfer")
    print("=" * 72)

    model   = ContextualLearnedModel(BASE_ACTION_MODEL)
    session = ContextSession()

    # ── Phase 1: Build up learning in security domain ───────────
    print("\n── Phase 1: Run security domain to build learned model ──")
    state = WorldState(risk=0.80, reversibility=0.45, epistemic_uncertainty=0.60)
    buffers = BufferStore()
    traj    = Trajectory()
    security_session = ContextSession(initial_override="security")  # force security context

    for i in range(8):
        ctx_decision = security_session.step(state, i)
        context      = CONTEXTS[ctx_decision.active]
        best, _, _   = select_action_contextual_learned(state, model, ACTIONS, context)
        decision, _  = memi_decision(state, best)
        effective    = replan(decision, best) or best
        state_before = state.copy()
        state        = next_state(state, effective)

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
                wa    = buf.weighted_adjustment()
                avg_c = sum(p.confidence for p in buf.proposals) / buf.n()
                mag   = max(abs(v) for v in wa.values())
                from memi_v112 import LearningProposal
                synth = LearningProposal(
                    proposal_id=f"p1_{uuid.uuid4().hex[:4]}",
                    action=effective,
                    expected_effect={}, observed_effect={},
                    delta=wa, confidence=round(avg_c, 3),
                    proposed_adjustment=wa,
                    adjustment_magnitude=round(mag, 4),
                    within_bounds=all(abs(v) <= LEARNING_BOUNDS.get(d, 0.05) for d, v in wa.items()),
                    requires_approval=(mag >= 0.05 or avg_c < 0.7),
                    justification=f"phase1 n={buf.n()}",
                )
                buf.clear()
                verdict = govern_learning(synth)
                if verdict.verdict == Verdict.ACCEPT:
                    entry = model.apply(synth, ctx_decision.active)
                    print(f"  ✎ LEARNED [{ctx_decision.active}]  {effective}  "
                          f"adj={ {k:round(v,4) for k,v in entry.adjustment.items() if abs(v)>0.001} }")
        traj.add({"step": i, "action": effective, "decision": decision,
                  "risk": state.risk, "eu": state.epistemic_uncertainty,
                  "context": ctx_decision.active, "mode": "normal"})

    # Show what was learned
    print(f"\n  Learned in security:")
    ds = model.domain_summary("security")
    for action, data in ds.items():
        for dim, lv in data["learned"].items():
            if abs(lv) > 0.001:
                print(f"    {action:22s}  {dim:26s}  learned={lv:+.4f}")

    # ── Phase 2: Propose and govern transfers ────────────────────
    print(f"\n── Phase 2: Propose cross-context transfers ──")
    print(f"  (security → industrial: compatible)")
    print(f"  (security → clinical:   incompatible — clinical is isolated)")

    transfer_proposals = []
    transfer_verdicts  = []
    transfer_applied   = []

    for action in ACTIONS:
        for target in ("industrial", "clinical"):
            tp = propose_transfer(model, "security", target, action)
            if tp:
                tv = govern_transfer(tp)
                transfer_proposals.append(tp)
                transfer_verdicts.append(tv)

                v_sym = {"ACCEPT": "✓", "QUEUE": "⏸", "REJECT": "✗"}[tv.verdict.value]
                print(f"\n  {v_sym} {tp.summary()}")
                print(f"    {'; '.join(tv.reasons)}")
                if tv.queued_for:
                    print(f"    queued_for: {tv.queued_for}")

                entry = apply_transfer(tv, tp, model)
                if entry:
                    transfer_applied.append(entry)
                    print(f"    ✎ APPLIED to [{tp.target_domain}]  "
                          f"adj={ {k:round(v,4) for k,v in entry.adjustment.items() if abs(v)>0.001} }")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("Transfer summary")
    print("=" * 72)
    counts = {v: sum(1 for tv in transfer_verdicts if tv.verdict == v) for v in Verdict}
    print(f"  Proposals:  {len(transfer_proposals)}")
    print(f"  ACCEPT:     {counts[Verdict.ACCEPT]}")
    print(f"  QUEUE:      {counts[Verdict.QUEUE]}")
    print(f"  REJECT:     {counts[Verdict.REJECT]}")
    print(f"  Applied:    {len(transfer_applied)}")

    print(f"\n  Industrial learned after transfer:")
    ds_ind = model.domain_summary("industrial")
    if ds_ind:
        for action, data in ds_ind.items():
            for dim, lv in data["learned"].items():
                if abs(lv) > 0.001:
                    print(f"    {action:22s}  {dim:26s}  learned={lv:+.4f}  (transferred from security)")
    else:
        print("    (no adjustments transferred)")

    print(f"\n  Clinical: no transfers — isolated domain")
    ds_clin = model.domain_summary("clinical")
    print(f"    clinical learned_effects: {sum(1 for d in ds_clin.values() for v in d['learned'].values() if abs(v)>0.001)} non-trivial entries")

    print(f"\n  Audit log ({len(model.learning_log())} entries):")
    for e in model.learning_log():
        dom = getattr(e, "domain", "?")
        print(f"    [{e.entry_id}]  [{dom:12s}]  {e.action:22s}  "
              f"adj={ {k:round(v,4) for k,v in e.adjustment.items() if abs(v)>0.001} }")

    print(f"\n  Transfer is never automatic.")
    print(f"  Clinical domain is constitutionally isolated from transfer.")


if __name__ == "__main__":
    run()
