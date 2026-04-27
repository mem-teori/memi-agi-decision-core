# M.E.M.I. v11.x — Governed Learning
## Architecture Note · April 2026

---

## Bærende sætning

    A system is not aligned because it learns.
    It is aligned because it decides what it is allowed to learn.

---

## The problem

v11.1 closes the decision loop: uncertainty travels with the decision.

But the system does not change.

Every run of v11.x starts with the same ACTION_MODEL, the same thresholds,
the same evaluation function. The system cannot improve its predictions
based on what it observes. It cannot discover that `collect_telemetry`
reduces risk faster under certain conditions. It cannot notice that
its foresight was wrong and adjust.

This is the gap:

    The system observes outcomes.
    It does not update its model from them.

---

## The shift

From:

    decision → action → outcome → (forgotten)

To:

    decision → action → outcome
                           ↓
                    compare (expected vs observed)
                           ↓
                    learning proposal
                           ↓
                    governance check
                           ↓
                    accept / reject / defer

Learning is not automatic. It is a governed decision.

---

## What the system may learn

Three bounded domains:

**1. Effect magnitudes**
The actual effect of an action on the world may differ from the model.
`collect_telemetry` may reduce risk by 0.09 in practice, not 0.12.
The system may propose adjusting the effect coefficient — within bounds.

**2. Threshold calibration**
`EPISTEMIC_GATE = 0.7` was set by design.
Observed data may suggest the gate fires too early or too late.
The system may propose a bounded adjustment — not removal.

**3. Condition validity**
`avoid_when: lambda s: s.risk < 0.3` may be wrong for certain state patterns.
The system may flag a condition as "frequently violated without consequence"
and propose weakening it — subject to governance approval.

---

## What the system may NOT learn

These are constitutional. Learning cannot touch them.

**1. Governance authority thresholds**
VETO_1 and VETO_2 conditions are not parameters.
Learning cannot adjust when a veto fires.

**2. The uncertainty_travels flag**
`EpistemicPayload.uncertainty_travels = True` is invariant.
Learning cannot make the system stop propagating uncertainty.

**3. The epistemic gating principle**
The system must always prefer epistemic actions when uncertainty is high.
The gate threshold may shift marginally. The principle cannot be removed.

**4. The governance gate itself**
`memi_decision()` is not learnable.
Authority evaluation is not a function to be optimised.

---

## The learning proposal structure

```python
@dataclass
class LearningProposal:
    step_id:              str
    action:               str
    expected_effect:      dict    # from ACTION_MODEL at decision time
    observed_effect:      dict    # measured from state_before → state_after
    delta:                dict    # observed - expected
    confidence:           float   # how reliable is this observation?
    proposed_adjustment:  dict    # what change is proposed
    adjustment_magnitude: float   # how large is the proposed change? (0–1)
    within_bounds:        bool    # is the proposal within allowed range?
    requires_approval:    bool    # must governance approve before applying?
    justification:        str
```

---

## The governance check for learning

Learning proposals pass through a governance layer before any update:

```
LearningProposal
    → magnitude_check()      — is the proposed change within bounds?
    → consistency_check()    — does it conflict with architectural invariants?
    → confidence_check()     — is there enough evidence to justify the update?
    → approval_decision()    — auto-apply / queue for operator / reject
```

Auto-apply conditions (no operator needed):
- magnitude < 0.05 (small adjustment)
- confidence >= 0.7
- within_bounds = True
- does not touch veto conditions

Operator approval required:
- magnitude >= 0.05
- or confidence < 0.7
- or touches threshold calibration

Reject immediately:
- proposed change would disable epistemic gating
- proposed change would modify veto conditions
- within_bounds = False

---

## What changes in the code

**New:** `LearningProposal` dataclass

**New:** `observe_outcome(state_before, state_after, action, expected)` — measures actual effect

**New:** `build_learning_proposal(action, expected, observed, state)` — constructs proposal

**New:** `govern_learning(proposal)` — checks bounds, confidence, invariants → accept/queue/reject

**New:** `apply_learning(proposal, action_model)` — applies approved adjustment to ACTION_MODEL

**Extended:** `ACTION_MODEL` entries gain `learned_effects` dict alongside `effects` — the base
             values are never overwritten, only the learned adjustment is stored separately

**Extended:** `simulate_from_model()` uses `effects + learned_adjustment` for projection

**Unchanged:** `memi_decision()` — governance not affected
**Unchanged:** `epistemic_gating()` — gating principle not affected
**Unchanged:** `VETO_1 / VETO_2` conditions — constitutional
**Unchanged:** `uncertainty_travels` flag — invariant

---

## The architectural invariant (extended)

    Foresight selects the proposal.
    Effect model explains why.
    Epistemic state qualifies confidence.
    Governance decides whether it is permitted.
    Decision payload carries uncertainty beyond the system.
    Learning proposals are governed decisions — not automatic updates.

---

## The distinction that matters

Classical learning:

    outcome → update model

Governed learning:

    outcome → compare with expectation
           → propose adjustment
           → governance decides
           → update (if permitted)

The model does not learn. The system proposes learning.
Governance decides what the system is allowed to learn.

---

## Failure mode to avoid

If learning updates `memi_decision()` thresholds based on past approvals,
the system begins to optimise for getting approved — not for being correct.

This is the alignment failure mode.

Guard: governance thresholds are read-only to the learning layer.
The learning layer can only write to `ACTION_MODEL[action]["learned_effects"]`.

---

## Version plan

    v11.2  observe_outcome() + build_learning_proposal()
           → system measures what actually happened vs what it predicted

    v11.3  govern_learning()
           → proposals pass through governance before applying

    v11.4  apply_learning() + learned_effects in ACTION_MODEL
           → approved adjustments stored separately from base effects

    v11.5  audit trail for learning
           → every accepted/rejected proposal is logged with reason

---

## Bærende sætning (gentaget)

    A system is not aligned because it learns.
    It is aligned because it decides what it is allowed to learn.

---

*Next: v11.2 implementation*
