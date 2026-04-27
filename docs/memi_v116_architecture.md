# M.E.M.I. v11.6 — Admissibility Layer
## Architecture Note · April 2026

---

## Udgangspunkt

Three structural tensions were identified in v11.5:

1. **Epistemic loop** — the system needs knowledge to act, but can only gain knowledge by acting.
2. **Foresight validity** — foresight scores futures without checking if they are admissible.
3. **Governance after selection** — governance vetoes what evaluation already preferred.

None of these are bugs. They are places where the architecture is implicit.
v11.6 makes them explicit.

---

## The core question

> *What actually makes an action admissible?*

Currently admissibility is distributed across three layers:
- Epistemic state (epistemic_gating)
- Effect model conditions (avoid_when, effective_when)
- Governance thresholds (memi_decision)

None of these call it admissibility. None of them are consulted before scoring.

v11.6 makes admissibility a first-class condition — evaluated before foresight scores,
not applied after selection.

---

## Tension 1 — The epistemic loop

**The problem:**

```
knowledge → permit action
action    → produce knowledge
```

This is circular. If epistemic insufficiency blocks all actions,
the system can never gain the knowledge needed to act.

**The resolution (already implicit — now made explicit):**

```
Epistemic insufficiency must block:    operative and irreversible actions
Epistemic insufficiency must NOT block: epistemic actions
```

Epistemic actions (`collect_telemetry`, `monitor`) are unconditionally admissible
with respect to epistemic state. They exist precisely to break the loop.

**Bærende formulering:**

```
The system does not need knowledge to act.
It needs knowledge to act irreversibly.
```

**What changes in the code:**

`is_admissible(action, state)` explicitly separates action types:

```python
if action_type == "epistemic":
    return True   # always admissible — breaks epistemic loop

if action_type in ("operative", "irreversible"):
    if state.epistemic_uncertainty > EPISTEMIC_GATE:
        return False   # insufficient knowledge to act
    return True
```

---

## Tension 2 — Foresight scores inadmissible futures

**The problem:**

Foresight selects the highest-scoring candidate action.
Governance may then veto that action.

The system evaluates futures it is not allowed to enter.
This creates a split between evaluation and permission —
the system can strongly prefer something it is not allowed to do.

**The resolution:**

Admissibility check runs before scoring.
Inadmissible actions receive score = -∞ and are excluded from selection.

```python
for action in candidates:
    if not is_admissible(action, state):
        scores[action] = -inf
        continue
    # ... normal foresight evaluation
```

This means the foresight evaluation space and the governance permission space
are aligned. The system only strongly prefers what it is allowed to do.

**Bærende formulering:**

```
The system must not evaluate futures it is not allowed to enter.
```

**Important:** This is a pre-filter, not a replacement for governance.
`memi_decision()` still runs after selection — it remains the constitutional gate.
Admissibility makes evaluation self-consistent. Governance makes execution permitted.

---

## Tension 3 — Governance after selection

**The problem:**

The current flow is:

```
evaluate → select → memi_decision → (veto if not permitted)
```

Governance is downstream of evaluation. The system can invest heavily in
planning an action that is then blocked — producing constraint_streak
as a symptom of misaligned preference.

**The resolution:**

Partial governance moves into evaluation via `is_admissible()`.

This is not the same as removing the governance gate. It means:

```
is_admissible()    →  evaluation-time pre-filter (fast, approximate)
memi_decision()    →  execution-time constitutional gate (always runs)
```

The two layers have different roles:
- Admissibility makes evaluation honest about what is possible
- Governance makes execution safe

**Bærende formulering:**

```
The system should not strongly prefer what it is not allowed to do.
```

---

## Admissibility definition

An action is admissible if all of the following hold:

```
1. Type condition:
   - epistemic actions:    always admissible
   - operative actions:    admissible if epistemic_uncertainty <= EPISTEMIC_GATE
   - irreversible actions: admissible if epistemic_uncertainty <= 0.5
                           AND reversibility > 0.4
                           AND risk > 0.3 (action is justified)

2. Effect condition:
   - avoid_when(state) is False

3. Governance pre-check:
   - risk <= 0.9 OR reversibility >= 0.2  (VETO_2 would not fire)
   - NOT (sensor_dropout AND urgency >= 0.75 AND action == stabilize)  (VETO_1 would not fire)
   Note: this is a heuristic pre-check, not the constitutional gate.
   memi_decision() still runs after selection.
```

---

## What changes in the code

**New:** `is_admissible(action, state)` — single function that encodes all three conditions

**Changed:** `select_action_2step()` — check admissibility before scoring each candidate

**Changed:** `best_single_step()` — same admissibility pre-filter for follow-up selection

**Unchanged:** `memi_decision()` — still runs after selection (constitutional)
**Unchanged:** `epistemic_gating()` — preserved as additional guard
**Unchanged:** `ACTION_MODEL` — effect model unchanged
**Unchanged:** all learning layers (v11.2–v11.5)

---

## The relationship between layers

```
is_admissible()         →  evaluation-time (before scoring)
semantic_filter()       →  evaluation-time (penalties, not exclusion)
epistemic_gating()      →  evaluation-time (candidate restriction)
memi_decision()         →  execution-time (constitutional gate)
```

All four are needed. They operate at different points in the pipeline
and serve different purposes. Admissibility does not replace any of them.

---

## Architectural invariant (updated)

```
Foresight selects the proposal.
Foresight only evaluates admissible proposals.
Effect model explains why.
Epistemic state qualifies confidence.
Governance decides whether it is permitted.
Decision payload carries uncertainty beyond the system.
Learning proposals are governed decisions — not automatic updates.
```

---

## The boundary that matters

```
v11.5:
The system learns from consistency across experiences.

v11.6:
The system only plans over what it is allowed to do.
```

---

## Bærende sætninger

```
The system does not need knowledge to act.
It needs knowledge to act irreversibly.

The system must not evaluate futures it is not allowed to enter.

The system should not strongly prefer what it is not allowed to do.
```

---

*Next: v11.6 implementation*
