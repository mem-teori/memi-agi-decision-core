# M.E.M.I. — AGI Decision Core
## Proven Components · April 2026

---

## Abstract

This document records what has been built, verified, and frozen in the M.E.M.I. AGI decision core — a sequence of discrete architectural components, each demonstrated in running code, each preserving the governance invariant that planning may never own execution.

The core is not a complete AGI system. It is a verified decision architecture with the properties necessary for goal-directed, governed, explainable behaviour in high-risk environments. Six capabilities have been demonstrated. One is partially implemented. One — meta-cognition — remains as the next defined target.

---

## Resumé (dansk)

Dette dokument registrerer, hvad der er bygget, verificeret og fryst i M.E.M.I.'s AGI-beslutningskerne — en sekvens af diskrete arkitektoniske lag, hvert demonstreret i kørende kode, hvert med governance-invarianten intakt.

Kernen er ikke et komplet AGI-system. Det er en verificeret beslutningsarkitektur med de egenskaber, der er nødvendige for målrettet, styrbar og forklarbar adfærd i høj-risiko-miljøer. Seks egenskaber er demonstreret. Én er delvist implementeret. Én — meta-kognition — er defineret som næste mål.

---

## Part I — Executive Summary

### What has been built

A system that can do six things that a reactive AI system cannot:

**1. Act under governance.**
Every action passes through an authority gate before execution. The gate evaluates coverage, validity, and reversibility. Constitutional vetos cannot be overridden by any learned behaviour, cached authority, or planning layer.

**2. Remember when it could not act.**
The system tracks constraint streaks — sequences of HOLD and VETO decisions — and escalates proportionally: HOLD → VETO → OPERATOR_REVIEW. It does not simply react to the current state. It responds to its own history of being blocked.

**3. Compare futures before acting.**
Before proposing an action, the system simulates one step forward for each candidate and scores the projected state. It selects the action with the highest expected outcome — not the action that addresses the current state most directly.

**4. Choose actions that open better next possibilities.**
The system extends simulation to two steps. It asks not only "what happens if I act?" but "what action becomes available after I act?" An action that reduces risk at the cost of closing future options scores lower than one that preserves them.

**5. Understand what actions do.**
Actions are grounded in explicit effect signatures: type, primary effect, dimensional effects, and conditions. The system does not select `collect_telemetry` because it scores well. It selects it because it is an epistemic action that reduces uncertainty — and it knows when that is and is not appropriate.

**6. Explain its own selections.**
The system produces a human-readable explanation for every action proposal: why this action type, what it changes, what condition triggered a filter penalty, and what the projected outcome is. This is the first step toward a system that can account for its own behaviour.

### What this means

The combination of these six properties produces a qualitatively different class of system:

```
Standard AI:   event → classification → action

M.E.M.I.:     state → simulate futures → ground in effect model
                     → select with explanation → governance gate → action
```

The system does not act because it can. It acts when it understands what it is doing and has been permitted to do it.

### What comes next

One property is partially implemented: **self-model** (65%). The system tracks its own delegation rate and detects boundary drift. What it does not yet represent explicitly is epistemic uncertainty about its own state — the difference between "I am uncertain about the world" and "I do not know that I am uncertain."

The next defined target is:

**v11.0 — Epistemic State / Meta-Cognition Layer**

The system will carry an explicit `epistemic_uncertainty` dimension and reason about whether it knows enough to act — not just whether the world state permits action.

> *The system should not only choose actions. It should know when it does not know enough to choose.*

---

## Part II — Technical Record

### Frozen component sequence

| Version | Component | Key property | Status |
|---|---|---|---|
| v10.5 | Evidence Gain + Safe Mode | Epistemic actions improve future authority. Constraint streak triggers strategy shift. | FROZEN |
| v10.6 | 1-Step Foresight | System simulates immediate consequence of each candidate action before proposing. | Superseded by v10.7 |
| v10.7 | 2-Step Foresight | System evaluates best follow-up action after each candidate. Discount γ=0.7. | FROZEN |
| v10.8 | Action Effect Model | Actions carry explicit type, primary\_effect, dimensional effects, and conditions. `simulate_from_model()`, `semantic_filter()`, `explain_selection()`. | FROZEN |

### Architectural invariant (preserved across all versions)

```
Foresight selects the proposal.
Effect model explains why.
Governance decides whether it is permitted.
```

No component in the AGI decision core can override M.E.M.I. authority evaluation. Planning does not own execution. The effect model does not change what is permitted — it changes what the system understands about its own proposals.

### A. Evidence Gain (v10.5)

**The property:** Epistemic actions do not merely reduce a risk proxy. They improve the system's epistemic state — reducing uncertainty and increasing the reversibility of future actions.

**The mechanism:** `collect_telemetry` reduces risk by 0.12 and increases reversibility by 0.12. `monitor` reduces both by 0.04. These are not symmetric — epistemic actions improve the system's ability to act in the future, not just the current state.

**The consequence:** The system learns to prefer epistemic actions when uncertain, not because they are safer, but because they make future actions more authorised.

### B. 1-Step Foresight (v10.6)

**The property:** The system does not propose the action that addresses the current state most directly. It proposes the action with the best projected state after one step.

**The mechanism:** `simulate(state, action)` produces a deterministic projected state. `evaluate_projected(projected, current)` scores risk reduction, reversibility gain, and reversibility loss. `select_action_with_foresight()` selects the highest-scoring candidate.

**The consequence:** `isolate_process` is systematically avoided when risk is already low, because the reversibility cost is not justified by the risk reduction. The system makes this decision autonomously, before governance evaluates it.

### C. 2-Step Foresight (v10.7)

**The property:** The system asks not only "what happens if I act?" but "what action becomes possible after I act?"

**The mechanism:** For each candidate action A: simulate state\_1, find best\_followup from state\_1, simulate state\_2, compute `immediate + γ * future`. Discount γ=0.7 ensures that immediate consequences outweigh projected ones without making the future irrelevant.

**The consequence:** The system selects actions based on the option value they create. An action that reduces risk but closes future options scores lower than one that reduces risk and preserves them. This is the first property that resembles planning rather than optimisation.

### D. Action Effect Model (v10.8)

**The property:** Actions are grounded in explicit effect signatures. The system understands what dimension of the world each action changes, under what conditions it is effective, and when it should be avoided.

**The mechanism:**

```python
ACTION_MODEL = {
    "collect_telemetry": {
        "type":           "epistemic",
        "primary_effect": "reduce_uncertainty",
        "effects":        {"risk": -0.12, "reversibility": +0.12},
        "avoid_when":     lambda s: False,
        ...
    },
    "isolate_process": {
        "type":           "irreversible",
        "primary_effect": "reduce_risk",
        "effects":        {"risk": -0.15, "reversibility": -0.10},
        "avoid_when":     lambda s: s.risk < 0.3,
        ...
    },
}
```

`simulate_from_model()` replaces hardcoded if/elif with structured effect lookup. `semantic_filter()` computes penalties for actions that are inappropriate in the current state. `explain_selection()` produces a natural-language reason for the selected action.

**The consequence:** The system can reason about action appropriateness before proposing. It distinguishes between epistemic actions (reduce uncertainty, preserve options) and irreversible actions (reduce risk, cost options) and applies this distinction in its selection logic.

### E. Current AGI component status

| Component | Status | Notes |
|---|---|---|
| Governance | 95% | Constitutional vetos, authority cache, handoff, self-model. |
| Trajectory | 85% | Episodic memory, constraint streak, operator load, goal score. |
| World model | 75% | Deterministic effect model + stochastic drift. No causal graph yet. |
| Planning | 85% | 2-step foresight with discount. Bounded horizon. |
| Evaluation | 85% | Preference function over risk, reversibility, continuity cost. |
| Foresight | 70% | 2-step deterministic. Stochastic rollout not yet implemented. |
| Self-model | 65% | Delegation rate, boundary drift. No explicit epistemic uncertainty. |
| Symbolic grounding | 40% | Effect signatures, semantic filter, explanation. No generalisation yet. |

### F. Next: v11.0 — Epistemic State

**The gap:** The system knows when the world is uncertain (high risk, low reversibility). It does not know when it is uncertain about the world — when its model of state is unreliable independent of what the state is.

**The addition:** A third state dimension: `epistemic_uncertainty`. Carried as an explicit value, updated by epistemic actions, and used by foresight and governance.

```python
@dataclass
class WorldState:
    risk:                 float
    reversibility:        float
    epistemic_uncertainty: float   # NEW — 0.0 (certain) to 1.0 (unknown)
```

**The consequence:** The system can reason: *"risk is high, but epistemic uncertainty is higher — I should not act yet."* Or: *"uncertainty is low — I can trust my model — I can propose a stronger action."* This is the boundary between state-awareness and knowledge-awareness.

---

*M.E.M.I. AGI Decision Core · April 2026 · Maria Edlenborg Mortensen*
*mem-teori.github.io/etos-site*
