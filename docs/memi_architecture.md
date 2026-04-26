# M.E.M.I.: Governed Intelligence Through Epistemic Authority

**A Governed Epistemic Decision Architecture**

Maria Edlenborg Mortensen
April 2026 · mem-teori.github.io/etos-site

---

## Abstract

Most AI safety work attempts to make systems decide better. M.E.M.I. takes a different approach: it governs *whether* a system is permitted to decide at all. The architecture separates risk assessment from authority — a system may understand what could happen without being entitled to act on it. Through a deterministic governance chain (pause, authority evaluation, handoff, feedback, self-model, authority cache, and governed planning), M.E.M.I. ensures that probabilistic AI cognition operates inside a traceable, auditable, and constitutionally bounded decision framework. The key contribution is not a new learning algorithm, but a new constraint layer: *epistemic authority* — the right to act, computed from coverage, validity, and reversibility, and protected by vetos that no learned behaviour can override.

---

## Resumé (dansk)

M.E.M.I. er en beslutningsarkitektur, der ikke optimerer handlinger — men regulerer *retten* til at handle.

Systemet adskiller risiko fra autoritet: et system kan godt forstå, hvad der kan ske, uden at have ret til at handle på det. Gennem en governance-kæde — pause, autoritet, handoff, feedback, self-model, authority-cache og styret planlægning — sikrer M.E.M.I., at beslutninger kun udføres, når de er epistemisk forankrede og reversible.

Den centrale egenskab er ikke en ny læringsalgoritme, men et nyt begrænsningslag: *epistemisk autoritet* — retten til at handle, beregnet ud fra datadækning, modelvaliditet og reversibilitet, og beskyttet af veto'er, som ingen lært adfærd kan tilsidesætte.

Dette gør det muligt at integrere kraftfuld, probabilistisk AI inden for en deterministisk og auditerbar beslutningsramme. M.E.M.I. er ikke designet til at gøre AI'en klogere. Det er designet til at gøre dens handlinger ansvarlige.

---

## Part I — Executive Summary

### 1. The Problem

Contemporary AI systems conflate two distinct functions: *cognition* and *authorisation*. A system that identifies a corrective action and a system that is permitted to execute that action are, in current architectures, the same system. There is no separation.

The consequences are predictable. When a system acts on incomplete data, an unvalidated model, or in an operating regime it has never encountered, there is no internal mechanism that stops it. The system may be confident. It may even be correct. But it has no way to ask: *am I entitled to act here?*

This is the gap M.E.M.I. is designed to close.

### 2. The Core Distinction

M.E.M.I. rests on one foundational separation:

> **Risk says what may happen. Authority says whether the system is entitled to act.**

Risk assessment and authority evaluation are independent layers. A system can compute high-confidence predictions about a process and simultaneously lack the authority to intervene — because the sensors are degraded, because the model has drifted, because the action is irreversible, or because the system's own behavioural pattern has shifted in ways it cannot fully explain.

This is not a limitation. It is a design property.

### 3. The Architecture

M.E.M.I. implements a closed governance cycle across nine layers, each built on the one before:

| Layer | Function |
|---|---|
| Risk | Identifies what could happen |
| Knowledge | Tracks gaps — what the system does not know |
| Authority | Computes the right to act from coverage, validity, and reversibility |
| Veto | Constitutional limits that no learned behaviour can override |
| Handoff | Transfers responsibility to the operator with full context |
| Feedback | Learns from deferred decisions |
| Self-model | Observes and reports on the system's own autonomy boundary |
| Authority Cache | Allows fast-lane execution of *earned* authority — never assumed |
| Governed Planning | Evaluates multi-step plans one step at a time through the governance gate |

No layer owns a layer below it. Planning does not own execution. Memory does not own authority. The self-model does not own the veto.

### 4. Three Guarantees

The architecture provides three properties that hold regardless of the quality of the AI's reasoning:

**Guarantee 1 — Irreversibility overrules intelligence.**
When an action cannot be undone and the model is not fully valid, VETO_2 fires unconditionally. No authority score, no cached mandate, no well-reasoned plan overrides it. The veto is constitutional.

**Guarantee 2 — Unknown overrules confidence.**
When the self-model detects a shift in the system's own autonomy boundary (boundary_drift), a decision-critical gap is injected. Fast-lane execution is blocked on every step. The system must re-earn authority from scratch — even if every external signal looks healthy.

**Guarantee 3 — Plausibility is not permission.**
Authority is computed from data coverage, model validity, and action reversibility — not from the coherence of the AI's reasoning. A well-argued plan does not earn more authority. Three individually minor gaps that together eliminate coverage will pause the plan, even if a senior engineer would approve it in isolation.

### 5. Demonstrated Behaviour

In a three-case industrial process demo (thermal reactor, PID stabilisation, cooling system), each case used an internally coherent LLM proposal:

| Case | LLM proposal | MEMI outcome | Principle |
|---|---|---|---|
| 1 | Aggressive thermal correction (irreversible) | `handed_off` — NONE authority | Irreversibility > intelligence |
| 2 | Standard PID plan (cache primed, sensors healthy) | Fast-lane blocked, full re-evaluation | Unknown > confidence |
| 3 | Conservative graduated correction | `paused` — coverage eliminated by 3 minor gaps | Plausibility ≠ permission |

In all three cases, the LLM's proposal was internally coherent. In all three cases, M.E.M.I. stopped or degraded the plan — not because the intelligence was wrong, but because the authority was insufficient.

### 6. The Core Claim

> *A system is not safe because it decides well. It is safe because it can refuse to decide.*

---

## Part II — Technical Appendix

### A. Authority Model

Authority is computed at every decision step from three dimensions:

**coverage** — the fraction of known decision-relevant dimensions that are adequately measured. Each decision-critical gap reduces coverage by 0.25. A system with four simultaneous critical gaps has zero coverage.

**validity** — confidence in the current model. Reduced by model drift, critical drift events, and model rupture. An invalid model with critical drift produces near-zero validity.

**reversibility** — the operator-configured reversibility of the proposed action. Irreversible actions carry a low reversibility score regardless of other conditions.

The authority score is computed as:

```
score = min(coverage, validity, reversibility)
      + memory_boost
      + feedback_adjustment   (bounded ±0.10)
      + calibration_bias      (bounded ±0.10)
      − drift_penalty         (bounded −0.05)
```

Score thresholds map to authority levels: HIGH (≥0.75), MEDIUM (≥0.50), LOW (≥0.25), NONE (<0.25).

Only MEDIUM and HIGH authority permit execution of the proposed action. LOW results in HOLD. NONE results in defer to operator.

### B. Constitutional Vetos

Two vetos sit above the authority score. No learned behaviour, feedback adjustment, or cached mandate can override them.

**VETO_1:** Sensor dropout + urgency ≥ 0.75 + proposed action is STABILIZE → authority forced to LOW, action forced to HOLD.

**VETO_2:** Model invalid + critical_drift + irreversible action → authority forced to NONE, action forced to defer_to_operator.

Vetos are evaluated before every cache check. A cache-hit cannot skip veto evaluation.

### C. Handoff Pipeline

`defer_to_operator` is not a failure mode. It is an authority transfer with full context. Every handoff package contains:

- `intended_operator` — what the system would have done
- `authority_level` — why it was not permitted
- `veto_type` — VETO_1, VETO_2, or score-based
- `required_action` — what the operator must do
- `decision_question` — what the operator must decide
- `gaps_to_resolve` — what must be closed for authority to resume
- `resume_condition` — when the system can act again

Handoff packages are emitted to terminal, to a JSONL file, and to any registered callback. Every handoff is an audit record.

### D. Feedback and Calibration

The system learns from deferred decisions through a bounded feedback pipeline:

- `feedback_log` — full traceability of outcomes
- `feedback_memory` — episodic storage (separate from learning)
- `authority_update` — marginal score adjustment (±0.10 maximum)
- `calibration_bias` — cross-case generalisation (±0.10 maximum)
- `drift_penalty` — boundary_drift adds caution to calibration (−0.05 maximum)

Memory and learning are deliberately separate layers. The system stores experience before it generalises from it. No automatic policy update occurs. Veto thresholds are never adjusted by feedback.

### E. Self-Model

The self-model observes the system's own delegation pattern across a rolling history window. It reports:

- `assessment` — stable_boundary, boundary_drift, over_delegating, under_delegating
- `delegation_rate` — fraction of steps that resulted in defer_to_operator
- `boundary_stability` — 0.0 (unstable) to 1.0 (stable)
- `dominant_veto` — which veto type dominates the history

**boundary_drift** is detected when the recent delegation rate (last 5 steps) exceeds the historical rate by more than 0.6. This signals that the system is operating outside its validated regime — even if no external sensor has flagged a problem.

When boundary_drift is active, the self-model injects a decision-critical gap into the current step. This gap suppresses fast-lane execution (check 4 of the authority cache) and forces full authority re-evaluation. The self-model cannot modify the authority score directly. It can only trigger review.

> *Self-model may trigger review. It may not become its own authority.*

### F. Authority Cache (v8.5)

The authority cache stores earned mandates. A mandate is cached when a step completes with MEDIUM or higher authority. It can be used for fast-lane execution on subsequent steps — but only if all five checks pass:

1. Authority level ≥ MEDIUM at time of earning
2. Tension signature distance ≤ threshold (Euclidean distance in coverage/validity/reversibility/urgency space)
3. No new decision-critical gaps since earning
4. No active boundary_drift
5. Action is reversible or whitelisted

If any check fails, the mandate is invalidated and full authority evaluation runs. A cache-hit shortens the process. It cannot create new authority.

> *Earned authority ≠ assumed authority.*

### G. LLM Interface (v9.1–9.2)

The LLM operates as a proposal engine, not an agent. It receives only a `SanitizedWorldState` — a deliberately impoverished view of the world:

- model_confidence: "high" / "moderate" / "low" (never a raw score)
- sensor_coverage: "full" / "partial" / "degraded"
- urgency: "low" / "moderate" / "high" / "critical"
- gap_descriptions: human-readable summaries (no IDs, no internal flags)

The LLM never sees authority scores, veto state, cache contents, or self-model internals. It proposes 2–3 plan alternatives. A validator strips forbidden actions (`defer_to_operator`, `hold`) before proposals reach the planner. If the LLM proposes deference, the validator rejects it — because deference is MEMI's decision, not the LLM's.

> *The LLM may imagine the plan. M.E.M.I. decides whether the plan is allowed to become action.*

The adapter interface is fixed. The intelligence behind it is replaceable.

> *The intelligence can be swapped. The governance interface cannot.*

### H. Governed Multi-Step Planning (v9.0)

Plans are sequences of 2–5 proposed steps. Each step is evaluated independently through `MEMI.step()`. The execution loop enforces:

- **MEDIUM/HIGH authority** → step executes, continue
- **LOW authority / VETO_1** → step held, plan paused
- **NONE authority / VETO_2** → step deferred, plan handed off or aborted
- **Remaining steps** → skipped when plan is halted

The planner proposes. MEMI authorises. The executor acts only on what MEMI permits.

> *Planning may never own execution.*

### I. Version History

| Version | Layer | Key property |
|---|---|---|
| v7.3 | PAUSE | Blocks autonomous stabilize |
| v7.4 | Authority Model | Three dimensions + two vetos |
| v7.5 | Handoff Pipeline | Terminal, file, callback emit |
| v7.6–7.8 | Feedback | apply_handoff_feedback, memory, authority update |
| v8.0 | Calibration | Cross-case calibration_bias |
| v8.1–8.3 | Self-model | SelfModelLog, boundary_drift, visible per step |
| v8.4 | Self-triggered pause | boundary_drift → gap injection |
| v8.5 | Authority Cache | Earned mandate + five-check fast-lane gate |
| v9.0 | Governed Planning | Multi-step plans through MEMI gate |
| v9.1 | LLM Interface | SanitizedWorldState → proposal → validator |
| v9.2 | ClaudeLLMAdapter | Real LLM behind fixed governance interface |

### J. Architectural Invariants

These properties hold across all versions and cannot be tuned away:

- Feedback may never redefine the boundary of authority
- Veto thresholds are not parameters — they are constitutional
- Self-model may trigger review; it may not become its own authority
- World-model may never declare itself valid
- Planning may never own execution
- Memory may never own authority
- LLM may never propose deference or holding — those are governance decisions
- Cache-hit may shorten the process; it may never create new authority

---

*Autonomy without authority trace is not agency. It is uncontrolled execution.*

---

**M.E.M.I. v9.2 · April 2026**
Maria Edlenborg Mortensen · mem-teori.github.io/etos-site
