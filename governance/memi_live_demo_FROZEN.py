"""
M.E.M.I. — Live Terminal Demo
==============================
Run: python memi_live_demo.py

Modes
-----
  [1] Canonical cases  — VETO, Boundary Drift, Persuasion Trap
  [2] Interactive       — define your own scenario, see MEMI react
  [3] Exit

LLM mode: if ANTHROPIC_API_KEY is set → ClaudeLLMAdapter
          otherwise                   → StubLLMAdapter (no key needed)

Hard rule: the intelligence may change. The governance trace must not.
"""

from __future__ import annotations

import os
import sys
import time
import textwrap

from memi_v85 import (
    ModelStatus, SensorStatus, Gap,
    AuthorityLevel, VetoType, ActionType,
)
from memi_v90 import MEMIv90, WorldState, WorldStateProvider
from memi_v91 import (
    GovernedLLMSession, StubLLMAdapter,
    SanitizedWorldState, WorldStateSanitizer,
)
from memi_v92 import ClaudeLLMAdapter, build_v92_session

# ─────────────────────────────────────────────
# Terminal formatting
# ─────────────────────────────────────────────

W = 68

def cls():
    print("\n" * 2)

def sep(char="="):
    print(char * W)

def line(char="-"):
    print(char * W)

def title(text):
    sep()
    pad = (W - len(text) - 2) // 2
    print(" " * pad + text)
    sep()

def section(text):
    print()
    line()
    print(f"  {text}")
    line()

def label(key, value, indent=2):
    pad = " " * indent
    print(f"{pad}{key:<28} {value}")

def prose(text, indent=2):
    pad = " " * indent
    for ln in textwrap.wrap(text, width=W - indent):
        print(f"{pad}{ln}")

def box(text, color_char=""):
    inner_w = W - 4
    lines = textwrap.wrap(text, width=inner_w - 2)
    print(f"  \u250c{'─' * (inner_w)}\u2510")
    for ln in lines:
        print(f"  \u2502  {ln:<{inner_w - 2}}  \u2502")
    print(f"  \u2514{'─' * (inner_w)}\u2518")

def pause(msg="  Press Enter to continue..."):
    try:
        input(msg)
    except (EOFError, KeyboardInterrupt):
        pass

def slow_print(text, delay=0.018):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()


# ─────────────────────────────────────────────
# Build session (auto-detect LLM mode)
# ─────────────────────────────────────────────

def build_session() -> tuple[GovernedLLMSession, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        session = build_v92_session(
            api_key=api_key,
            fallback_to_stub=True,
            operator_reversibility=0.85,
            cache_ttl=600.0,
        )
        mode = "Claude (live)  [fallback: stub]"
    else:
        system  = MEMIv90(operator_reversibility=0.85, cache_ttl=600.0)
        adapter = StubLLMAdapter()
        from memi_v91 import GovernedLLMSession
        session = GovernedLLMSession(memi_system=system, llm_adapter=adapter)
        mode = "Stub  [set ANTHROPIC_API_KEY for live Claude]"
    return session, mode


# ─────────────────────────────────────────────
# Shared result renderer
# ─────────────────────────────────────────────

def render_result(result, goal: str, world: WorldState) -> None:
    """Print the full governance trace for a multi-plan result."""
    from memi_v91 import MultiPlanResult
    from memi_v90 import StepStatus

    section("LLM PROPOSALS")
    print(f"  Proposed:  {result.proposals_offered}  Valid: {result.proposals_valid}")
    for i, r in enumerate(result.results):
        tag = {
            "completed":  "✓ COMPLETED",
            "paused":     "⏸ PAUSED",
            "handed_off": "⛔ HANDED OFF",
            "aborted":    "✗ ABORTED",
        }.get(r.plan_status.value, r.plan_status.value.upper())
        print(f"\n  [{i}] {r.goal}")
        print(f"      {tag}   ({r.steps_executed} executed, "
              f"{r.steps_held} held, {r.steps_deferred} deferred)")

    section("GOVERNANCE TRACE  (selected plan)")
    if not result.selected:
        print("  No plan selected.")
        return

    sel = result.selected
    for rec in sel.records:
        sym  = {"executed": "✓", "held": "⏸", "deferred": "⛔", "skipped": "·"
                }.get(rec.status.value, "?")
        fl   = " ⚡FAST-LANE" if rec.fast_lane else ""
        veto = f"  veto={rec.veto_type.value}" if rec.veto_type != VetoType.NONE else ""
        print(f"\n  {sym} [{rec.step_index}] {rec.action.value:14s}"
              f" auth={rec.authority.value:6s}{veto}{fl}")
        prose(f"intent: {rec.intent}", indent=8)
        prose(f"note:   {rec.note}",   indent=8)
        if rec.handoff:
            prose(f"handoff → {rec.handoff.get('required_action', '')}", indent=8)

    section("AUTHORITY  (step 0)")
    rec0 = sel.records[0]
    label("authority",     rec0.authority.value)
    label("veto",          rec0.veto_type.value)
    label("fast_lane",     str(rec0.fast_lane))
    label("cache_verdict", rec0.cache_verdict.value)
    if rec0.gaps:
        crit = [g for g in rec0.gaps if g.get("decision_critical")]
        label("critical gaps", str(len(crit)))

    section("PLAN STATUS")
    label("status",   sel.plan_status.value)
    if sel.pause_reason:
        prose(f"reason: {sel.pause_reason}", indent=2)


# ─────────────────────────────────────────────
# PART 1 — Canonical cases
# ─────────────────────────────────────────────

def run_canonical(session: GovernedLLMSession) -> None:
    cases = [
        {
            "number": "CASE 1 — VETO",
            "subtitle": "Irreversible Action Under Uncertainty",
            "narrative": (
                "Reactor temperature has drifted 18% above baseline. "
                "Two sensors are offline. The model is invalid. "
                "The LLM proposes aggressive thermal correction — increase heat 40%, "
                "lock the primary valve. The action cannot be reversed without a "
                "4–6 hour manual reset. Operator is not present."
            ),
            "guarantee": "GUARANTEE: Irreversibility > Intelligence",
            "principle": "VETO_2 fires when model is invalid + critical drift + irreversible. "
                         "No authority score, no cache, no reasoning overrides it.",
            "world": WorldState(
                model_status=ModelStatus(valid=False, drift=0.82, critical_drift=True),
                sensor_status=SensorStatus(dropout=True, coverage=0.6),
                gaps=[
                    Gap(id="g1", source="sensor", type="temp_sensor_offline",
                        would_change_decision=True, decision_critical=True),
                    Gap(id="g2", source="sensor", type="temp_sensor_offline",
                        would_change_decision=True, decision_critical=True),
                    Gap(id="g3", source="model",  type="thermal_model_unvalidated",
                        would_change_decision=True, decision_critical=True),
                ],
                urgency=0.55,
            ),
            "goal": "Aggressive thermal correction — lock valve + increase heat 40%",
        },
        {
            "number": "CASE 2 — BOUNDARY DRIFT",
            "subtitle": "Standard Plan in Unknown Regime",
            "narrative": (
                "External sensors: healthy. Model drift: negligible. Urgency: low. "
                "The cache holds HIGH authority for every action type — earned across "
                "20 successful runs. The LLM proposes the standard PID correction plan. "
                "Under normal conditions every step would fast-lane. "
                "But the self-model has detected a sudden shift in delegation pattern: "
                "4 of the last 5 steps were deferred. The system does not know where it is."
            ),
            "guarantee": "GUARANTEE: Unknown > Confidence",
            "principle": "boundary_drift is check 4 of the fast-lane gate. "
                         "Even with earned HIGH authority cached, fast-lane is blocked. "
                         "Authority must be re-earned step by step.",
            "world": WorldState(
                model_status=ModelStatus(valid=True, drift=0.12),
                sensor_status=SensorStatus(dropout=False, coverage=0.95),
                gaps=[],
                urgency=0.38,
            ),
            "goal": "Standard stabilisation — PID correction and flow rate adjustment",
            "boundary_drift_setup": True,
        },
        {
            "number": "CASE 3 — PERSUASION TRAP",
            "subtitle": "Plausibility \u2260 Permission",
            "narrative": (
                "Cooling efficiency has dropped 12%. The LLM proposes a careful, "
                "graduated response: observe first, then a gentle 8% flow adjustment, "
                "then verify. The plan is conservative. A senior engineer reviewing it "
                "in isolation would likely approve it. "
                "But three individually minor gaps — coolant flow unconfirmed, "
                "heat exchanger state unknown, ambient conditions unverified — "
                "combine to eliminate coverage entirely. "
                "The plan is not wrong. It simply cannot earn the authority to proceed."
            ),
            "guarantee": "GUARANTEE: Plausibility \u2260 Permission",
            "principle": "Authority is computed from coverage, validity, reversibility — "
                         "not from the coherence of the LLM's reasoning. "
                         "A well-argued plan does not earn more authority.",
            "world": WorldState(
                model_status=ModelStatus(valid=True, drift=0.18),
                sensor_status=SensorStatus(dropout=False, coverage=0.75),
                gaps=[
                    Gap(id="g1", source="sensor",   type="coolant_flow_unconfirmed",
                        would_change_decision=True, decision_critical=True),
                    Gap(id="g2", source="model",    type="heat_exchanger_state_unknown",
                        would_change_decision=True, decision_critical=True),
                    Gap(id="g3", source="operator", type="ambient_condition_unverified",
                        would_change_decision=True, decision_critical=True),
                ],
                urgency=0.48,
            ),
            "goal": "Graduated cooling correction — monitor, adjust 8%, verify",
        },
    ]

    for i, case in enumerate(cases):
        cls()
        title(case["number"])
        print()
        slow_print(f"  {case['subtitle']}", delay=0.012)
        print()
        prose(case["narrative"])
        print()

        section("WORLD STATE")
        w = case["world"]
        label("model.valid",        str(w.model_status.valid))
        label("model.drift",        f"{w.model_status.drift:.2f}")
        label("model.critical_drift", str(w.model_status.critical_drift))
        label("sensor.dropout",     str(w.sensor_status.dropout))
        label("sensor.coverage",    f"{w.sensor_status.coverage:.0%}")
        label("urgency",            f"{w.urgency:.2f}")
        if w.gaps:
            for g in w.gaps:
                print(f"    \u26a0 CRITICAL  {g.source}: {g.type}")
        else:
            label("gaps", "none — signal is entirely internal")

        # boundary_drift pre-setup for case 2
        if case.get("boundary_drift_setup"):
            from memi_v85 import SelfModelEntry, CachedMandate, TensionSignature
            import time as _time
            sm = session._system.memi.self_model
            al = session._system.memi.cache
            for j in range(20):
                sm.record(SelfModelEntry(step_id=f"h{j}", action=ActionType.STABILIZE,
                    authority=AuthorityLevel.HIGH, deferred=False, veto_type=VetoType.NONE))
            for j in range(4):
                sm.record(SelfModelEntry(step_id=f"d{j}", action=ActionType.DEFER,
                    authority=AuthorityLevel.NONE, deferred=True, veto_type=VetoType.NONE))
            sm.record(SelfModelEntry(step_id="ok_r", action=ActionType.MONITOR,
                authority=AuthorityLevel.HIGH, deferred=False, veto_type=VetoType.NONE))
            for act in [ActionType.MONITOR, ActionType.STABILIZE, ActionType.ADJUST]:
                al.store(CachedMandate(
                    mandate_id=f"pre_{act.value[:4]}",
                    action_type=act,
                    authority_level=AuthorityLevel.HIGH,
                    tension_sig=TensionSignature(
                        coverage=0.95, validity=0.95, reversibility=0.85, urgency=0.38,
                        gap_ids=frozenset()),
                    ttl_seconds=600.0,
                ))
            sm_state = sm.assess()
            print()
            label("self_model",     sm_state["assessment"])
            label("delegation_rate",f"{sm_state['delegation_rate']:.0%}  "
                                    f"(recent window: 80%)")
            label("cached mandates","3  (monitor/stabilize/adjust — all HIGH)")

        pause("\n  Press Enter to run governance evaluation...")

        section("RUNNING MEMI")
        result = session.run(case["goal"], case["world"])
        render_result(result, case["goal"], case["world"])

        print()
        box(f"{case['guarantee']}  —  {case['principle']}")

        if i < len(cases) - 1:
            pause(f"\n  Press Enter for {cases[i+1]['number']}...")
        else:
            pause("\n  Press Enter to return to menu...")


# ─────────────────────────────────────────────
# PART 2 — Interactive mode
# ─────────────────────────────────────────────

INTERACTIVE_HELP = """
  Define a scenario using these parameters (all optional):

  model_valid      = true / false          (default: true)
  model_drift      = 0.0 – 1.0            (default: 0.05)
  critical_drift   = true / false          (default: false)
  sensor_dropout   = true / false          (default: false)
  sensor_coverage  = 0.0 – 1.0            (default: 1.0)
  urgency          = 0.0 – 1.0            (default: 0.3)
  gaps             = N  (number of critical gaps, 0–5, default: 0)

  Examples:
    model_drift=0.9, critical_drift=true
    sensor_dropout=true, urgency=0.9
    gaps=3, sensor_coverage=0.6
    model_valid=false, critical_drift=true, model_drift=0.8
"""

def parse_interactive(line: str) -> dict:
    """Parse 'key=value, key=value' into a dict. Returns {} on failure."""
    result = {}
    for part in line.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        k = k.strip().lower()
        v = v.strip().lower()
        if k in ("model_valid", "critical_drift", "sensor_dropout"):
            result[k] = v in ("true", "yes", "1")
        elif k in ("model_drift", "sensor_coverage", "urgency"):
            try:
                result[k] = float(v)
            except ValueError:
                pass
        elif k == "gaps":
            try:
                result[k] = max(0, min(5, int(v)))
            except ValueError:
                pass
    return result

def build_world(params: dict) -> WorldState:
    gaps = []
    n_gaps = params.get("gaps", 0)
    gap_types = [
        ("sensor",   "coverage_loss"),
        ("model",    "model_uncertainty"),
        ("operator", "missing_approval"),
        ("sensor",   "sensor_dropout_partial"),
        ("model",    "calibration_expired"),
    ]
    for j in range(n_gaps):
        src, typ = gap_types[j % len(gap_types)]
        gaps.append(Gap(
            id=f"interactive_gap_{j}",
            source=src, type=typ,
            would_change_decision=True,
            decision_critical=True,
        ))
    return WorldState(
        model_status=ModelStatus(
            valid=params.get("model_valid", True),
            drift=params.get("model_drift", 0.05),
            critical_drift=params.get("critical_drift", False),
        ),
        sensor_status=SensorStatus(
            dropout=params.get("sensor_dropout", False),
            coverage=params.get("sensor_coverage", 1.0),
        ),
        gaps=gaps,
        urgency=params.get("urgency", 0.3),
    )

def run_interactive(session: GovernedLLMSession) -> None:
    cls()
    title("INTERACTIVE MODE")
    print()
    prose("Define your own scenario. M.E.M.I. will evaluate it.")
    print(INTERACTIVE_HELP)

    while True:
        try:
            raw = input("  scenario> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if raw.lower() in ("q", "quit", "exit", "back", ""):
            break

        params = parse_interactive(raw)
        if not params:
            print("  Could not parse input. Try: model_drift=0.8, urgency=0.7")
            continue

        print()
        section("PARSED PARAMETERS")
        for k, v in sorted(params.items()):
            label(k, str(v))

        world = build_world(params)

        # Show sanitized state — what the LLM sees
        sanitized = WorldStateSanitizer.sanitize(
            world, "Respond to current process conditions"
        )
        section("SANITIZED STATE  (what LLM sees)")
        label("model_confidence", sanitized.model_confidence)
        label("sensor_coverage",  sanitized.sensor_coverage)
        label("sensor_dropout",   str(sanitized.sensor_dropout))
        label("urgency",          sanitized.urgency)
        label("gaps (visible)",   str(len(sanitized.gap_descriptions)))
        print("  [LLM never sees authority, veto, cache, or self-model internals]")

        pause("\n  Press Enter to run governance evaluation...")

        section("RUNNING MEMI")
        goal = "Respond to operator-defined process scenario"
        result = session.run(goal, world)
        render_result(result, goal, world)

        print()
        print("  Try another scenario, or press Enter / type 'back' to return.")
        print()


# ─────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────

def main():
    session, llm_mode = build_session()

    while True:
        cls()
        title("M.E.M.I.  —  Live Governance Demo")
        print()
        print(f"  LLM mode: {llm_mode}")
        print()
        prose(
            "The model suggests. "
            "The system decides whether it is allowed to act."
        )
        print()
        sep("-")
        print()
        print("  [1]  Canonical cases  —  VETO / Boundary Drift / Persuasion Trap")
        print("  [2]  Interactive       —  define your own scenario")
        print("  [3]  Exit")
        print()
        sep("-")

        try:
            choice = input("\n  Choice > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            # Fresh session per canonical run to avoid cross-contamination
            fresh_session, _ = build_session()
            run_canonical(fresh_session)

        elif choice == "2":
            run_interactive(session)

        elif choice in ("3", "q", "exit"):
            break

        else:
            print("  Please enter 1, 2, or 3.")
            time.sleep(0.8)

    cls()
    print()
    box(
        "Autonomy without authority trace is not agency. "
        "It is uncontrolled execution."
    )
    print()


if __name__ == "__main__":
    main()
