"""
M.E.M.I. v9.0 — Governed Multi-Step Planning
=============================================

Builds on v8.5 (Authority Cache / Fast-Lane Gate).

New in v9.0
-----------
- Plan: a sequence of 3–5 proposed actions with intent and preconditions
- Planner: proposes plans — never executes
- PlanExecutor: executes one step at a time through MEMI gate
- PlanResult: full audit trail — per-step authority, pauses, handoffs
- GovernedPlanRun: the governed execution loop

Architectural rules (never violated)
--------------------------------------
  Planning may never own execution.
  Memory may never own authority.
  Self-model may never own veto.
  World-model may never declare itself valid.

Bærende sætning
---------------
  AGI is not reached when the system can plan.
  It begins when planning itself is governed.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Callable

# Re-use all v8.5 primitives
from memi_v85 import (
    MEMI, AuthorityLevel, AuthorityCache, ActionType, VetoType,
    ModelStatus, SensorStatus, Gap, TensionSignature,
    CachedMandate, CacheVerdict, StepResult,
    SelfAssessment,
)


# ─────────────────────────────────────────────
# Plan primitives
# ─────────────────────────────────────────────

class PlanStatus(str, Enum):
    PENDING    = "pending"     # not yet started
    RUNNING    = "running"     # currently executing
    COMPLETED  = "completed"   # all steps executed with authority
    PAUSED     = "paused"      # authority insufficient — waiting
    HANDED_OFF = "handed_off"  # transferred to operator
    ABORTED    = "aborted"     # plan cannot continue (veto / irrecoverable)


class StepStatus(str, Enum):
    PENDING    = "pending"
    EXECUTED   = "executed"
    HELD       = "held"        # LOW authority — holding
    DEFERRED   = "deferred"    # NONE authority — handed off
    SKIPPED    = "skipped"     # plan aborted before this step reached


@dataclass
class PlannedStep:
    """
    A single proposed action within a plan.
    The Planner creates these. MEMI decides whether they may execute.
    """
    step_index:      int
    action:          ActionType
    intent:          str                  # why this step exists
    preconditions:   list[str] = field(default_factory=list)
    reversible:      bool = True
    whitelisted:     bool = False
    expected_gaps:   list[str] = field(default_factory=list)  # gap types anticipated


@dataclass
class Plan:
    """
    An ordered sequence of PlannedSteps with a shared goal.
    Plans are immutable once created — they may only be truncated by governance.
    """
    plan_id:    str
    goal:       str
    steps:      list[PlannedStep]
    created_at: float = field(default_factory=time.time)
    max_steps:  int = 5

    def __post_init__(self):
        if len(self.steps) > self.max_steps:
            raise ValueError(
                f"Plan exceeds max_steps ({self.max_steps}). "
                "Decompose into shorter plans."
            )
        if len(self.steps) < 1:
            raise ValueError("Plan must have at least one step.")

    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class StepExecutionRecord:
    """Full record of one step's governance evaluation and outcome."""
    step_index:     int
    action:         ActionType
    intent:         str
    status:         StepStatus
    authority:      AuthorityLevel
    veto_type:      VetoType
    fast_lane:      bool
    cache_verdict:  CacheVerdict
    mandate_id:     Optional[str]
    gaps:           list[dict]
    handoff:        Optional[dict]
    self_model:     Optional[dict]
    executed_at:    float = field(default_factory=time.time)
    note:           str = ""


@dataclass
class PlanResult:
    """
    Complete audit trail of a governed plan execution.

    Every step is recorded regardless of outcome.
    The plan_status reflects the terminal state of the run.
    """
    plan_id:        str
    goal:           str
    plan_status:    PlanStatus
    steps_total:    int
    steps_executed: int
    steps_held:     int
    steps_deferred: int
    steps_skipped:  int
    records:        list[StepExecutionRecord]
    pause_reason:   Optional[str] = None
    handoff_at:     Optional[int] = None   # step_index where handoff occurred
    started_at:     float = field(default_factory=time.time)
    completed_at:   Optional[float] = None

    def summary(self) -> dict:
        return {
            "plan_id":        self.plan_id,
            "goal":           self.goal,
            "plan_status":    self.plan_status.value,
            "steps_total":    self.steps_total,
            "steps_executed": self.steps_executed,
            "steps_held":     self.steps_held,
            "steps_deferred": self.steps_deferred,
            "steps_skipped":  self.steps_skipped,
            "pause_reason":   self.pause_reason,
            "handoff_at":     self.handoff_at,
        }


# ─────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────

class Planner:
    """
    Proposes plans. Never executes.

    The Planner's only job is to construct a Plan object.
    It has no access to MEMI, no authority model, no execution path.

    In v9.0: deterministic (explicit step definitions).
    In v9.1+: LLM-generated plans may be injected here,
              but must conform to the Plan/PlannedStep interface.
    """

    def create(
        self,
        goal: str,
        steps: list[dict],
        plan_id: Optional[str] = None,
    ) -> Plan:
        """
        Build a Plan from a list of step dicts.

        Each step dict:
          action       : ActionType
          intent       : str
          preconditions: list[str]  (optional)
          reversible   : bool       (optional, default True)
          whitelisted  : bool       (optional, default False)
        """
        planned = []
        for i, s in enumerate(steps):
            planned.append(PlannedStep(
                step_index=i,
                action=ActionType(s["action"]),
                intent=s["intent"],
                preconditions=s.get("preconditions", []),
                reversible=s.get("reversible", True),
                whitelisted=s.get("whitelisted", False),
                expected_gaps=s.get("expected_gaps", []),
            ))

        return Plan(
            plan_id=plan_id or str(uuid.uuid4())[:8],
            goal=goal,
            steps=planned,
        )


# ─────────────────────────────────────────────
# World State (runtime context per step)
# ─────────────────────────────────────────────

@dataclass
class WorldState:
    """
    The current observable state of the world at the time of step execution.
    Provided by the caller — never generated by the Planner or Executor.

    World-model may never declare itself valid.
    → Validity is always evaluated by AuthorityModel, never by WorldState itself.
    """
    model_status:  ModelStatus
    sensor_status: SensorStatus
    gaps:          list[Gap]
    urgency:       float
    memory_boost:  float = 0.0

    def with_urgency(self, urgency: float) -> "WorldState":
        """Return a copy with updated urgency."""
        return WorldState(
            model_status=self.model_status,
            sensor_status=self.sensor_status,
            gaps=self.gaps,
            urgency=urgency,
            memory_boost=self.memory_boost,
        )


# ─────────────────────────────────────────────
# World State Provider
# ─────────────────────────────────────────────

class WorldStateProvider:
    """
    Supplies WorldState for each step of a plan.

    In v9.0: static (same state for all steps) or step-indexed.
    In v9.x: sensor integration, external APIs, satellite data —
             but always through SAT-filter + authority.
    """

    def __init__(self, states: dict[int, WorldState] | WorldState):
        """
        states: either a single WorldState (used for all steps)
                or a dict mapping step_index → WorldState.
        """
        if isinstance(states, WorldState):
            self._static = states
            self._indexed = {}
        else:
            self._static = None
            self._indexed = states

    def get(self, step_index: int) -> WorldState:
        if step_index in self._indexed:
            return self._indexed[step_index]
        if self._static:
            return self._static
        raise KeyError(f"No WorldState defined for step_index={step_index}")


# ─────────────────────────────────────────────
# Governed Plan Executor
# ─────────────────────────────────────────────

class PlanExecutor:
    """
    Executes a Plan one step at a time through the MEMI gate.

    Execution rules
    ---------------
    - Each step is submitted to MEMI.step() independently.
    - If authority >= MEDIUM: step executes, continue to next.
    - If authority == LOW: step is held. Plan is PAUSED.
      (Held ≠ failed. Operator may resume with new WorldState.)
    - If authority == NONE: step is deferred. Plan is HANDED_OFF.
    - If VETO_2: plan is ABORTED. Remaining steps are SKIPPED.
    - If VETO_1: step is held. Plan is PAUSED.

    Planning may never own execution.
    → PlanExecutor has no authority model. It only reads StepResult.
    """

    def __init__(self, memi: MEMI):
        self._memi = memi

    def run(
        self,
        plan: Plan,
        state_provider: WorldStateProvider,
        on_step: Optional[Callable[[StepExecutionRecord], None]] = None,
    ) -> PlanResult:
        """
        Execute all steps in plan order under MEMI governance.

        on_step: optional callback called after each step is evaluated.
                 Useful for real-time monitoring / operator dashboards.
        """
        records: list[StepExecutionRecord] = []
        plan_status   = PlanStatus.RUNNING
        pause_reason: Optional[str] = None
        handoff_at:   Optional[int] = None

        executed = held = deferred = skipped = 0

        for step in plan.steps:

            # If plan already stopped — skip remaining steps
            if plan_status in (PlanStatus.PAUSED, PlanStatus.HANDED_OFF, PlanStatus.ABORTED):
                rec = StepExecutionRecord(
                    step_index=step.step_index,
                    action=step.action,
                    intent=step.intent,
                    status=StepStatus.SKIPPED,
                    authority=AuthorityLevel.NONE,
                    veto_type=VetoType.NONE,
                    fast_lane=False,
                    cache_verdict=CacheVerdict.MISS,
                    mandate_id=None,
                    gaps=[],
                    handoff=None,
                    self_model=None,
                    note="Skipped — plan halted before this step.",
                )
                records.append(rec)
                skipped += 1
                if on_step:
                    on_step(rec)
                continue

            # Get world state for this step
            world = state_provider.get(step.step_index)

            # ── MEMI gate ───────────────────────
            result: StepResult = self._memi.step(
                proposed_action=step.action,
                model_status=world.model_status,
                sensor_status=world.sensor_status,
                gaps=world.gaps,
                urgency=world.urgency,
                reversible=step.reversible,
                memory_boost=world.memory_boost,
                whitelisted=step.whitelisted,
            )

            # ── Interpret result ─────────────────
            if result.veto_type == VetoType.VETO_2:
                # Hard stop — abort entire plan
                status     = StepStatus.DEFERRED
                plan_status = PlanStatus.ABORTED
                pause_reason = f"VETO_2 at step {step.step_index}: {step.intent}"
                handoff_at = step.step_index
                deferred += 1
                note = "VETO_2 — plan aborted. Remaining steps skipped."

            elif result.veto_type == VetoType.VETO_1:
                status     = StepStatus.HELD
                plan_status = PlanStatus.PAUSED
                pause_reason = f"VETO_1 at step {step.step_index}: sensor dropout + high urgency"
                held += 1
                note = "VETO_1 — step held. Plan paused."

            elif result.action == ActionType.DEFER:
                status     = StepStatus.DEFERRED
                plan_status = PlanStatus.HANDED_OFF
                pause_reason = f"Authority NONE at step {step.step_index}: {step.intent}"
                handoff_at = step.step_index
                deferred += 1
                note = "Authority NONE — deferred to operator. Plan handed off."

            elif result.action == ActionType.HOLD:
                status     = StepStatus.HELD
                plan_status = PlanStatus.PAUSED
                pause_reason = f"Authority LOW at step {step.step_index}: {step.intent}"
                held += 1
                note = "Authority LOW — step held. Plan paused."

            else:
                # MEDIUM or HIGH — step executes
                status = StepStatus.EXECUTED
                executed += 1
                note = (
                    f"Fast-lane (mandate {result.mandate_id})"
                    if result.fast_lane
                    else f"Full evaluation (mandate {result.mandate_id})"
                )

            rec = StepExecutionRecord(
                step_index=step.step_index,
                action=step.action,
                intent=step.intent,
                status=status,
                authority=result.authority,
                veto_type=result.veto_type,
                fast_lane=result.fast_lane,
                cache_verdict=result.cache_verdict,
                mandate_id=result.mandate_id,
                gaps=result.gaps,
                handoff=result.handoff,
                self_model=result.self_model,
                note=note,
            )
            records.append(rec)
            if on_step:
                on_step(rec)

        # Final plan status
        if plan_status == PlanStatus.RUNNING:
            plan_status = PlanStatus.COMPLETED

        return PlanResult(
            plan_id=plan.plan_id,
            goal=plan.goal,
            plan_status=plan_status,
            steps_total=plan.step_count(),
            steps_executed=executed,
            steps_held=held,
            steps_deferred=deferred,
            steps_skipped=skipped,
            records=records,
            pause_reason=pause_reason,
            handoff_at=handoff_at,
            completed_at=time.time(),
        )


# ─────────────────────────────────────────────
# Convenience façade
# ─────────────────────────────────────────────

class MEMIv90:
    """
    Top-level façade for v9.0.

    Usage:
        system = MEMIv90()
        plan   = system.planner.create(goal="...", steps=[...])
        states = WorldStateProvider(WorldState(...))
        result = system.execute(plan, states)
    """

    def __init__(
        self,
        operator_reversibility: float = 0.8,
        cache_ttl: float = 300.0,
        handoff_file: Optional[str] = None,
        handoff_callback: Optional[Callable] = None,
    ):
        self.memi = MEMI(
            operator_reversibility=operator_reversibility,
            cache_ttl=cache_ttl,
            handoff_file=handoff_file,
            handoff_callback=handoff_callback,
        )
        self.planner  = Planner()
        self.executor = PlanExecutor(self.memi)

    def execute(
        self,
        plan: Plan,
        state_provider: WorldStateProvider,
        on_step: Optional[Callable[[StepExecutionRecord], None]] = None,
    ) -> PlanResult:
        return self.executor.run(plan, state_provider, on_step=on_step)

    def apply_feedback(self, feedback: dict) -> None:
        self.memi.apply_handoff_feedback(feedback)

    def cache_summary(self) -> dict:
        return self.memi.cache_summary()

    def self_model(self) -> dict:
        return self.memi.self_model.assess()


# ─────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────

def print_result(result: PlanResult) -> None:
    print(f"\n  Plan: '{result.goal}'")
    print(f"  Status: {result.plan_status.value}  "
          f"({result.steps_executed} executed, "
          f"{result.steps_held} held, "
          f"{result.steps_deferred} deferred, "
          f"{result.steps_skipped} skipped)")
    if result.pause_reason:
        print(f"  Pause: {result.pause_reason}")
    for r in result.records:
        fl = " [FAST-LANE]" if r.fast_lane else ""
        print(f"    [{r.step_index}] {r.action.value:20s} "
              f"auth={r.authority.value:6s} "
              f"veto={r.veto_type.value:6s} "
              f"status={r.status.value:10s}{fl}")
        print(f"         intent: {r.intent}")
        print(f"         note:   {r.note}")


def demo():
    print("=" * 65)
    print("M.E.M.I. v9.0 — Governed Multi-Step Planning")
    print("=" * 65)

    system = MEMIv90(operator_reversibility=0.85, cache_ttl=600.0)

    healthy = WorldState(
        model_status=ModelStatus(valid=True, drift=0.05),
        sensor_status=SensorStatus(dropout=False, coverage=1.0),
        gaps=[],
        urgency=0.35,
    )

    # ── Case 1: Full plan completes ─────────────────────────────
    print("\n\n── Case 1: All conditions healthy — plan should complete ──")
    plan1 = system.planner.create(
        goal="Stabilise and monitor process after anomaly detection",
        steps=[
            {"action": "monitor",    "intent": "Establish baseline before acting",        "reversible": True},
            {"action": "stabilize",  "intent": "Bring system to safe operating state",    "reversible": True},
            {"action": "monitor",    "intent": "Confirm stabilisation",                   "reversible": True},
            {"action": "adjust",     "intent": "Fine-tune parameters within safe range",  "reversible": True},
            {"action": "monitor",    "intent": "Verify adjustment outcome",               "reversible": True},
        ]
    )
    result1 = system.execute(plan1, WorldStateProvider(healthy))
    print_result(result1)

    # ── Case 2: Fast-lane activates on repeated plan ─────────────
    print("\n\n── Case 2: Repeated plan — fast-lane should activate ──")
    plan2 = system.planner.create(
        goal="Routine monitoring cycle (known-good conditions)",
        steps=[
            {"action": "monitor",   "intent": "Scheduled check",            "reversible": True},
            {"action": "stabilize", "intent": "Maintain current set-point", "reversible": True},
            {"action": "monitor",   "intent": "Confirm stability",          "reversible": True},
        ]
    )
    result2 = system.execute(plan2, WorldStateProvider(healthy))
    print_result(result2)

    # ── Case 3: Multiple critical gaps mid-plan → plan pauses ────
    print("\n\n── Case 3: Multiple critical gaps at step 1 — plan should pause ──")
    gaps_severe = [
        Gap(id="g1", source="sensor",   type="coverage_loss",    would_change_decision=True, decision_critical=True),
        Gap(id="g2", source="model",    type="model_uncertainty", would_change_decision=True, decision_critical=True),
        Gap(id="g3", source="operator", type="missing_approval",  would_change_decision=True, decision_critical=True),
        Gap(id="g4", source="sensor",   type="dropout_partial",   would_change_decision=True, decision_critical=True),
    ]
    severely_degraded = WorldState(
        model_status=ModelStatus(valid=True, drift=0.15),
        sensor_status=SensorStatus(dropout=False, coverage=0.5),
        gaps=gaps_severe,
        urgency=0.5,
    )
    plan3 = system.planner.create(
        goal="Adjust pressure after severe sensor degradation",
        steps=[
            {"action": "monitor",   "intent": "Assess current state",       "reversible": True},
            {"action": "adjust",    "intent": "Reduce pressure by 10%",     "reversible": True},
            {"action": "stabilize", "intent": "Hold new pressure target",   "reversible": True},
        ]
    )
    # Step 0 healthy, steps 1-2 severely degraded
    result3 = system.execute(
        plan3,
        WorldStateProvider({0: healthy, 1: severely_degraded, 2: severely_degraded})
    )
    print_result(result3)

    # ── Case 4: VETO_2 — aborts entire plan ─────────────────────
    print("\n\n── Case 4: VETO_2 at step 1 — plan should abort ──")
    veto_state = WorldState(
        model_status=ModelStatus(valid=False, drift=0.95, critical_drift=True),
        sensor_status=SensorStatus(dropout=False, coverage=1.0),
        gaps=[],
        urgency=0.5,
    )
    plan4 = system.planner.create(
        goal="Emergency reconfiguration (irreversible)",
        steps=[
            {"action": "monitor",    "intent": "Pre-check",                    "reversible": True},
            {"action": "adjust",     "intent": "Irreversible config change",   "reversible": False},
            {"action": "stabilize",  "intent": "Lock in new config",           "reversible": False},
        ]
    )
    result4 = system.execute(
        plan4,
        WorldStateProvider({0: healthy, 1: veto_state, 2: veto_state})
    )
    print_result(result4)

    # ── Case 5: VETO_1 — sensor dropout + high urgency ───────────
    print("\n\n── Case 5: VETO_1 — sensor dropout + high urgency ──")
    veto1_state = WorldState(
        model_status=ModelStatus(valid=True, drift=0.05),
        sensor_status=SensorStatus(dropout=True, coverage=0.2),
        gaps=[],
        urgency=0.9,
    )
    plan5 = system.planner.create(
        goal="Stabilise under sensor dropout",
        steps=[
            {"action": "monitor",   "intent": "Read available sensors",        "reversible": True},
            {"action": "stabilize", "intent": "Stabilise despite blind spots", "reversible": True},
            {"action": "monitor",   "intent": "Re-check",                      "reversible": True},
        ]
    )
    result5 = system.execute(
        plan5,
        WorldStateProvider({0: healthy, 1: veto1_state, 2: veto1_state})
    )
    print_result(result5)

    # ── Final diagnostics ────────────────────────────────────────
    print("\n\n── Cache summary ──")
    import pprint
    pprint.pprint(system.cache_summary())

    print("\n── Self-model ──")
    pprint.pprint(system.self_model())

    print("\n✓ v9.0 smoke-test complete.\n")


if __name__ == "__main__":
    demo()
