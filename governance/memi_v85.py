"""
M.E.M.I. v8.5 — Governed Epistemic Decision Architecture
Authority Cache / Fast-Lane Gate

Builds on v8.4 (self-triggered pause, boundary_drift → gap injection).
New in v8.5: Earned authority can be cached and reused — but never assumed.

Hard guarantee:
  Cache-hit can shorten the process.
  Cache-hit can NEVER overrule veto, gaps, or self-model-triggered review.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Callable, Any


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class AuthorityLevel(str, Enum):
    NONE   = "NONE"
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class ActionType(str, Enum):
    STABILIZE   = "stabilize"
    HOLD        = "hold"
    MONITOR     = "monitor"
    ADJUST      = "adjust"
    DEFER       = "defer_to_operator"


class VetoType(str, Enum):
    NONE   = "none"
    VETO_1 = "VETO_1"   # sensor_dropout + urgency >= 0.75 + stabilize → LOW + hold
    VETO_2 = "VETO_2"   # model invalid + critical_drift + irreversible → NONE + defer


class SelfAssessment(str, Enum):
    STABLE_BOUNDARY    = "stable_boundary"
    BOUNDARY_DRIFT     = "boundary_drift"
    OVER_DELEGATING    = "over_delegating"
    UNDER_DELEGATING   = "under_delegating"


class CacheVerdict(str, Enum):
    FAST_LANE   = "fast_lane"     # All checks passed → skip full evaluation
    MISS        = "miss"          # No matching cache entry
    INVALIDATED = "invalidated"   # Entry found but one or more checks failed


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class ModelStatus:
    valid: bool = True
    drift: float = 0.0          # 0.0–1.0
    critical_drift: bool = False
    rupture: bool = False


@dataclass
class SensorStatus:
    dropout: bool = False
    coverage: float = 1.0       # fraction of expected sensors reporting


@dataclass
class Gap:
    id: str
    source: str                 # "sensor" | "model" | "operator" | "self_model"
    type: str
    would_change_decision: bool = True
    suggested_query: dict = field(default_factory=dict)
    decision_critical: bool = True


@dataclass
class TensionSignature:
    """
    Compact fingerprint of the epistemic state at decision time.
    Used to verify that a cached mandate still matches the present.
    """
    coverage: float
    validity: float
    reversibility: float
    urgency: float
    gap_ids: frozenset = field(default_factory=frozenset)

    def distance(self, other: "TensionSignature") -> float:
        """Euclidean distance in the 4-dimensional authority space."""
        return math.sqrt(
            (self.coverage   - other.coverage)   ** 2 +
            (self.validity   - other.validity)   ** 2 +
            (self.reversibility - other.reversibility) ** 2 +
            (self.urgency    - other.urgency)    ** 2
        )

    def gap_delta(self, other: "TensionSignature") -> set:
        """Gaps present in `other` but not in `self` (new gaps since caching)."""
        return set(other.gap_ids) - set(self.gap_ids)


@dataclass
class CachedMandate:
    """
    A stored authority grant that may enable fast-lane execution.

    Fields
    ------
    mandate_id      : unique identifier
    action_type     : the action that earned this authority
    authority_level : must be >= MEDIUM to be cacheable
    tension_sig     : epistemic fingerprint at time of earning
    whitelisted     : True if action is unconditionally reversible/safe
    earned_at       : timestamp (epoch seconds)
    ttl_seconds     : time-to-live; None = no expiry
    use_count       : how many times this mandate was used (audit)
    """
    mandate_id:      str
    action_type:     ActionType
    authority_level: AuthorityLevel
    tension_sig:     TensionSignature
    whitelisted:     bool = False
    earned_at:       float = field(default_factory=time.time)
    ttl_seconds:     Optional[float] = 300.0
    use_count:       int = 0

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.earned_at) > self.ttl_seconds

    # Authority level ordering
    _LEVEL_ORDER = {
        AuthorityLevel.NONE:   0,
        AuthorityLevel.LOW:    1,
        AuthorityLevel.MEDIUM: 2,
        AuthorityLevel.HIGH:   3,
    }

    def level_sufficient(self) -> bool:
        return self._LEVEL_ORDER[self.authority_level] >= self._LEVEL_ORDER[AuthorityLevel.MEDIUM]


@dataclass
class CacheCheckResult:
    verdict:          CacheVerdict
    mandate_id:       Optional[str]      = None
    failed_checks:    list[str]          = field(default_factory=list)
    authority_level:  Optional[AuthorityLevel] = None
    detail:           str                = ""


@dataclass
class AuthorityResult:
    level:      AuthorityLevel
    score:      float
    veto_type:  VetoType
    scores:     dict = field(default_factory=dict)
    gaps:       list[Gap] = field(default_factory=list)
    tension_sig: Optional[TensionSignature] = None


@dataclass
class StepResult:
    step_id:        str
    action:         ActionType
    authority:      AuthorityLevel
    veto_type:      VetoType
    fast_lane:      bool            # True = cache was used
    cache_verdict:  CacheVerdict
    handoff:        Optional[dict]
    self_model:     Optional[dict]
    gaps:           list[dict]
    tension_sig:    Optional[dict]
    mandate_id:     Optional[str]   # ID of mandate used or earned


# ─────────────────────────────────────────────
# Authority Cache
# ─────────────────────────────────────────────

class AuthorityCache:
    """
    Stores earned mandates keyed by action_type.
    One mandate per action type (last earned wins).

    Lookup protocol (all five checks must pass):
      1. authority_level >= MEDIUM
      2. tension signature within threshold
      3. no new decision-critical gaps since earning
      4. no boundary_drift flag
      5. action is reversible or whitelisted

    Hard guarantee: Cache.hit() never bypasses veto evaluation.
    Veto evaluation is always the caller's responsibility.
    """

    SIGNATURE_THRESHOLD = 0.15   # max Euclidean distance allowed

    def __init__(self):
        self._store: dict[ActionType, CachedMandate] = {}
        self._eviction_log: list[dict] = []

    # ── Write ──────────────────────────────────

    def store(self, mandate: CachedMandate) -> None:
        """Store a new mandate. Only MEDIUM+ authority is accepted."""
        if not mandate.level_sufficient():
            return   # silently reject — low authority is never cached
        self._store[mandate.action_type] = mandate

    def invalidate(self, action_type: ActionType, reason: str) -> None:
        """Explicitly evict a cached mandate."""
        if action_type in self._store:
            m = self._store.pop(action_type)
            self._eviction_log.append({
                "mandate_id": m.mandate_id,
                "action_type": action_type,
                "reason": reason,
                "evicted_at": time.time(),
            })

    def invalidate_all(self, reason: str) -> None:
        for at in list(self._store.keys()):
            self.invalidate(at, reason)

    # ── Read ───────────────────────────────────

    def check(
        self,
        action_type: ActionType,
        current_sig: TensionSignature,
        boundary_drift: bool,
        reversible: bool,
    ) -> CacheCheckResult:
        """
        Run all five fast-lane gate checks.
        Returns CacheCheckResult with verdict FAST_LANE | MISS | INVALIDATED.

        Does NOT evaluate veto — that is always the caller's responsibility.
        """
        mandate = self._store.get(action_type)

        if mandate is None:
            return CacheCheckResult(
                verdict=CacheVerdict.MISS,
                detail="No cached mandate for this action type."
            )

        failed: list[str] = []

        # Check 1 — authority level
        if not mandate.level_sufficient():
            failed.append("authority_level < MEDIUM")

        # Check 2 — tension signature distance
        dist = mandate.tension_sig.distance(current_sig)
        if dist > self.SIGNATURE_THRESHOLD:
            failed.append(
                f"tension_signature_drift ({dist:.3f} > {self.SIGNATURE_THRESHOLD})"
            )

        # Check 3 — new decision-critical gaps
        new_gaps = mandate.tension_sig.gap_delta(current_sig)
        if new_gaps:
            failed.append(f"new_decision_critical_gaps: {new_gaps}")

        # Check 4 — boundary drift
        if boundary_drift:
            failed.append("boundary_drift_active")

        # Check 5 — reversibility or whitelist
        if not reversible and not mandate.whitelisted:
            failed.append("action_not_reversible_or_whitelisted")

        # Expiry (not a formal check, but evicts the entry)
        if mandate.is_expired():
            self.invalidate(action_type, "ttl_expired")
            failed.append("mandate_expired")

        if failed:
            return CacheCheckResult(
                verdict=CacheVerdict.INVALIDATED,
                mandate_id=mandate.mandate_id,
                failed_checks=failed,
                authority_level=mandate.authority_level,
                detail=f"Cache invalidated: {'; '.join(failed)}",
            )

        # All checks passed
        mandate.use_count += 1
        return CacheCheckResult(
            verdict=CacheVerdict.FAST_LANE,
            mandate_id=mandate.mandate_id,
            authority_level=mandate.authority_level,
            detail="Fast-lane authorised. All five checks passed.",
        )

    # ── Introspection ──────────────────────────

    def summary(self) -> dict:
        return {
            "cached_actions": [at.value for at in self._store],
            "entries": [
                {
                    "action_type": m.action_type.value,
                    "authority_level": m.authority_level.value,
                    "earned_at": m.earned_at,
                    "use_count": m.use_count,
                    "whitelisted": m.whitelisted,
                    "expired": m.is_expired(),
                }
                for m in self._store.values()
            ],
            "eviction_log": self._eviction_log,
        }


# ─────────────────────────────────────────────
# Authority Model  (unchanged from v8.4)
# ─────────────────────────────────────────────

class AuthorityModel:

    VETO_SCORE_MAP = {
        AuthorityLevel.NONE:   0.0,
        AuthorityLevel.LOW:    0.3,
        AuthorityLevel.MEDIUM: 0.6,
        AuthorityLevel.HIGH:   1.0,
    }

    def __init__(
        self,
        operator_reversibility: float = 0.8,
        calibration_bias: float = 0.0,
    ):
        self.operator_reversibility = operator_reversibility
        self.calibration_bias = calibration_bias
        self.feedback_adjustment: float = 0.0

    def _coverage(self, gaps: list[Gap]) -> float:
        if not gaps:
            return 1.0
        decision_critical = sum(1 for g in gaps if g.decision_critical)
        return max(0.0, 1.0 - 0.25 * decision_critical)

    def _validity(self, model_status: ModelStatus) -> float:
        v = 1.0
        if not model_status.valid:
            v -= 0.5
        if model_status.critical_drift:
            v -= 0.3
        if model_status.rupture:
            v -= 0.2
        v -= model_status.drift * 0.2
        return max(0.0, v)

    def _reversibility(self) -> float:
        return self.operator_reversibility

    def evaluate(
        self,
        model_status:  ModelStatus,
        sensor_status: SensorStatus,
        gaps:          list[Gap],
        urgency:       float,
        proposed_action: ActionType,
        memory_boost:  float = 0.0,
        drift_penalty: float = 0.0,
    ) -> AuthorityResult:

        coverage     = self._coverage(gaps)
        validity     = self._validity(model_status)
        reversibility = self._reversibility()

        raw_score = (
            min(coverage, validity, reversibility)
            + memory_boost
            + self.feedback_adjustment
            + self.calibration_bias
            - drift_penalty
        )
        score = max(0.0, min(1.0, raw_score))

        # ── VETO 2 ──────────────────────────────
        if (
            not model_status.valid
            and model_status.critical_drift
            and reversibility < 0.3
        ):
            return AuthorityResult(
                level=AuthorityLevel.NONE,
                score=0.0,
                veto_type=VetoType.VETO_2,
                scores={"coverage": coverage, "validity": validity,
                        "reversibility": reversibility, "final": 0.0},
                gaps=gaps,
                tension_sig=TensionSignature(
                    coverage=coverage, validity=validity,
                    reversibility=reversibility, urgency=urgency,
                    gap_ids=frozenset(g.id for g in gaps),
                ),
            )

        # ── VETO 1 ──────────────────────────────
        if (
            sensor_status.dropout
            and urgency >= 0.75
            and proposed_action == ActionType.STABILIZE
        ):
            return AuthorityResult(
                level=AuthorityLevel.LOW,
                score=score,
                veto_type=VetoType.VETO_1,
                scores={"coverage": coverage, "validity": validity,
                        "reversibility": reversibility, "final": score},
                gaps=gaps,
                tension_sig=TensionSignature(
                    coverage=coverage, validity=validity,
                    reversibility=reversibility, urgency=urgency,
                    gap_ids=frozenset(g.id for g in gaps),
                ),
            )

        # ── Score → level ────────────────────────
        if score >= 0.75:
            level = AuthorityLevel.HIGH
        elif score >= 0.5:
            level = AuthorityLevel.MEDIUM
        elif score >= 0.25:
            level = AuthorityLevel.LOW
        else:
            level = AuthorityLevel.NONE

        return AuthorityResult(
            level=level,
            score=score,
            veto_type=VetoType.NONE,
            scores={"coverage": coverage, "validity": validity,
                    "reversibility": reversibility, "final": score},
            gaps=gaps,
            tension_sig=TensionSignature(
                coverage=coverage, validity=validity,
                reversibility=reversibility, urgency=urgency,
                gap_ids=frozenset(g.id for g in gaps),
            ),
        )


# ─────────────────────────────────────────────
# Self-Model  (unchanged from v8.4)
# ─────────────────────────────────────────────

@dataclass
class SelfModelEntry:
    step_id:     str
    action:      ActionType
    authority:   AuthorityLevel
    deferred:    bool
    veto_type:   VetoType


class SelfModelLog:
    DRIFT_WINDOW = 5
    DRIFT_THRESHOLD = 0.6

    def __init__(self):
        self._history: list[SelfModelEntry] = []

    def record(self, entry: SelfModelEntry) -> None:
        self._history.append(entry)

    def assess(self) -> dict:
        if not self._history:
            return {"assessment": SelfAssessment.STABLE_BOUNDARY.value,
                    "boundary_stability": 1.0, "delegation_rate": 0.0,
                    "dominant_veto": "none", "autonomy_boundary": "no history"}

        total      = len(self._history)
        deferred   = sum(1 for e in self._history if e.deferred)
        del_rate   = deferred / total

        recent = self._history[-self.DRIFT_WINDOW:]
        recent_def = sum(1 for e in recent if e.deferred)
        recent_rate = recent_def / len(recent) if recent else 0.0

        stability = 1.0 - abs(recent_rate - del_rate)

        veto_counts = {VetoType.VETO_1: 0, VetoType.VETO_2: 0, VetoType.NONE: 0}
        for e in self._history:
            veto_counts[e.veto_type] += 1
        dominant = max(veto_counts, key=veto_counts.get)
        dom_str = ("none" if dominant == VetoType.NONE
                   else "mixed" if veto_counts[VetoType.VETO_1] == veto_counts[VetoType.VETO_2]
                   else dominant.value)

        if recent_rate > del_rate + self.DRIFT_THRESHOLD:
            assessment = SelfAssessment.BOUNDARY_DRIFT
        elif del_rate > 0.8:
            assessment = SelfAssessment.OVER_DELEGATING
        elif del_rate < 0.1 and total >= 5:
            assessment = SelfAssessment.UNDER_DELEGATING
        else:
            assessment = SelfAssessment.STABLE_BOUNDARY

        return {
            "assessment":        assessment.value,
            "autonomy_boundary": f"Delegates ~{del_rate*100:.0f}% of decisions",
            "boundary_stability": round(stability, 2),
            "delegation_rate":   round(del_rate, 2),
            "dominant_veto":     dom_str,
        }

    def boundary_drift_active(self) -> bool:
        return self.assess()["assessment"] == SelfAssessment.BOUNDARY_DRIFT.value


# ─────────────────────────────────────────────
# Handoff Pipeline  (unchanged from v8.4)
# ─────────────────────────────────────────────

class HandoffPipeline:
    def __init__(self):
        self._log: list[dict] = []
        self._callback: Optional[Callable] = None
        self._file: Optional[str] = None

    def set_callback(self, fn: Callable) -> None:
        self._callback = fn

    def set_file(self, path: str) -> None:
        self._file = path

    def emit(
        self,
        step_id: str,
        intended_operator: ActionType,
        authority_level: AuthorityLevel,
        veto_type: VetoType,
        required_action: str,
        decision_question: str,
        gaps: list[Gap],
        resume_condition: str,
    ) -> dict:
        pkg = {
            "step_id":           step_id,
            "timestamp":         time.time(),
            "intended_operator": intended_operator.value,
            "authority_level":   authority_level.value,
            "veto_type":         veto_type.value,
            "required_action":   required_action,
            "decision_question": decision_question,
            "gaps_to_resolve":   [asdict(g) for g in gaps],
            "resume_condition":  resume_condition,
        }
        self._log.append(pkg)
        if self._callback:
            self._callback(pkg)
        if self._file:
            with open(self._file, "a") as f:
                f.write(json.dumps(pkg) + "\n")
        print(f"\n[HANDOFF → operator] step={step_id} veto={veto_type.value} "
              f"authority={authority_level.value}")
        return pkg


# ─────────────────────────────────────────────
# MEMI v8.5
# ─────────────────────────────────────────────

class MEMI:
    """
    M.E.M.I. v8.5 — Governed Epistemic Decision Architecture
    with Authority Cache / Fast-Lane Gate.

    New in v8.5
    -----------
    - AuthorityCache stores earned mandates
    - step() runs cache check BEFORE full authority evaluation
    - Fast-lane skips evaluation IFF all five checks pass AND veto is clear
    - Mandate is stored after every MEDIUM+ authority evaluation
    - boundary_drift and veto always re-evaluated, never cached away
    """

    def __init__(
        self,
        operator_reversibility: float = 0.8,
        handoff_file: Optional[str] = None,
        handoff_callback: Optional[Callable] = None,
        cache_ttl: float = 300.0,
        signature_threshold: float = 0.15,
    ):
        self.authority_model = AuthorityModel(
            operator_reversibility=operator_reversibility
        )
        self.self_model   = SelfModelLog()
        self.handoff      = HandoffPipeline()
        self.cache        = AuthorityCache()
        self.cache.SIGNATURE_THRESHOLD = signature_threshold

        if handoff_file:
            self.handoff.set_file(handoff_file)
        if handoff_callback:
            self.handoff.set_callback(handoff_callback)

        self._default_cache_ttl = cache_ttl
        self._step_log: list[StepResult] = []

    # ── Public API ────────────────────────────

    def step(
        self,
        proposed_action: ActionType,
        model_status:    ModelStatus,
        sensor_status:   SensorStatus,
        gaps:            list[Gap],
        urgency:         float,
        reversible:      bool = True,
        memory_boost:    float = 0.0,
        whitelisted:     bool = False,
    ) -> StepResult:
        """
        Execute one decision step.

        Fast-lane path (v8.5):
            1. Check for boundary_drift (self-model)
            2. Pre-evaluate veto conditions (constitutional — never skipped)
            3. If no boundary_drift AND no veto → run cache check
            4. Cache HIT → use cached authority, skip full evaluation
            5. Cache MISS/INVALIDATED → full authority evaluation
            6. Store mandate if authority >= MEDIUM

        Returns StepResult with fast_lane=True/False.
        """

        step_id = str(uuid.uuid4())[:8]

        # ── Step A: self-model assessment ────────
        self_model_data = self.self_model.assess()
        boundary_drift  = self.self_model.boundary_drift_active()

        # Inject self-model gap if boundary_drift (v8.4 behaviour preserved)
        if boundary_drift:
            sm_gap = Gap(
                id=f"sm_drift_{step_id}",
                source="self_model",
                type="boundary_drift",
                would_change_decision=True,
                suggested_query={
                    "targets": ["autonomy_boundary_review"],
                    "urgency": 0.85,
                },
                decision_critical=True,
            )
            gaps = gaps + [sm_gap]

        # ── Step B: Veto pre-check (always runs) ──
        # We compute the provisional authority to catch veto conditions
        # before consulting the cache. This is the hard guarantee.
        provisional = self.authority_model.evaluate(
            model_status=model_status,
            sensor_status=sensor_status,
            gaps=gaps,
            urgency=urgency,
            proposed_action=proposed_action,
            memory_boost=memory_boost,
        )

        veto_active = provisional.veto_type != VetoType.NONE

        # ── Step C: Cache check ──────────────────
        # Only attempted if: no boundary_drift AND no veto
        cache_result = CacheCheckResult(verdict=CacheVerdict.MISS)
        fast_lane    = False
        mandate_id:  Optional[str] = None

        if not boundary_drift and not veto_active:
            cache_result = self.cache.check(
                action_type=proposed_action,
                current_sig=provisional.tension_sig,
                boundary_drift=boundary_drift,
                reversible=reversible,
            )

            if cache_result.verdict == CacheVerdict.FAST_LANE:
                fast_lane   = True
                mandate_id  = cache_result.mandate_id
                # Use cached authority level (already verified by cache checks)
                authority_result = AuthorityResult(
                    level=cache_result.authority_level,
                    score=provisional.score,
                    veto_type=VetoType.NONE,
                    scores=provisional.scores,
                    gaps=gaps,
                    tension_sig=provisional.tension_sig,
                )

        # ── Step D: Full evaluation (if no fast-lane) ─
        if not fast_lane:
            authority_result = provisional  # already computed above

            # Store mandate if authority earned is MEDIUM+
            if authority_result.veto_type == VetoType.NONE:
                level_order = {
                    AuthorityLevel.NONE: 0, AuthorityLevel.LOW: 1,
                    AuthorityLevel.MEDIUM: 2, AuthorityLevel.HIGH: 3,
                }
                if level_order[authority_result.level] >= level_order[AuthorityLevel.MEDIUM]:
                    mandate = CachedMandate(
                        mandate_id=str(uuid.uuid4())[:8],
                        action_type=proposed_action,
                        authority_level=authority_result.level,
                        tension_sig=authority_result.tension_sig,
                        whitelisted=whitelisted,
                        ttl_seconds=self._default_cache_ttl,
                    )
                    self.cache.store(mandate)
                    mandate_id = mandate.mandate_id

        # ── Step E: Decide action + handoff ─────
        action, handoff_pkg = self._decide(
            step_id=step_id,
            proposed=proposed_action,
            authority=authority_result,
            gaps=gaps,
        )

        # ── Step F: Record in self-model ─────────
        self.self_model.record(SelfModelEntry(
            step_id=step_id,
            action=action,
            authority=authority_result.level,
            deferred=(action == ActionType.DEFER),
            veto_type=authority_result.veto_type,
        ))

        result = StepResult(
            step_id=step_id,
            action=action,
            authority=authority_result.level,
            veto_type=authority_result.veto_type,
            fast_lane=fast_lane,
            cache_verdict=cache_result.verdict,
            handoff=handoff_pkg,
            self_model=self_model_data,
            gaps=[asdict(g) for g in gaps],
            tension_sig=asdict(authority_result.tension_sig) if authority_result.tension_sig else None,
            mandate_id=mandate_id,
        )
        self._step_log.append(result)
        return result

    # ── Internal ──────────────────────────────

    def _decide(
        self,
        step_id: str,
        proposed: ActionType,
        authority: AuthorityResult,
        gaps: list[Gap],
    ) -> tuple[ActionType, Optional[dict]]:

        level = authority.level
        veto  = authority.veto_type

        # VETO 2 → defer
        if veto == VetoType.VETO_2:
            pkg = self.handoff.emit(
                step_id=step_id,
                intended_operator=proposed,
                authority_level=level,
                veto_type=veto,
                required_action="Operator must validate model and confirm irreversible action.",
                decision_question="Is the model state reliable enough to proceed?",
                gaps=gaps,
                resume_condition="model_status.valid=True and no critical_drift",
            )
            return ActionType.DEFER, pkg

        # VETO 1 → hold
        if veto == VetoType.VETO_1:
            return ActionType.HOLD, None

        # Score-based
        if level == AuthorityLevel.NONE:
            pkg = self.handoff.emit(
                step_id=step_id,
                intended_operator=proposed,
                authority_level=level,
                veto_type=veto,
                required_action="Operator must resolve gaps and re-authorise.",
                decision_question="Should the system proceed with the proposed action?",
                gaps=gaps,
                resume_condition="authority_score >= 0.25 with gaps resolved",
            )
            return ActionType.DEFER, pkg

        if level == AuthorityLevel.LOW:
            return ActionType.HOLD, None

        if level in (AuthorityLevel.MEDIUM, AuthorityLevel.HIGH):
            return proposed, None

        return ActionType.HOLD, None

    # ── Feedback API  (v7.6–v8.0 preserved) ──

    def apply_handoff_feedback(self, feedback: dict) -> None:
        outcome = feedback.get("outcome", "neutral")
        adjustment = {"positive": 0.10, "negative": -0.10}.get(outcome, 0.0)
        self.authority_model.feedback_adjustment = max(
            -0.10,
            min(0.10, self.authority_model.feedback_adjustment + adjustment)
        )

    # ── Diagnostics ───────────────────────────

    def cache_summary(self) -> dict:
        return self.cache.summary()

    def step_log(self) -> list[StepResult]:
        return self._step_log


# ─────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("M.E.M.I. v8.5 — Authority Cache / Fast-Lane Gate")
    print("=" * 60)

    memi = MEMI(operator_reversibility=0.85, cache_ttl=600.0)

    healthy_model  = ModelStatus(valid=True, drift=0.05)
    healthy_sensor = SensorStatus(dropout=False, coverage=1.0)

    print("\n── Step 1: First execution (cold cache) ──")
    r1 = memi.step(
        proposed_action=ActionType.STABILIZE,
        model_status=healthy_model,
        sensor_status=healthy_sensor,
        gaps=[],
        urgency=0.4,
        reversible=True,
    )
    print(f"  action={r1.action.value}  authority={r1.authority.value}  "
          f"fast_lane={r1.fast_lane}  cache={r1.cache_verdict.value}  "
          f"mandate={r1.mandate_id}")

    print("\n── Step 2: Same conditions → fast-lane ──")
    r2 = memi.step(
        proposed_action=ActionType.STABILIZE,
        model_status=healthy_model,
        sensor_status=healthy_sensor,
        gaps=[],
        urgency=0.42,   # small drift, within threshold
        reversible=True,
    )
    print(f"  action={r2.action.value}  authority={r2.authority.value}  "
          f"fast_lane={r2.fast_lane}  cache={r2.cache_verdict.value}  "
          f"mandate={r2.mandate_id}")

    print("\n── Step 3: New decision-critical gap → cache invalidated ──")
    new_gap = Gap(id="g1", source="sensor", type="coverage_loss",
                  would_change_decision=True, decision_critical=True)
    r3 = memi.step(
        proposed_action=ActionType.STABILIZE,
        model_status=healthy_model,
        sensor_status=healthy_sensor,
        gaps=[new_gap],
        urgency=0.4,
        reversible=True,
    )
    print(f"  action={r3.action.value}  authority={r3.authority.value}  "
          f"fast_lane={r3.fast_lane}  cache={r3.cache_verdict.value}")

    print("\n── Step 4: VETO_1 — sensor dropout + high urgency ──")
    r4 = memi.step(
        proposed_action=ActionType.STABILIZE,
        model_status=healthy_model,
        sensor_status=SensorStatus(dropout=True, coverage=0.3),
        gaps=[],
        urgency=0.9,
        reversible=True,
    )
    print(f"  action={r4.action.value}  authority={r4.authority.value}  "
          f"veto={r4.veto_type.value}  fast_lane={r4.fast_lane}")

    print("\n── Step 5: VETO_2 — model invalid + critical drift + irreversible ──")
    r5 = memi.step(
        proposed_action=ActionType.ADJUST,
        model_status=ModelStatus(valid=False, drift=0.9, critical_drift=True),
        sensor_status=healthy_sensor,
        gaps=[],
        urgency=0.5,
        reversible=False,
    )
    print(f"  action={r5.action.value}  authority={r5.authority.value}  "
          f"veto={r5.veto_type.value}  fast_lane={r5.fast_lane}")

    print("\n── Cache summary ──")
    import pprint
    pprint.pprint(memi.cache_summary())

    print("\n── Self-model ──")
    pprint.pprint(memi.self_model.assess())

    print("\n✓ Smoke-test complete.\n")


if __name__ == "__main__":
    demo()
