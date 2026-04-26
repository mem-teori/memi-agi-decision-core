"""
M.E.M.I. x ETOS-AD Bridge
===========================
Drop-in replacement for ETOS-AD backend/app.py

New pipeline:
  security event
    → map_event_to_tensions()      (ETOS — domain knowledge, unchanged)
    → etos_evaluate()              (ETOS — initial mode + recommended_action)
    → memi_world_state()           (bridge — translate to M.E.M.I. WorldState)
    → MEMI.step()                  (M.E.M.I. — authority check + governance)
    → merge_decision()             (bridge — combine ETOS + MEMI into final response)
    → log + return

API response adds:
  memi_authority     HIGH / MEDIUM / LOW / NONE
  memi_veto          VETO_1 / VETO_2 / none
  memi_fast_lane     bool
  memi_action        stabilize / hold / defer_to_operator / monitor
  memi_handoff       dict (if deferred)
  memi_mandate_id    str (if cached)
  final_action       MEMI-governed final action string
  governed           True — flag so frontend knows MEMI is active

Frontend receives same fields as before (mode, score, recommended_action)
plus new memi_* fields. Zero breaking changes to existing frontend.

Governance rules
----------------
  ETOS decides the domain interpretation.
  M.E.M.I. decides whether the system is allowed to act on it.
  ETOS may never override M.E.M.I. veto.
  M.E.M.I. may never override ETOS tension mapping.
"""

from __future__ import annotations

import sys
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── M.E.M.I. imports ────────────────────────────────────────────────
# Add parent directory to path if running from ETOS-AD backend folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memi_v85 import (
    ModelStatus, SensorStatus, Gap,
    AuthorityLevel, VetoType, ActionType,
)
from memi_v90 import MEMIv90, WorldState

# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

DECISIONS_FILE = "decisions_memi.jsonl"
FEEDBACK_FILE  = "feedback_memi.jsonl"

for f in [DECISIONS_FILE, FEEDBACK_FILE]:
    if not os.path.exists(f):
        open(f, "a", encoding="utf-8").close()

# ── Single shared MEMI instance ──────────────
# operator_reversibility: security actions are generally less reversible
# cache_ttl: 5 minutes — security context changes fast
memi = MEMIv90(
    operator_reversibility=0.55,
    cache_ttl=300.0,
)

# ─────────────────────────────────────────────
# ETOS domain layer (unchanged from original)
# ─────────────────────────────────────────────

def map_event_to_tensions(event: dict) -> dict:
    """Original ETOS tension mapping — domain knowledge, not modified."""
    import math
    anomaly    = float(event.get("anomaly_score", 0.0))
    confidence = float(event.get("confidence", 0.8))
    impact     = event.get("impact_category", "medium")
    hour       = int(event.get("hour_of_day", 12))
    whitelist  = bool(event.get("whitelist_match", False))
    live_c2    = bool(event.get("live_c2", False))
    proc       = str(event.get("process_name", "")).lower()
    cmd        = str(event.get("command_line", "")).lower()

    coercion    = min(5, round(anomaly * 6))
    refusal     = 4 if whitelist else 1
    capacity    = round((1 - confidence) * 5)
    risk_map    = {"low": 1, "medium": 3, "high": 4, "critical": 5}
    risk        = risk_map.get(impact, 2)
    time_urgency= 5 if (hour <= 5 or hour >= 22 or live_c2) else 2
    advance     = 5 if whitelist else 0

    if "powershell" in proc and ("-enc" in cmd or "base64" in cmd):
        coercion     = max(coercion, 5)
        risk         = max(risk, 4)
        time_urgency = max(time_urgency, 5)

    if any(x in proc for x in ["certutil", "bitsadmin", "regsvr32", "mshta", "wscript"]):
        coercion = max(coercion, 4)
        risk     = max(risk, 4)

    if "vssadmin" in proc and "delete shadows" in cmd:
        coercion      = 5
        risk          = 5
        time_urgency  = 5

    if proc in {"explorer.exe", "msedge.exe", "notepad.exe", "svchost.exe"} and anomaly < 0.4:
        coercion = min(coercion, 1)
        risk     = 1

    return {
        "coercion": coercion,
        "refusal":  refusal,
        "capacity": capacity,
        "risk":     risk,
        "time":     time_urgency,
        "advance_directive": advance,
    }


def etos_evaluate(tensions: dict, event: dict) -> tuple[str, float, str]:
    """Original ETOS evaluation — determines initial mode and recommended action."""
    total = sum(tensions.values())
    score = min(1.0, total / 30.0)

    if tensions["advance_directive"] >= 4 and tensions["coercion"] >= 4:
        return "Incompatible", round(score, 3), "Block + Escalate (principkonflikt)"

    proc = str(event.get("process_name", "")).lower()
    if proc in {"explorer.exe", "msedge.exe", "svchost.exe"} and \
       float(event.get("anomaly_score", 0)) < 0.35:
        return "Normal", round(min(score, 0.28), 3), "Log & Monitor"

    if score > 0.78:
        return "Emergency",       round(score, 3), "Kill + Isolate"
    if score > 0.48:
        return "Requires Review", round(score, 3), "Analyst Review"
    if score > 0.32:
        return "Incompatible",    round(score, 3), "Block + Escalate"
    return "Normal",              round(score, 3), "Log & Monitor"


# ─────────────────────────────────────────────
# Bridge: ETOS → M.E.M.I. WorldState
# ─────────────────────────────────────────────

# Action type mapping: ETOS recommended_action → M.E.M.I. ActionType
ACTION_MAP = {
    "Log & Monitor":           ActionType.MONITOR,
    "Analyst Review":          ActionType.STABILIZE,   # review = stabilise + observe
    "Block + Escalate":        ActionType.ADJUST,       # block = adjust system posture
    "Block + Escalate (principkonflikt)": ActionType.ADJUST,
    "Kill + Isolate":          ActionType.STABILIZE,   # strongest reversible action
}

# Reversibility: how reversible is each ETOS action?
REVERSIBILITY_MAP = {
    "Log & Monitor":           True,
    "Analyst Review":          True,
    "Block + Escalate":        False,   # blocking may have operational impact
    "Block + Escalate (principkonflikt)": False,
    "Kill + Isolate":          False,   # process kill is not easily undone
}


def event_to_world_state(
    event: dict,
    tensions: dict,
    etos_score: float,
    etos_mode: str,
) -> tuple[WorldState, list[Gap], ActionType, bool]:
    """
    Translate ETOS event + tensions into M.E.M.I. WorldState.

    Returns (world_state, gaps, proposed_action, reversible).

    Translation rules
    -----------------
    coverage  ← 1 - (capacity / 5)  [capacity = 1 - confidence]
    validity  ← based on etos_score bands
    urgency   ← time tension / 5
    gaps      ← critical signals from tensions + event properties
    reversible← from REVERSIBILITY_MAP
    """
    confidence  = float(event.get("confidence", 0.8))
    live_c2     = bool(event.get("live_c2", False))
    whitelist   = bool(event.get("whitelist_match", False))
    proc        = str(event.get("process_name", "")).lower()
    anomaly     = float(event.get("anomaly_score", 0.0))
    hour        = int(event.get("hour_of_day", 12))
    impact      = event.get("impact_category", "medium")

    # ── Model status ────────────────────────
    # High anomaly + low confidence = model is uncertain about this event
    model_drift   = max(0.0, min(1.0, (anomaly * 0.5) + ((1 - confidence) * 0.5)))
    critical_drift= (tensions["coercion"] >= 5 and tensions["risk"] >= 4)
    model_valid   = confidence >= 0.45 and not (anomaly > 0.9 and confidence < 0.5)

    # ── Sensor status ────────────────────────
    # capacity tension = missing/uncertain data = sensor degradation analog
    sensor_coverage = max(0.1, 1.0 - (tensions["capacity"] / 5))
    sensor_dropout  = (tensions["capacity"] >= 4) or live_c2

    # ── Urgency ─────────────────────────────
    urgency = min(1.0, tensions["time"] / 5)
    if live_c2:
        urgency = max(urgency, 0.9)
    if impact == "critical":
        urgency = max(urgency, 0.75)

    # ── Gaps ────────────────────────────────
    gaps: list[Gap] = []
    gap_id = 0

    if tensions["capacity"] >= 3:
        gaps.append(Gap(
            id=f"g_{gap_id}", source="sensor",
            type="low_confidence_detection",
            would_change_decision=True,
            decision_critical=tensions["capacity"] >= 4,
        ))
        gap_id += 1

    if live_c2:
        gaps.append(Gap(
            id=f"g_{gap_id}", source="sensor",
            type="live_c2_channel_active",
            would_change_decision=True,
            decision_critical=True,
        ))
        gap_id += 1

    if tensions["coercion"] >= 4 and not whitelist:
        gaps.append(Gap(
            id=f"g_{gap_id}", source="model",
            type="high_coercion_unverified",
            would_change_decision=True,
            decision_critical=tensions["coercion"] >= 5,
        ))
        gap_id += 1

    if etos_mode == "Emergency" and not whitelist:
        gaps.append(Gap(
            id=f"g_{gap_id}", source="operator",
            type="emergency_mode_no_prior_approval",
            would_change_decision=True,
            decision_critical=True,
        ))
        gap_id += 1

    # ── Proposed action + reversibility ─────
    etos_recommended = etos_evaluate(tensions, event)[2]
    proposed_action  = ACTION_MAP.get(etos_recommended, ActionType.MONITOR)
    reversible       = REVERSIBILITY_MAP.get(etos_recommended, True)

    world = WorldState(
        model_status=ModelStatus(
            valid=model_valid,
            drift=model_drift,
            critical_drift=critical_drift,
        ),
        sensor_status=SensorStatus(
            dropout=sensor_dropout,
            coverage=sensor_coverage,
        ),
        gaps=gaps,
        urgency=urgency,
        memory_boost=0.05 if whitelist else 0.0,
    )

    return world, gaps, proposed_action, reversible


def memi_action_to_final(
    memi_action:      ActionType,
    etos_recommended: str,
    etos_mode:        str,
    authority:        AuthorityLevel,
    veto_type:        VetoType,
) -> str:
    """
    Translate M.E.M.I. decision back into a human-readable final action.
    ETOS recommendation is used as input — MEMI decides whether it's permitted.
    """
    if veto_type == VetoType.VETO_2:
        return "BLOCKED — Operator required (VETO: model invalid + irreversible)"
    if veto_type == VetoType.VETO_1:
        return "HELD — Sensor dropout + high urgency (VETO: await data)"
    if memi_action == ActionType.DEFER:
        return f"DEFERRED — {etos_recommended} pending operator authorisation"
    if memi_action == ActionType.HOLD:
        return f"HELD — {etos_recommended} pending authority review"
    # MEDIUM/HIGH: ETOS recommendation is authorised
    return etos_recommended


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def log_decision(data: dict) -> None:
    entry = {"timestamp": datetime.utcnow().isoformat() + "Z", **data}
    with open(DECISIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_feedback(data: dict) -> None:
    entry = {"timestamp": datetime.utcnow().isoformat() + "Z", **data}
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(
        f"[FEEDBACK] {entry['feedback_type']} | "
        f"{entry.get('process_name', 'ukendt')} | "
        f"mode={entry['mode']} | action={entry['recommended_action']}"
    )


# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    sm = memi.self_model()
    return jsonify({
        "status":       "ok",
        "service":      "etos-ad-memi-bridge",
        "memi_version": "9.2",
        "self_model":   sm,
        "cache":        memi.cache_summary(),
    })


@app.post("/api/decision")
def decision():
    data  = request.get_json(silent=True) or {}
    event = data.get("event", {})

    # ── Step 1: ETOS domain evaluation ──────
    tensions                           = map_event_to_tensions(event)
    etos_mode, etos_score, etos_action = etos_evaluate(tensions, event)

    # ── Step 2: Translate to M.E.M.I. WorldState ──
    world, gaps, proposed_action, reversible = event_to_world_state(
        event, tensions, etos_score, etos_mode
    )

    # ── Step 3: M.E.M.I. authority check ────
    step_result = memi.memi.step(
        proposed_action=proposed_action,
        model_status=world.model_status,
        sensor_status=world.sensor_status,
        gaps=gaps,
        urgency=world.urgency,
        reversible=reversible,
        memory_boost=world.memory_boost,
    )

    # ── Step 4: Compose final decision ──────
    final_action = memi_action_to_final(
        memi_action=step_result.action,
        etos_recommended=etos_action,
        etos_mode=etos_mode,
        authority=step_result.authority,
        veto_type=step_result.veto_type,
    )

    # ── Step 5: Log ──────────────────────────
    log_entry = {
        "event":           event,
        "tensions":        tensions,
        "etos_mode":       etos_mode,
        "etos_score":      etos_score,
        "etos_action":     etos_action,
        "memi_authority":  step_result.authority.value,
        "memi_veto":       step_result.veto_type.value,
        "memi_action":     step_result.action.value,
        "memi_fast_lane":  step_result.fast_lane,
        "memi_mandate_id": step_result.mandate_id,
        "final_action":    final_action,
    }
    log_decision(log_entry)

    print(
        f"[DECISION] {event.get('process_name', '?'):20s} "
        f"etos={etos_mode:16s} "
        f"auth={step_result.authority.value:6s} "
        f"veto={step_result.veto_type.value:6s} "
        f"→ {final_action}"
    )

    # ── Step 6: Return (backwards-compatible + memi_* fields) ──
    response = {
        # Original ETOS fields — unchanged for frontend compatibility
        "mode":               etos_mode,
        "score":              etos_score,
        "tensions":           tensions,
        "recommended_action": etos_action,
        "event":              event,
        # M.E.M.I. governance fields
        "memi_authority":     step_result.authority.value,
        "memi_veto":          step_result.veto_type.value,
        "memi_action":        step_result.action.value,
        "memi_fast_lane":     step_result.fast_lane,
        "memi_mandate_id":    step_result.mandate_id,
        "memi_gaps":          len(gaps),
        "memi_self_model":    step_result.self_model.get("assessment") if step_result.self_model else None,
        "final_action":       final_action,
        "governed":           True,
    }

    if step_result.handoff:
        response["memi_handoff"] = {
            "required_action":  step_result.handoff.get("required_action"),
            "decision_question":step_result.handoff.get("decision_question"),
            "resume_condition": step_result.handoff.get("resume_condition"),
            "veto_type":        step_result.handoff.get("veto_type"),
        }

    return jsonify(response)


@app.post("/api/feedback")
def feedback():
    data          = request.get_json(silent=True) or {}
    feedback_type = data.get("feedback_type")

    entry = {
        "event_id":           data.get("event_id"),
        "process_name":       data.get("process_name"),
        "mode":               data.get("mode"),
        "recommended_action": data.get("recommended_action"),
        "feedback_type":      feedback_type,
        "notes":              data.get("notes", ""),
    }

    if feedback_type == "clear_feedback":
        entry["notes"] = "Feedback ryddet / undo"

    log_feedback(entry)

    # ── Feed back into M.E.M.I. calibration ─
    outcome_map = {
        "confirm_action": "positive",
        "false_positive": "negative",
        "clear_feedback": "neutral",
    }
    memi.apply_feedback({"outcome": outcome_map.get(feedback_type, "neutral")})

    return jsonify({
        "status":  "ok",
        "message": "Feedback modtaget og logget",
        "entry":   entry,
    })


@app.get("/api/memi/status")
def memi_status():
    """New endpoint: expose full M.E.M.I. governance state."""
    return jsonify({
        "self_model":    memi.self_model(),
        "cache":         memi.cache_summary(),
        "memi_version":  "9.2",
    })


if __name__ == "__main__":
    print("[M.E.M.I. x ETOS-AD] Bridge starting on port 8000")
    print("[M.E.M.I. x ETOS-AD] Governance active — authority check on every event")
    app.run(port=8000, debug=True)
