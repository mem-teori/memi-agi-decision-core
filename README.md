# M.E.M.I. — Governed Intelligence Stack

> *A system is not safe because it decides well.  
> It is safe because it can refuse to decide.*

M.E.M.I. (Maria Edlenborg Mortensen Intelligence) is a governance-native AI decision architecture designed for high-risk environments.

It separates two things most AI systems conflate:

```
Risk    = what could go wrong
Authority = whether the system is permitted to act on it
```

**The system does not act because it can. It acts when it understands what it is doing and has been permitted to do it.**

---

## Repository structure

```
docs/               Architecture papers and overviews
governance/         Core governance engine (v8.5 → v9.2)
agi-core/           AGI decision core (v10.5 → v11.0)
deployment/         ETOS-AD bridge + live dashboard
```

---

## Governance stack (v8.5 → v9.2)

| File | Description |
|---|---|
| `governance/memi_v85.py` | Authority Cache / Fast-Lane Gate |
| `governance/memi_v90.py` | Governed Multi-Step Planning |
| `governance/memi_v91_FROZEN.py` | LLM Plan Proposal Interface |
| `governance/memi_v92_FROZEN.py` | ClaudeLLMAdapter (live LLM behind fixed interface) |
| `governance/memi_demo.py` | Three canonical cases: VETO / Boundary Drift / Persuasion Trap |
| `governance/memi_live_demo_FROZEN.py` | Interactive terminal demo |

Run the live demo:
```bash
python governance/memi_live_demo_FROZEN.py
# With live Claude:
ANTHROPIC_API_KEY=<key> python governance/memi_live_demo_FROZEN.py
```

---

## AGI decision core (v10.5 → v11.0)

| File | Description |
|---|---|
| `agi-core/memi_v107_FROZEN.py` | 2-Step Foresight |
| `agi-core/memi_v108_FROZEN.py` | Action Effect Model / Symbolic Grounding |
| `agi-core/memi_v110_FROZEN.py` | Epistemic State / Meta-Cognition |

Each version builds on the previous. Run any directly:
```bash
python agi-core/memi_v110_FROZEN.py
```

**The four verified properties:**
1. Simulate futures before acting (v10.7)
2. Understand what actions do (v10.8)
3. Know when it does not know enough to act (v11.0)
4. All under governance — planning never owns execution

---

## Deployment: ETOS-AD bridge

| File | Description |
|---|---|
| `deployment/etos_memi_bridge.py` | Flask API — governance gate for security events |
| `deployment/memi_governance_dashboard.html` | Live browser dashboard — open directly |

```bash
pip install flask flask-cors
python deployment/etos_memi_bridge.py
# Open deployment/memi_governance_dashboard.html in browser
```

Test:
```bash
curl -X POST http://127.0.0.1:8000/api/decision \
  -H "Content-Type: application/json" \
  -d '{"event":{"process_name":"powershell.exe","anomaly_score":0.88,"live_c2":true,"confidence":0.55,"impact_category":"high","hour_of_day":3,"whitelist_match":false,"command_line":"-enc test"}}'
```

---

## Core documents

| File | Description |
|---|---|
| `docs/memi_architecture.md` | Full architecture paper (EN + DA) |
| `docs/memi_agi_core.md` | AGI decision core — proven components |
| `docs/etos_memi_overview.md` | One-page stack overview (EN + DA) |

---

## Architectural invariant

```
Foresight selects the proposal.
Effect model explains why.
Epistemic state qualifies confidence.
Governance decides whether it is permitted.
```

No component can override M.E.M.I. authority evaluation.  
Planning does not own execution.  
Learning does not own authority.

---

## Contact

Maria Edlenborg Mortensen  
mem-teori.github.io/etos-site  
April 2026
