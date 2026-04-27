# M.E.M.I. — Governed Intelligence Stack

> *A system is not safe because it decides well.  
> It is safe because it can refuse to decide.*

M.E.M.I. (Maria Edlenborg Mortensen Intelligence) is a governance-native AI decision architecture designed for high-risk environments.

It separates two things most AI systems conflate:


Risk = what could go wrong
Authority = whether the system is permitted to act on it


The system does not act because it can.  
It acts when it understands what it is doing and has been permitted to do it.

**M.E.M.I. is not classical machine learning.**  
It is an **adaptive decision architecture under epistemic governance**.

---

## Start here

If you are new to this project:

1. Read: `docs/memi_agi_core.md`  
2. Run: `python agi-core/memi_v123_FROZEN.py`  
3. Explore: `python governance/memi_live_demo_FROZEN.py`  

This repository documents a **verified decision architecture**, not a general AI system.

---

## Repository structure


docs/ Architecture papers and overviews
governance/ Core governance engine (v8.5 → v9.2)
agi-core/ AGI decision core (v10.7 → v12.3)
deployment/ ETOS-AD bridge + live dashboard


---

## Governance stack (v8.5 → v9.2)

| File | Description |
|------|------------|
| `governance/memi_v85.py` | Authority Cache / Fast-Lane Gate |
| `governance/memi_v90.py` | Governed Multi-Step Planning |
| `governance/memi_v91_FROZEN.py` | LLM Plan Proposal Interface |
| `governance/memi_v92_FROZEN.py` | ClaudeLLMAdapter |
| `governance/memi_demo.py` | Canonical cases |
| `governance/memi_live_demo_FROZEN.py` | Interactive demo |

Run:

```bash
python governance/memi_live_demo_FROZEN.py
AGI decision core (v10.7 → v12.3)
Reading order

The AGI core is presented as a sequence of frozen versions.
Each version proves a specific property.

Run in order:

memi_v107 → memi_v123
Planning & grounding
File	Description
memi_v107_FROZEN.py	2-Step Foresight
memi_v108_FROZEN.py	Action Effect Model
memi_v110_FROZEN.py	Epistemic State
Governed learning
File	Description
memi_v111_FROZEN.py	Uncertainty Propagation
memi_v112_FROZEN.py	Observe + Propose
memi_v113_FROZEN.py	Govern Learning
memi_v114_FROZEN.py	Apply Accepted Learning
memi_v115_FROZEN.py	Multi-Observation Learning
memi_v116_FROZEN.py	Admissibility Layer
Contextual intelligence
File	Description
memi_v120_FROZEN.py	Contextual Admissibility
memi_v121_FROZEN.py	Context-Scoped Learning
memi_v122_FROZEN.py	Context Switching
memi_v123_FROZEN.py	Governed Cross-Context Transfer

Run latest:

python agi-core/memi_v123_FROZEN.py
Verified properties
Simulates futures before acting
Understands what actions do
Knows when it does not know enough to act
Carries uncertainty with decisions
Proposes learning without applying it automatically
Governs learning before model update
Applies only accepted learning
Learns only from consistent experience
Evaluates only admissible futures
Admissibility is context-dependent
Learning is stored per domain
Context is a governed decision
Cross-domain transfer is governed — never automatic
All under governance — planning never owns execution
Deployment (ETOS-AD)
pip install flask flask-cors
python deployment/etos_memi_bridge.py

Open:

deployment/memi_governance_dashboard.html
Core documents
docs/memi_agi_core.md
docs/memi_architecture.md
docs/etos_memi_overview.md
docs/memi_v11x_architecture.md
docs/memi_v12x_architecture.md
Architectural invariant
Foresight selects the proposal.
Foresight only evaluates admissible proposals.
Effect model explains why.
Epistemic state qualifies confidence.
Governance decides whether it is permitted.
Decision payload carries uncertainty beyond the system.
Learning proposals are governed decisions — not automatic updates.

The system does not need knowledge to act.
It needs knowledge to act irreversibly.

License

This repository is made publicly available for research, evaluation, and reference purposes only.

Use in production systems, commercial environments, or operational decision-making contexts requires a separate license agreement.

The M.E.M.I. decision architecture, including all components in this repository, remains the intellectual property of Maria Edlenborg Mortensen.

Commercial licensing inquiries:

📩 m.e.m.terori@gmail.com