## Start here

If you are new to this project:

1. Read: `docs/memi_agi_core.md`  
2. Run: `python agi-core/memi_v115_FROZEN.py`  
3. Explore: `python governance/memi_live_demo_FROZEN.py`  

This repository documents a **verified decision architecture**, not a general AI system.

---

# M.E.M.I. — Governed Intelligence Stack

> A system is not safe because it decides well.  
> It is safe because it can refuse to decide.

Most AI systems act under uncertainty.  
This one evaluates whether it knows enough to act at all.

M.E.M.I. is not classical machine learning.  
It is an **adaptive decision architecture under epistemic governance**.

---

## What this is

M.E.M.I. (Maria Edlenborg Mortensen Intelligence) is a governance-native decision architecture designed for high-risk environments.

It separates two things most AI systems conflate:

```text
Risk       = what could go wrong
Authority  = whether the system is permitted to act on it

The system does not act because it can.
It acts when it understands what it is doing and has been permitted to do it.

Repository structure
docs/         Architecture papers and overviews
governance/   Core governance engine (v8.5 → v9.2)
agi-core/     AGI decision core (v10.5 → v11.5)
deployment/   ETOS-AD bridge + live dashboard
Governance stack (v8.5 → v9.2)
File	Description
governance/memi_v85.py	Authority Cache / Fast-Lane Gate
governance/memi_v90.py	Governed Multi-Step Planning
governance/memi_v91_FROZEN.py	LLM Plan Proposal Interface
governance/memi_v92_FROZEN.py	ClaudeLLMAdapter
governance/memi_demo.py	Canonical cases
governance/memi_live_demo_FROZEN.py	Interactive demo

Run:

python governance/memi_live_demo_FROZEN.py
AGI decision core (v10.5 → v11.5)
File	Description
agi-core/memi_v107_FROZEN.py	2-Step Foresight
agi-core/memi_v108_FROZEN.py	Action Effect Model / Symbolic Grounding
agi-core/memi_v110_FROZEN.py	Epistemic State / Meta-Cognition
agi-core/memi_v111_FROZEN.py	Decision Payload / Uncertainty Propagation
agi-core/memi_v112_FROZEN.py	Observe Outcome + Build Learning Proposal
agi-core/memi_v113_FROZEN.py	Govern Learning
agi-core/memi_v114_FROZEN.py	Apply Accepted Learning
agi-core/memi_v115_FROZEN.py	Multi-Observation Learning

Run:

python agi-core/memi_v115_FROZEN.py
Verified properties
Simulates futures before acting
Understands what actions do
Knows when it does not know enough to act
Carries uncertainty with decisions
Proposes learning without applying it automatically
Governs learning before model update
Applies only accepted learning
Learns only from consistent multi-observation evidence

All under governance — planning never owns execution.

Deployment (ETOS-AD)

Run:

pip install flask flask-cors
python deployment/etos_memi_bridge.py

Open:

deployment/memi_governance_dashboard.html
Core documents
docs/memi_agi_core.md — Proven components
docs/memi_architecture.md — Full architecture
docs/etos_memi_overview.md — Overview
docs/v10.8_action_effect_model.md — Action Effect Model architecture note
docs/memi_v11x_architecture.md — Governed Learning architecture note
Architectural invariant
Foresight selects the proposal.
Effect model explains why.
Epistemic state qualifies confidence.
Governance decides whether it is permitted.
Decision payload carries uncertainty beyond the system.
Learning proposals are governed decisions — not automatic updates.
Only accepted proposals may update the model.
The system does not trust a single experience.
It learns from consistency across experiences.
License

This repository is made publicly available for research, evaluation, and reference purposes only.

Use in production systems, commercial environments, or operational decision-making contexts requires a separate license agreement.

The M.E.M.I. decision architecture and its extensions remain the intellectual property of Maria Edlenborg Mortensen.

For licensing inquiries:
📩 m.e.m.terori@gmail.com
