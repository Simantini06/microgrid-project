"""Energy Management System (EMS) core - shared by every approach.

This package holds the microgrid *physics/accounting* environment, the common
controller contract, the metrics, and the rule-based baseline controller. The
Generative-AI (Stage 4) and Agentic-AI (Stage 5) approaches plug into the SAME
environment and are scored with the SAME metrics, so the comparison is fair.

Design contract (the same for all three approaches):
* A controller *decides* using **forecasts** (what it would know in advance).
* The environment then *settles* the hour on **actuals** (real generation and
  demand), updates the battery, and books cost / CO2 / renewable use.
"""
