# Revenue Pipeline Forecaster

[![tests](https://github.com/f20220970-debug/revenue-pipeline-forecaster/actions/workflows/tests.yml/badge.svg)](https://github.com/f20220970-debug/revenue-pipeline-forecaster/actions/workflows/tests.yml)

Three questions a founder asks every week about an inbound sales team:

1. Which open deals will close?
2. How much will we bind next month?
3. How many agents do we need on each hour?

All three are answered with methods that are **back-tested against what actually happened**.

> **Simulated data:** 94,240 opportunities over 20 months for a US small-business insurance brokerage growing 6% a month. Each deal has a true close probability and close date, so every method can be checked. No company data is used.

Run it: `pip install -r requirements.txt && python forecaster.py && pytest -q` (about 15 seconds).

## 1. Lead scoring: ranking and calibration

Models are trained on months 0–11 and tested on months 12–15, so they never see the future.

| Model | AUC | Brier score | Share of wins in the top 20% of scores | Worst calibration gap |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.729 | 0.184 | 36.5% | 2.6 pts |
| **Gradient boosting** | **0.747** | **0.179** | **38.2%** | 3.6 pts |
| Oracle (true probabilities) | 0.750 | | | |

- **Gradient boosting is close to the ceiling.** It sits 0.003 AUC below a model that knows the true probabilities, so the remaining error is genuine uncertainty, not a modelling gap.
- **Calibration matters as much as ranking.** The forecast in section 2 sums probabilities, so a 3-point gap feeds straight into revenue.

## 2. Next-month bound commission from the open pipeline

The pipeline is forecast at 7 month-ends, each using only data known on that day.

| Method | Mean error | Bias | Months the actual fell inside the P10–P90 band |
| --- | --- | --- | --- |
| Stage-weighted (CRM default: 10% / 30% / 55% by stage) | **82.8%** | +82.8% (always over) | n/a |
| Monte Carlo, raw (deal probability × time-to-close, 5,000 draws) | 7.8% | | 14.3% (bands far too narrow) |
| **Monte Carlo, self-calibrated** | **3.8%** | | **85.7%** (target 80%) |

**What went wrong with each method, and the fix:**
- **Stage weights count a deal's whole probability as if it lands next month.** Many deals close later or quietly die. The Monte Carlo instead conditions on **how long the deal has already been open**: an old open deal is less likely to win and less likely to close soon.
- **Raw simulation bands only reflect deal-by-deal luck, not model error.** The calibrated version learns the forecast's own error from earlier month-ends, then shifts and widens the band. Its P10–P90 band now contains the actual about as often as it should.

## 3. Capacity: agents per hour (US Eastern) for a 5-minute answer SLA

An Erlang C queue model, at 6,000 leads a month and 12 minutes of handling per lead, sizes the team so that 80% of leads are answered within 5 minutes:

- **Peak:** 5 agents at 9–11 AM ET.
- **Overnight:** 1–2 agents.
- **Shaped staffing needs 78 agent-hours a day, against 120 for flat staffing at the peak level (−35%).**

See `capacity_by_hour.csv`. This is also where an AI first-response agent pays off: it covers the overnight hours that are expensive to staff.

## Files

```
forecaster.py         simulator + scoring + back-tested forecast + Erlang C capacity
results.json          headline numbers
backtest.csv          each month-end: actual vs stage-weighted vs Monte Carlo bands
capacity_by_hour.csv  agents and SLA attainment per hour
tests/                no future leakage; model near oracle; MC beats stage weights; Erlang C sanity
```

Companion repos: [soma-brokerage-ops-lab](https://github.com/f20220970-debug/soma-brokerage-ops-lab) and [arr-growth-model](https://github.com/f20220970-debug/arr-growth-model).
