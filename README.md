# Credit Stress-Testing Engine

**A credit stress-testing project that links loan-level default risk to macroeconomic stress through a PD → satellite → scenario pipeline.**

This project builds an end-to-end stress-testing engine for unsecured consumer credit using LendingClub loan data. It estimates fixed-horizon loan-level probability of default, calibrates out-of-time predictions to observed portfolio default levels, links realized-vs-expected default experience to macroeconomic conditions, and transmits historical recession paths through a scenario engine to estimate stressed default and expected loss.

**The project is model‑risk‑aware**: the focus extends beyond predictive modeling and the final loss estimate to include leakage control, out‑of‑time validation, calibration, stability testing, and the diagnostic process that corrected an earlier flawed lifetime‑PD design to arrive at the final fixed‑horizon framework.


---

## Project Highlights

- Built a leakage-safe **24-month probability-of-default model** using origination-time borrower, loan, and credit-bureau attributes.
- Validated the PD model on true out-of-time vintages, with **OOT AUC = 0.707** and stable validation/OOT performance.
- Recalibrated raw PDs with an intercept-only adjustment, correcting default-level underprediction while preserving rank ordering.
- Built a macro satellite model that estimates an actual-to-expected default multiplier as a function of macro conditions.
- Found a robust unemployment transmission channel that is stable across specifications: a **+1 percentage point larger rise in unemployment increases the A/E default multiplier by about 22.6%**.
- Transmitted calm, adverse, and severely adverse unemployment paths through the satellite model.
- Estimated that a GFC-paced unemployment shock raises annual expected loss by about **36%**, or roughly **+$21.4 million per $1 billion of exposure**, through the unemployment channel alone.
- Documented a legacy model-risk failure mode: an earlier terminal-status lifetime-PD design generated a spurious late-window loss escalation, which was traced to survivorship/truncation bias and corrected by rebuilding the outcome definition around a fixed 24-month horizon.

---


## Technology Stack

**Modeling & analysis:** Python · pandas · NumPy · scikit-learn · statsmodels · matplotlib  
**Cloud & deployment:** AWS EC2 · Amazon S3 · IAM · boto3 · Git/GitHub

---

## AWS Deployment

Component 3 was refactored from the original notebook workflow into a reusable Python scenario engine and deployed on AWS. Prepared model inputs are stored in Amazon S3, the stress-testing computation runs on an Amazon EC2 Linux instance, and the resulting three CSV tables and four PNG figures are written back to S3.

The EC2 instance accesses S3 through an IAM instance role rather than embedded credentials, with `boto3` handling S3 access from Python. The same scenario engine can also run locally. Components 1 and 2 remain the upstream modeling workflow that produces the scored loan-level PDs and macro-satellite coefficients consumed by Component 3.

---

## Repository layout

```text
├── README.md                         this file
├── WRITEUP.md                        full technical report
├── feature_selection.md              details of feature selection process
├── requirements.txt                  Python dependencies
├── main.py                           entry point for the refactored scenario engine
├── src/
│   └── scenario_engine.py            reusable Component 3 Python module with S3 integration
├── 01_pd_model.ipynb                 Component 1 — PD model
├── 02_macro_satellite.ipynb          Component 2 — macro satellite
├── 03_scenario_engine.ipynb          Component 3 — original notebook workflow
├── artifacts_component1/             PD tables & figures
├── artifacts_component2/             satellite tables & figures
├── artifacts_component3/             scenario tables & figures
└── legacy/                           legacy version and diagnostics, including artifacts
    └── legacy_PD_satellite.ipynb
```

Each notebook writes its tables and figures to its own `artifacts_component*/` folder.

The layer hand-off files passed between notebooks are generated within notebooks:
- scored loan-level PD: `scored_loan_level_pd.parquet`, `scored_loan_level_pd.csv`
- satellite coefficients: `satellite_coefficients_baseline.csv`

For the AWS deployment, `scored_loan_level_pd.parquet` and `satellite_coefficients_baseline.csv` are stored in the S3 `inputs/` prefix and consumed by the refactored Component 3 Python module.


---

## Component 1 — Probability of Default

The first layer estimates each loan's probability of default within a fixed 24-month horizon from origination-time information only. The target is `default_24m`: default or charge-off within 24 months of origination, evaluated only on loans observed for a full 24 months.

Key design choices:

- Fixed-horizon outcome rather than terminal lifetime status.
- Eligibility waterfall to remove loans without sufficient follow-up, ambiguous status, or unresolved timing.
- Leakage audit excluding post-origination variables.
- Logistic regression scorecard using loan terms, borrower financials, and backward-looking credit-bureau attributes.
- Time-based train/validation/OOT split by vintage rather than random sampling.
- Intercept-only calibration to reconcile predicted and observed default levels.

Primary validation results:

| Metric | Train | Validation | OOT |
|---|---:|---:|---:|
| AUC | 0.680 | 0.709 | 0.707 |
| Gini | 0.360 | 0.418 | 0.414 |
| KS | 0.261 | 0.307 | 0.304 |
| Brier | 0.096 | 0.110 | 0.111 |

The calibrated PD is used as the primary reporting basis in downstream satellite and scenario layers.

---

## Component 2 — Macro Satellite

The second layer links portfolio default experience to macroeconomic conditions. For each calendar quarter, the model constructs an actual-to-expected default multiplier:

- Expected defaults are created by distributing each loan's 24-month PD across calendar quarters using an empirical default-age curve.
- Actual defaults are observed realized defaults in each quarter.
- The multiplier is actual defaults divided by expected defaults.

The satellite model regresses the log A/E multiplier on lagged macro variables using Newey-West/HAC standard errors.

Headline result:

> A +1 percentage point larger rise in unemployment increases the A/E default multiplier by about 22.6%.

The coefficient remains correctly signed and significant across specifications that add credit spreads, GDP growth, and a time trend.

![Figure 2.2 — Unemployment coefficient stability across specifications](artifacts_component2/figure2.2_coefficient_stability.png)
*Figure 2.2 — The unemployment coefficient (Δunemployment, lag 1) across the five specifications, calibrated (primary) vs. raw (sensitivity), with HAC 95% confidence intervals.*

---

## Component 3 — Scenario Engine

The third layer transmits historical macro stress paths through the unemployment channel to estimate stressed portfolio default and expected loss.

Scenarios:

| Scenario | Description |
|---|---|
| Baseline | Flat unemployment at 4.0% |
| Adverse | 2001 recession unemployment profile |
| Severely adverse | 2008–09 financial-crisis unemployment profile |

Scenario results:

| Scenario | Annual default | 24-month default | Annual EL | EL per $1B exposure |
|---|---:|---:|---:|---:|
| Baseline | 7.26% | 13.76% | 5.95% | $59.5M |
| Adverse | 8.45% | 15.86% | 6.93% | $69.3M |
| Severe | 9.86% | 18.29% | 8.10% | $81.0M |

The severe scenario increases annual expected loss by about **36%**, or about **+$21.4M per $1B of exposure**, relative to baseline. Because the current satellite transmits only the unemployment channel, this severe-stress estimate should be interpreted as a single-channel lower bound rather than a full CCAR-style capital forecast.

---

## Model-Risk Diagnostic Journey

An earlier project design used a terminal-status, resolved-loans-only lifetime-PD outcome. That design produced an implausible loss escalation (A/E multiplier) in a calm, low-unemployment period. The diagnostic process separated seasoning, vintage, and macro explanations and showed that the escalation was not a seasoning artifact. This traced the issue back to the outcome definition: the terminal-status construction introduced survivorship/truncation bias as the data window closed. The project was therefore rebuilt around a fixed 24-month outcome, which removed the spurious late-window escalation and produced a more stable macro satellite.


![Figure L.1 — Contrast of Multiplier Paths, Legacy vs Redesign](legacy/figureL.1_multiplier_contrast.png)
*Figure L.1 — A/E multiplier, legacy vs. redesign. The legacy terminal-status multiplier escalates to ~4.7 by 2018; over the same window the redesign's 24-month fixed-horizon multiplier stays near 1.0. Open markers denote partial-coverage quarters excluded from estimation. Shared axes.*

---

## How to Run

### Full modeling workflow

1. Clone the repository.

2. Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

3. Raw data files are not committed to the repository because of size.  
Download the raw data [here](https://www.dropbox.com/scl/fi/k9gc2ny7ldzhcx4mt7gtc/accepted_2007_to_2018Q4.csv?rlkey=ox2c5xr2imxnlcgf13e4kix4r&st=qyf7ouqu&dl=0), then place the file in the project directory.

4. Run the notebooks in order:

```text
01_pd_model.ipynb
02_macro_satellite.ipynb
03_scenario_engine.ipynb
```

The legacy notebook does not produce final results but documents the model-risk diagnosis that led to the final fixed-horizon design.

### Refactored Component 3 scenario engine

Component 3 can also be run as a standalone Python pipeline:

```bash
python main.py
```

The refactored engine attempts to load its two prepared inputs from Amazon S3 and writes the seven verified Component 3 outputs—three CSV tables and four PNG figures—to the local `artifacts_component3/` directory and to the configured S3 `outputs/` prefix.

When run on EC2, S3 access is provided through the EC2 IAM instance role rather than embedded AWS credentials.

---

## Main Report

For the full technical narrative, see:

[WRITEUP.md](WRITEUP.md)

The writeup contains the complete model documentation: cohort construction, outcome definition, PD specification, validation, calibration, macro satellite construction, scenario methodology, results, limitations, and the legacy model diagnostic journey.

---


