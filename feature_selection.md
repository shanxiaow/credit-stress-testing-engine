# Feature Selection Rationale

Of the **151 columns** in the Lending Club dataset, a lean set of **28 columns** is retained — **25 predictors**, the `loan_status` field (source of the modeled outcome), `issue_d` (time grouping and the vintage-based train/validation/OOT split), and `last_pymnt_d` (carried only for outcome-timing construction and the macro linkage step — never a model feature). All other columns are excluded for the reasons documented below.

> **Guiding principle:** exclusion is *conservative* — wherever a field's measurement timing is ambiguous, or a field relates to the outcome itself, it is dropped. The cost of data leakage (a falsely inflated, indefensible model) far outweighs the value of any single additional feature.

> **Outcome note (v4 design):** the modeled target is **not** terminal loan status. `loan_status` is transformed into a **fixed-horizon** indicator, `default_24m` = 1 if the loan enters a default/charge-off status **within 24 months** of origination, else 0. Loans are included only if at least 24 months of follow-up are observable; ambiguous in-progress statuses (e.g., grace period, late) are excluded from the target; default timing is approximated from `last_pymnt_d`. Full construction lives in the modeling notebook (`PD4.ipynb`); this document covers column selection only.

---

## Retained Columns (28)

**Outcome source (1)**
- `loan_status` — transformed into the fixed-horizon `default_24m` target (see outcome note above)

**Time grouping & split (1)**
- `issue_d` — loan issue date; used to group loans by period for the macro linkage in Component 2, and to define the vintage-based train (<2015) / validation (2015) / OOT (>=2016) split

**Carried for outcome construction — non-feature (1)**
- `last_pymnt_d` — last payment date; used solely as a proxy for default/exit timing in the 24-month outcome and for the macro linkage. **Excluded from the feature matrix** (it is post-origination and would be leakage if used as a predictor).

**Loan terms (7)**
- `loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `sub_grade`, `purpose`

**Borrower financials (5)**
- `annual_inc`, `dti`, `emp_length`, `home_ownership`, `verification_status`

**Credit history — all backward-looking from application (13)**
- `fico_range_low`, `fico_range_high`, `earliest_cr_line` (engineered into credit-history length in **months**, `credit_history_months`), `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `pub_rec_bankruptcies`, `mort_acc`, `acc_open_past_24mths`

*The 25 predictors are the loan-terms, borrower-financials, and credit-history blocks. Two of them are dropped before modeling (see "Later Excluded" below): `grade` and one of the two FICO bounds.*

---

## Excluded Variables

### Leakage — information unavailable at the time of prediction

These fields reflect **post-origination** information that would not be known when a real PD prediction is made at underwriting. Several directly encode the very outcome being predicted.

- **Payment / balance outcomes** — how the loan was actually repaid: `out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`, `next_pymnt_d`
  - *Note:* `last_pymnt_d` is the one post-origination field that is **loaded** — but only to time the outcome, never as a predictor (see "Carried for outcome construction" above).
- **Post-origination / snapshot-dated credit updates:** `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low`, `acc_now_delinq` ("now" = post-origination snapshot; partly encodes default)
- **Ambiguous-timing distress events tied to the outcome** — "within 12 months" / "since" anchors not clearly *backward from application*: `chargeoff_within_12_mths`, `collections_12_mths_ex_med`, `mths_since_last_major_derog`
- **Hardship fields** (all `hardship_*`) — hardship plans arise *during* borrower distress
- **Settlement fields** (all `settlement_*`, `debt_settlement_flag`, `debt_settlement_flag_date`) — settlements occur *after* default

### Redundancy — duplicates of a retained feature

Dropped to avoid multicollinearity, which destabilizes logistic-regression coefficients and undermines interpretability.

- `funded_amnt`, `funded_amnt_inv` — effectively identical to the retained `loan_amnt` (requested ≈ funded for nearly all loans)

### Joint / Secondary Applicant — sparse for individual loans

Populated only for the small minority of joint applications, so overwhelmingly null.

- `annual_inc_joint`, `dti_joint`, `verification_status_joint`, `revol_bal_joint`, and all secondary-applicant fields (`sec_app_fico_range_low`, `sec_app_fico_range_high`, `sec_app_earliest_cr_line`, `sec_app_inq_last_6mths`, `sec_app_mort_acc`, `sec_app_open_acc`, `sec_app_revol_util`, `sec_app_open_act_il`, `sec_app_num_rev_accts`, `sec_app_chargeoff_within_12_mths`, `sec_app_collections_12_mths_ex_med`, `sec_app_mths_since_last_major_derog`)

### Identifiers / Non-Predictive / Free Text / Constant

No generalizable predictive signal — identifiers, free-form text, constant, or administrative fields.

- **Identifiers:** `id`, `member_id`, `url`
- **Free text / high-cardinality:** `desc`, `title`, `emp_title`, `zip_code`
- **Constant / administrative:** `policy_code` (constant), `pymnt_plan`, `initial_list_status`, `application_type`, `hardship_flag`, `disbursement_method`
- **Geography (optional drop):** `addr_state` — weak, high-cardinality predictor dropped for simplicity

### Detailed Bureau Attribute Block (`mths_since_*`, `num_*`, `mo_sin_*`, `il_*`, `bc_*`, etc.)

This large block (~60+ columns) consists of granular credit-bureau metrics. Most are *likely* measured at origination and are not strictly leakage, but they were excluded for three practical reasons:

1. **Ambiguous timing** — for several, it cannot be cleanly verified whether values reflect the application moment or a later snapshot.
2. **Heavy missingness** — many are sparsely populated.
3. **Model parsimony** — a focused, interpretable scorecard with ~25 well-understood features is more defensible, and more aligned with credit-modeling practice, than a bloated model with 100+ correlated inputs.

*A well-populated subset of these could be reintroduced as a future enhancement.*

---

## Later Excluded from Final Feature Set

These columns are loaded but dropped before (or excluded from) model fitting:

- **`grade`** — redundant with `sub_grade` at coarser granularity; excluded as a non-feature column alongside the target and time variables. Not used in any specification.
- **`sub_grade`** — LC's composite credit score. Excluded from the **primary** model to avoid proxying the lender's proprietary pricing signal; **added back only in the Sensitivity 2** specification to quantify how much rank-ordering power that proprietary signal carries.
- **`fico_range_low`, `fico_range_high`** — collapsed into a single midpoint feature (`fico_avg`); both raw bounds dropped after confirming near-perfect correlation.

*Net effect: of the 25 retained predictors, `grade` is dropped entirely and the two FICO bounds collapse to one, so the primary feature matrix is built from the remaining inputs plus the engineered `term_months`, `emp_length_years`, `credit_history_months`, and `fico_avg`. `sub_grade` enters only under Sensitivity 2.*
