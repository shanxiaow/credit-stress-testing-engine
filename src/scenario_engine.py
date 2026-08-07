"""Component 3: macro scenario stress-testing engine.

Refactored from 03_scenario_engine.ipynb so the component can be run as a
standalone Python module (locally or later on AWS).
"""

from pathlib import Path
from io import BytesIO
import os

import boto3
import matplotlib
matplotlib.use("Agg")  # Safe for headless/cloud execution.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORED_PATH = PROJECT_ROOT / "scored_loan_level_pd.parquet"
DEFAULT_COEF_PATH = PROJECT_ROOT / "satellite_coefficients_baseline.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts_component3"

# AWS/S3 configuration.
# The bucket name can be overridden later with the environment variable
# STRESS_TESTING_S3_BUCKET without changing the code.
S3_BUCKET = os.getenv(
    "STRESS_TESTING_S3_BUCKET",
    "stress-testing-468241617471-us-east-2-an",
)
S3_INPUT_PREFIX = "inputs"
S3_OUTPUT_PREFIX = "outputs"

FRED_CSV = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=UNRATE&cosd=2000-01-01&coed=2010-12-31"
)

# End-of-quarter unemployment-rate fallback values from the notebook.
_FALLBACK = {
    "rec": pd.Series(
        [3.9, 4.3, 4.5, 5.0, 5.7, 5.7, 5.8, 5.7, 6.0],
        index=pd.PeriodIndex(
            [
                "2000Q4", "2001Q1", "2001Q2", "2001Q3", "2001Q4",
                "2002Q1", "2002Q2", "2002Q3", "2002Q4",
            ],
            freq="Q",
        ),
    ),
    "gfc": pd.Series(
        [5.0, 5.1, 5.6, 6.1, 7.3, 8.7, 9.5, 9.8, 9.9],
        index=pd.PeriodIndex(
            [
                "2007Q4", "2008Q1", "2008Q2", "2008Q3", "2008Q4",
                "2009Q1", "2009Q2", "2009Q3", "2009Q4",
            ],
            freq="Q",
        ),
    ),
}

LGD = 0.80
REF_PORTFOLIO = 1_000_000_000
PD_COL = "pd24m_hat_calibrated"


def load_inputs_from_s3(
    bucket=S3_BUCKET,
    scored_key=f"{S3_INPUT_PREFIX}/scored_loan_level_pd.parquet",
    coef_key=f"{S3_INPUT_PREFIX}/satellite_coefficients_baseline.csv",
):
    """Load Component 3 inputs directly from Amazon S3."""
    s3 = boto3.client("s3")

    scored_obj = s3.get_object(Bucket=bucket, Key=scored_key)
    scored = pd.read_parquet(BytesIO(scored_obj["Body"].read()))

    coef_obj = s3.get_object(Bucket=bucket, Key=coef_key)
    coef = pd.read_csv(BytesIO(coef_obj["Body"].read()))

    return scored, coef


def load_inputs(
    scored_path=DEFAULT_SCORED_PATH,
    coef_path=DEFAULT_COEF_PATH,
    prefer_s3=True,
):
    """Load inputs from S3 first, with local files as a fallback."""
    if prefer_s3:
        try:
            scored, coef = load_inputs_from_s3()
            print(f"Input source: Amazon S3 ({S3_BUCKET}/{S3_INPUT_PREFIX}/)")
            return scored, coef
        except Exception as exc:
            print(
                f"[warn] S3 input load failed ({exc!r}); "
                "falling back to local files."
            )

    scored = pd.read_parquet(scored_path)
    coef = pd.read_csv(coef_path)
    print("Input source: local files")
    return scored, coef


def get_unemployment_beta(coef):
    """Read the baseline calibrated M1 unemployment transmission coefficient."""
    mask = (
        (coef["sample"] == "baseline_full_age_mix")
        & (coef["target"] == "calibrated")
        & (coef["model_id"] == "M1_unemp")
        & (coef["term"] == "unrate_diff_l1")
    )

    row = coef[mask]
    if len(row) != 1:
        raise ValueError(
            "Expected exactly one M1 calibrated unemployment coefficient, "
            f"found {len(row)}"
        )

    beta_unemp = float(row["coef"].iloc[0])
    return beta_unemp


def _quarterly_unrate():
    """Pull FRED UNRATE and aggregate to end-of-quarter values."""
    raw = pd.read_csv(FRED_CSV)
    raw.columns = ["date", "unrate"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["unrate"] = pd.to_numeric(raw["unrate"], errors="coerce")
    raw = raw.dropna()
    raw["q"] = raw["date"].dt.to_period("Q")
    return raw.groupby("q")["unrate"].last()


def load_episode_changes():
    """Return quarterly unemployment-change profiles for 2001 and the GFC."""
    try:
        q = _quarterly_unrate()
        gfc_q = q.loc["2007Q4":"2009Q4"]
        rec_q = q.loc["2000Q4":"2002Q4"]
        if len(gfc_q) != 9 or len(rec_q) != 9:
            raise ValueError("unexpected quarter count from FRED")
        source = "FRED UNRATE live pull (end-of-quarter)"
    except Exception as exc:
        print(
            f"[warn] FRED pull failed ({exc!r}); using rounded "
            "end-of-quarter fallback figures."
        )
        gfc_q = _FALLBACK["gfc"]
        rec_q = _FALLBACK["rec"]
        source = "end-of-quarter fallback (rounded 0.1)"

    gfc_changes = gfc_q.diff().dropna().to_numpy()
    rec_changes = rec_q.diff().dropna().to_numpy()

    return {
        "gfc_q": gfc_q,
        "rec_q": rec_q,
        "gfc_changes": gfc_changes,
        "rec_changes": rec_changes,
        "source": source,
    }


def build_scenarios(rec_changes, gfc_changes, u_start=4.0):
    """Build baseline, adverse, and severely-adverse unemployment paths."""
    horizon_q = len(gfc_changes)
    quarters = [f"Q{i}" for i in range(1, horizon_q + 1)]

    baseline_unemp = np.array([u_start] * horizon_q)
    adverse_unemp = np.round(u_start + np.cumsum(rec_changes), 2)
    severe_unemp = np.round(u_start + np.cumsum(gfc_changes), 2)

    scenario = pd.DataFrame(
        {
            "quarter": quarters,
            "baseline_unemp": baseline_unemp,
            "adverse_unemp": adverse_unemp,
            "severe_unemp": severe_unemp,
        }
    )

    for col in ["baseline_unemp", "adverse_unemp", "severe_unemp"]:
        scenario[col.replace("_unemp", "_dunemp")] = np.diff(
            np.concatenate([[u_start], scenario[col].to_numpy()])
        )

    paths = {
        "baseline": baseline_unemp,
        "adverse": adverse_unemp,
        "severe": severe_unemp,
    }

    return scenario, paths, quarters


def calculate_multipliers(scenario, paths, quarters, beta_unemp, u_start=4.0):
    """Calculate lagged quarterly stress multipliers and peak multiplier by scenario."""
    mt = pd.DataFrame(index=quarters)
    mt.index.name = "quarter"
    peak_mult = {}

    baseline_dunemp = scenario["baseline_dunemp"].to_numpy()

    for name, u in paths.items():
        dunemp = np.diff(np.concatenate([[u_start], u]))
        excess = dunemp - baseline_dunemp
        excess_lag1 = pd.Series(excess).shift(1).fillna(0.0).to_numpy()
        mt[name] = np.exp(beta_unemp * excess_lag1)
        peak_mult[name] = float(mt[name].max())

    return mt, peak_mult


def build_book(scored, pd_col=PD_COL):
    """Create the loan book used for scenario default-rate and EL calculations."""
    required = [pd_col, "pd24m_hat_raw", "loan_amnt"]
    missing = [c for c in required if c not in scored.columns]
    if missing:
        raise KeyError(f"Missing required scored-loan columns: {missing}")

    book = scored[required].copy()
    book = book.dropna(subset=[pd_col, "loan_amnt"])
    book["pd_baseline"] = book[pd_col]
    book["ead"] = book["loan_amnt"]
    return book


def annualize(cum_24m):
    """Convert a cumulative 24-month PD to a one-year equivalent under constant hazard."""
    return 1 - (1 - cum_24m) ** 0.5


def calculate_default_rates(book, paths, peak_mult):
    """Calculate mean portfolio default rates by scenario."""
    default_rate_24m = {}
    default_rate_annual = {}

    for name in paths:
        pd_stressed_24m = np.minimum(book["pd_baseline"] * peak_mult[name], 1.0)
        default_rate_24m[name] = float(pd_stressed_24m.mean())
        default_rate_annual[name] = float(annualize(pd_stressed_24m).mean())

    return default_rate_24m, default_rate_annual


def calculate_expected_losses(
    book,
    paths,
    peak_mult,
    lgd=LGD,
):
    """Calculate exposure-weighted expected-loss rates by scenario."""
    total_ead = book["ead"].sum()
    if total_ead <= 0:
        raise ValueError("Total EAD must be positive.")

    el_rate_24m = {}
    el_rate_annual = {}

    for name in paths:
        pd_stressed_24m = np.minimum(book["pd_baseline"] * peak_mult[name], 1.0)
        pd_stressed_annual = annualize(pd_stressed_24m)

        el_rate_24m[name] = float(
            (pd_stressed_24m * lgd * book["ead"]).sum() / total_ead
        )
        el_rate_annual[name] = float(
            (pd_stressed_annual * lgd * book["ead"]).sum() / total_ead
        )

    return el_rate_24m, el_rate_annual


def build_summary(
    paths,
    peak_mult,
    default_rate_24m,
    default_rate_annual,
    el_rate_24m,
    el_rate_annual,
    ref_portfolio=REF_PORTFOLIO,
):
    """Build the Component 3 scenario summary table."""
    default_rel_annual = {
        n: 100 * (default_rate_annual[n] / default_rate_annual["baseline"] - 1)
        for n in paths
    }
    default_rel_24m = {
        n: 100 * (default_rate_24m[n] / default_rate_24m["baseline"] - 1)
        for n in paths
    }
    el_rel_annual = {
        n: 100 * (el_rate_annual[n] / el_rate_annual["baseline"] - 1)
        for n in paths
    }
    el_rel_24m = {
        n: 100 * (el_rate_24m[n] / el_rate_24m["baseline"] - 1)
        for n in paths
    }

    summary = pd.DataFrame(
        {
            "peak_unemp_pct": {n: float(np.max(paths[n])) for n in paths},
            "peak_stress_multiplier": {n: peak_mult[n] for n in paths},
            "default_rate_annual_pct": {
                n: 100 * default_rate_annual[n] for n in paths
            },
            "default_rel_annual_pct": default_rel_annual,
            "el_rate_annual_pct": {n: 100 * el_rate_annual[n] for n in paths},
            "el_rel_annual_pct": el_rel_annual,
            "el_annual_per_1bn_$m": {
                n: el_rate_annual[n] * ref_portfolio / 1e6 for n in paths
            },
            "default_rate_24m_pct": {n: 100 * default_rate_24m[n] for n in paths},
            "default_rel_24m_pct": default_rel_24m,
            "el_rate_24m_pct": {n: 100 * el_rate_24m[n] for n in paths},
            "el_rel_24m_pct": el_rel_24m,
        }
    )
    summary.index.name = "scenario"

    base_ann = summary.loc["baseline", "el_annual_per_1bn_$m"]
    summary["incremental_el_annual_vs_baseline_$m"] = (
        summary["el_annual_per_1bn_$m"] - base_ann
    )

    return summary


def save_outputs(
    artifact_dir,
    scenario,
    mt,
    paths,
    quarters,
    default_rate_annual,
    el_rate_annual,
    summary,
    ref_portfolio=REF_PORTFOLIO,
):
    """Save the three Component 3 tables and four figures from the notebook."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Tables
    scenario.to_csv(
        artifact_dir / "table3.1_scenario_unemployment_paths.csv", index=False
    )
    mt.to_csv(artifact_dir / "table3.2_multiplier_paths.csv")
    summary.to_csv(artifact_dir / "table3.3_scenario_summary.csv")

    colors = {"baseline": "#4C72B0", "adverse": "#DD8452", "severe": "#C44E52"}
    labels = {
        "baseline": "Calm / baseline (flat 4.0%)",
        "adverse": "Adverse (2001 profile)",
        "severe": "Severely adverse (GFC profile)",
    }

    # Figure 3.1 — unemployment paths
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(quarters, scenario["baseline_unemp"], marker="o", color=colors["baseline"], label=labels["baseline"])
    ax.plot(quarters, scenario["adverse_unemp"], marker="o", color=colors["adverse"], label=labels["adverse"])
    ax.plot(quarters, scenario["severe_unemp"], marker="o", color=colors["severe"], label=labels["severe"])
    ax.set_ylabel("Unemployment rate (%)")
    ax.set_xlabel("Scenario quarter")
    ax.set_title("Component 3 - Scenario unemployment paths", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(artifact_dir / "figure3.1_scenario_unemployment_paths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3.2 — multiplier paths
    labels2 = {
        "baseline": "Calm / baseline",
        "adverse": "Adverse (2001 profile)",
        "severe": "Severely adverse (GFC profile)",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in paths:
        ax.plot(quarters, mt[name], marker="o", color=colors[name], label=labels2[name])
    ax.axhline(1.0, color="grey", lw=0.8, ls="--", label="no stress (mt = 1)")
    ax.set_ylabel("Stress multiplier (mt)")
    ax.set_xlabel("Scenario quarter")
    ax.set_title("Component 3 - Stress multiplier path by scenario severity", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(artifact_dir / "figure3.2_multiplier_paths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    order = ["baseline", "adverse", "severe"]
    xlabel = ["Calm / baseline", "Adverse", "Severely adverse"]

    # Figure 3.3 — annual default rate
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dvals = [100 * default_rate_annual[n] for n in order]
    bars = ax.bar(
        xlabel,
        dvals,
        color=[colors[n] for n in order],
        width=0.6,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, name in zip(bars, order):
        rel = 100 * (default_rate_annual[name] / default_rate_annual["baseline"] - 1)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}%/yr\n({rel:+.0f}% vs base)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Annual default rate (%/yr)")
    ax.set_title("Component 3 - Annual default rate by scenario severity", fontsize=11)
    ax.set_ylim(0, max(dvals) * 1.20)
    fig.tight_layout()
    fig.savefig(artifact_dir / "figure3.3_default_rate_by_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3.4 — annual expected loss per $1B
    fig, ax = plt.subplots(figsize=(7, 4.5))
    evals = [el_rate_annual[n] * ref_portfolio / 1e6 for n in order]
    bars = ax.bar(
        xlabel,
        evals,
        color=[colors[n] for n in order],
        width=0.6,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, name in zip(bars, order):
        rel = 100 * (el_rate_annual[name] / el_rate_annual["baseline"] - 1)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${bar.get_height():,.1f}M\n({100*el_rate_annual[name]:.1f}%/yr, {rel:+.0f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Annual EL per $1B exposure ($M/yr)")
    ax.set_title("Component 3 - Annual expected loss by scenario severity", fontsize=11)
    ax.set_ylim(0, max(evals) * 1.20)
    fig.tight_layout()
    fig.savefig(artifact_dir / "figure3.4_el_by_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def verify_outputs(artifact_dir):
    """Verify that all seven expected Component 3 artifacts were created."""
    expected = [
        "table3.1_scenario_unemployment_paths.csv",
        "table3.2_multiplier_paths.csv",
        "table3.3_scenario_summary.csv",
        "figure3.1_scenario_unemployment_paths.png",
        "figure3.2_multiplier_paths.png",
        "figure3.3_default_rate_by_scenario.png",
        "figure3.4_el_by_scenario.png",
    ]

    artifact_dir = Path(artifact_dir)
    present = {p.name for p in artifact_dir.iterdir()} if artifact_dir.exists() else set()
    missing = [name for name in expected if name not in present]

    if missing:
        raise RuntimeError(f"Missing expected Component 3 artifacts: {missing}")

    return expected


def upload_outputs_to_s3(
    artifact_dir,
    bucket=S3_BUCKET,
    prefix=S3_OUTPUT_PREFIX,
):
    """Upload all verified Component 3 artifacts to Amazon S3."""
    s3 = boto3.client("s3")
    artifact_dir = Path(artifact_dir)

    expected = verify_outputs(artifact_dir)
    for name in expected:
        local_path = artifact_dir / name
        key = f"{prefix}/{name}"
        s3.upload_file(str(local_path), bucket, key)
        print(f"Uploaded: s3://{bucket}/{key}")


def run_scenario_engine(
    scored_path=DEFAULT_SCORED_PATH,
    coef_path=DEFAULT_COEF_PATH,
    artifact_dir=DEFAULT_ARTIFACT_DIR,
    prefer_s3=True,
    upload_to_s3=True,
):
    """Run Component 3 end-to-end and return the scenario summary DataFrame."""
    print("=== Component 3: Scenario Engine ===")

    scored, coef = load_inputs(scored_path, coef_path, prefer_s3=prefer_s3)
    print(f"Loaded {len(scored):,} scored loans.")

    beta_unemp = get_unemployment_beta(coef)
    print(f"Transmission coefficient (M1 calibrated, Δunemployment): beta = {beta_unemp:.4f}")
    print(
        "Semi-elasticity: a +1pp larger rise in unemployment scales the "
        f"multiplier by exp(beta) = {np.exp(beta_unemp):.4f}"
    )

    episode = load_episode_changes()
    print(f"UNRATE source: {episode['source']}")

    scenario, paths, quarters = build_scenarios(
        episode["rec_changes"], episode["gfc_changes"]
    )
    print("\nScenario paths:")
    print(scenario.round(2).to_string(index=False))

    mt, peak_mult = calculate_multipliers(
        scenario, paths, quarters, beta_unemp
    )
    print("\nPeak single-quarter stress multipliers:")
    for name in paths:
        print(f"  {name:8s}: {peak_mult[name]:.4f}")

    book = build_book(scored)
    print(f"\nLoans in stress-testing book: {len(book):,}")

    default_rate_24m, default_rate_annual = calculate_default_rates(
        book, paths, peak_mult
    )
    el_rate_24m, el_rate_annual = calculate_expected_losses(
        book, paths, peak_mult
    )

    summary = build_summary(
        paths,
        peak_mult,
        default_rate_24m,
        default_rate_annual,
        el_rate_24m,
        el_rate_annual,
    )

    save_outputs(
        artifact_dir,
        scenario,
        mt,
        paths,
        quarters,
        default_rate_annual,
        el_rate_annual,
        summary,
    )
    verify_outputs(artifact_dir)

    if upload_to_s3:
        try:
            upload_outputs_to_s3(artifact_dir)
            print(
                f"Uploaded Component 3 artifacts to "
                f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/"
            )
        except Exception as exc:
            print(
                f"[warn] S3 output upload failed ({exc!r}); "
                "local outputs were preserved."
            )

    print("\nScenario summary (annual-equivalent primary):")
    print(
        summary[
            [
                "peak_unemp_pct",
                "peak_stress_multiplier",
                "default_rate_annual_pct",
                "default_rel_annual_pct",
                "el_rate_annual_pct",
                "el_rel_annual_pct",
                "el_annual_per_1bn_$m",
                "incremental_el_annual_vs_baseline_$m",
            ]
        ].round(2).to_string()
    )

    print(f"\nSaved Component 3 artifacts to: {Path(artifact_dir).resolve()}")
    return summary
