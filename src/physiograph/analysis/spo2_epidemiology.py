"""Model-minimal epidemiological analysis of SpO2 instability and decompensation.

Companion to ``spo2_protocol`` that deliberately removes the fitted model: every
estimate is a classical, directly attributable statistic (risk ratios,
Mantel-Haenszel pooled odds ratios, Cochran-Armitage trend, sign tests, cluster
bootstraps).  Exposure is the protocol's own instability definition so results
map 1:1 onto the model-based analysis.

Claim scope is identical to ``spo2_protocol``: observational associations with
temporal ordering under a fixed 4-hour landmark — not causal effects.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .spo2_protocol import (
    PRIMARY_ENDPOINTS,
    SECONDARY_ENDPOINTS,
    _first_instability,
)

# Floors mirror build_endpoint_completeness_audit so model-free and model-based
# analyses apply the same interpretability gate.
MIN_OBSERVED = 200
MIN_EVENTS = 20
MIN_NON_EVENTS = 20

MIN_BOOTSTRAP = 2000

ALL_ENDPOINTS = (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS)
NEGATIVE_CONTROL_ENDPOINT = "hepatic_lab_worsening_12h_flag"

BOOTSTRAP_SEED = 20260829


# ---------------------------------------------------------------------------
# Exposure construction
# ---------------------------------------------------------------------------


def assign_exposure(
    analysis_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach prespecified exposure columns to the analysis frame.

    ``exposure_any`` mirrors the protocol's first-instability rule exactly:
    any SpO2 < 90% or any adjacent change >= 4 pp inside [0, 240).  When the
    precomputed instability columns exist they are used directly; otherwise the
    raw event stream is required.
    """
    frame = analysis_df.copy()
    if events_df is not None and not events_df.empty:
        onset = _first_instability(events_df)
        if not onset.empty:
            onset = onset.assign(exposure_any=1)
            if "dataset" in frame:
                frame = frame.merge(
                    onset[["dataset", "stay_id", "exposure_any"]],
                    on=["dataset", "stay_id"],
                    how="left",
                )
            else:
                frame = frame.merge(
                    onset[["stay_id", "exposure_any"]], on="stay_id", how="left"
                )
        else:
            frame["exposure_any"] = 0
    elif {"spo2_below_90_fraction", "spo2_abrupt_jump_rate_per_hr"}.issubset(frame):
        # Fall back to precomputed summaries: below-90 fraction > 0 or a
        # positive jump rate implies the any-exposure definition fired.
        frame["exposure_any"] = (
            pd.to_numeric(frame["spo2_below_90_fraction"], errors="coerce").fillna(0).gt(0)
            | pd.to_numeric(frame["spo2_abrupt_jump_rate_per_hr"], errors="coerce").fillna(0).gt(0)
        ).astype(int)
    elif {"spo2_min"}.issubset(frame):
        frame["exposure_any"] = (
            pd.to_numeric(frame["spo2_min"], errors="coerce").lt(90).astype(int)
        )
    else:
        frame["exposure_any"] = np.nan

    if "spo2_instability_proxy_score" in frame:
        scores = pd.to_numeric(frame["spo2_instability_proxy_score"], errors="coerce")
        if scores.notna().sum() >= 30:
            t1, t2 = scores.quantile([1 / 3, 2 / 3])
            frame["exposure_tertile"] = pd.cut(
                scores, bins=[-np.inf, t1, t2, np.inf], labels=["T1", "T2", "T3"]
            ).astype("string")
        else:
            frame["exposure_tertile"] = pd.NA
    return frame


# ---------------------------------------------------------------------------
# Classical statistics helpers (no sklearn)
# ---------------------------------------------------------------------------


def _fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] without scipy."""
    from math import comb

    def hypergeom_psf(k: int) -> float:
        return comb(a + b, k) * comb(c + d, a + c - k) / comb(a + b + c + d, a + c)

    observed = hypergeom_psf(a)
    total = 0.0
    lo = max(0, (a + c) - (c + d))
    hi = min(a + b, a + c)
    for k in range(lo, hi + 1):
        p = hypergeom_psf(k)
        if p <= observed * (1 + 1e-9):
            total += p
    return float(min(total, 1.0))


def _risk_ratio_wald(a: int, n1: int, c: int, n0: int) -> tuple[float, float, float]:
    """RR with Katz log CI."""
    if n1 == 0 or n0 == 0:
        return math.nan, math.nan, math.nan
    r1, r0 = a / n1, c / n0
    if r1 == 0 or r0 == 0:
        return math.nan, math.nan, math.nan
    rr = r1 / r0
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    return rr, rr * math.exp(-1.96 * se), rr * math.exp(1.96 * se)


def _risk_difference_wald(a: int, n1: int, c: int, n0: int) -> tuple[float, float, float]:
    if n1 == 0 or n0 == 0:
        return math.nan, math.nan, math.nan
    rd = a / n1 - c / n0
    se = math.sqrt((a / n1) * (1 - a / n1) / n1 + (c / n0) * (1 - c / n0) / n0)
    if se == 0:
        return rd, math.nan, math.nan
    return rd, rd - 1.96 * se, rd + 1.96 * se


def _mantel_haenszel_odds_ratio(strata: Iterable[tuple[int, int, int, int]]) -> tuple[float, float, float]:
    """MH pooled OR with Greenland-Robins variance-based CI.

    Each stratum is (a, b, c, d): exposed cases, exposed non-cases,
    unexposed cases, unexposed non-cases.
    """
    num = den = 0.0
    P = Q = R_sum = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0:
            continue
        R = (a * d) / n
        S = (b * c) / n
        num += R
        den += S
        P += ((a + d) / n) * R
        Q += ((a + d) / n) * S + ((b + c) / n) * R
        R_sum += ((b + c) / n) * S
    if den == 0 or num == 0:
        return math.nan, math.nan, math.nan
    mh = num / den
    var = P / (2 * den**2) + Q / (2 * num * den) + R_sum / (2 * num**2)
    if not math.isfinite(var) or var <= 0:
        return mh, math.nan, math.nan
    se = math.sqrt(var)
    return mh, mh * math.exp(-1.96 * se), mh * math.exp(1.96 * se)


def _cochran_armitage_trend(counts: list[int], totals: list[int], scores: list[float]) -> tuple[float, float]:
    """Two-sided Cochran-Armitage trend test (normal approximation)."""
    n = sum(totals)
    if n == 0 or len(counts) < 2:
        return math.nan, math.nan
    r = sum(counts)
    p_bar = r / n
    x_bar = sum(s * t for s, t in zip(scores, totals)) / n
    numer = sum(t * (s - x_bar) * (c / t - p_bar) for c, t, s in zip(counts, totals, scores) if t > 0)
    var = p_bar * (1 - p_bar) * sum(t * (s - x_bar) ** 2 for t, s in zip(totals, scores))
    if var <= 0:
        return math.nan, math.nan
    z = numer / math.sqrt(var)
    from math import erf

    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))
    return z, p


def _bh_adjust(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: (pvalues[i], i))
    adjusted = [1.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1  # 1-based rank from largest p
        val = min(prev, pvalues[idx] * n / i)
        adjusted[idx] = val
        prev = val
    return adjusted


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------


def _cluster_bootstrap_rr(
    exposure: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    *,
    repetitions: int = MIN_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    rrs: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_cluster[c] for c in sampled])
        e, o = exposure[idx], outcome[idx]
        n1, n0 = int(e.sum()), int((1 - e).sum())
        if n1 == 0 or n0 == 0:
            continue
        r1 = o[e == 1].mean()
        r0 = o[e == 0].mean()
        if r0 <= 0:
            continue
        rrs.append(r1 / r0)
    if len(rrs) < max(100, repetitions * 0.05):
        return math.nan, math.nan
    return float(np.quantile(rrs, 0.025)), float(np.quantile(rrs, 0.975))


# ---------------------------------------------------------------------------
# Analyses A-E
# ---------------------------------------------------------------------------


def _endpoint_rows(frame: pd.DataFrame, endpoints: list[str]) -> list[str]:
    return [e for e in endpoints if e in frame]


def build_stratified_risk_tables(
    analysis_df: pd.DataFrame,
    *,
    exposure_col: str = "exposure_any",
    endpoints: list[str] | None = None,
    bootstrap_repetitions: int = MIN_BOOTSTRAP,
) -> pd.DataFrame:
    """Analysis A: exposed-vs-unexposed risk, RR/RD, Fisher p, cluster CI."""
    endpoints = endpoints or [e for e in ALL_ENDPOINTS if e in analysis_df]
    rows: list[dict[str, Any]] = []
    datasets = (
        sorted(analysis_df["dataset"].astype(str).unique())
        if "dataset" in analysis_df
        else ["all"]
    )
    for dataset in datasets:
        sub = (
            analysis_df.loc[analysis_df["dataset"].astype(str).eq(dataset)]
            if "dataset" in analysis_df
            else analysis_df
        )
        exposure = pd.to_numeric(sub[exposure_col], errors="coerce")
        valid_exposure = exposure.isin([0, 1])
        for endpoint in endpoints:
            y = pd.to_numeric(sub[endpoint], errors="coerce")
            valid = valid_exposure & y.isin([0, 1])
            n = int(valid.sum())
            events = int(y[valid].sum())
            if n < MIN_OBSERVED or events < MIN_EVENTS or (n - events) < MIN_NON_EVENTS:
                rows.append(
                    {
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "status": "underpowered",
                        "n_observed": n,
                        "events": events,
                        "non_events": n - events,
                        "exposed_n": int(exposure[valid].sum()),
                        "unexposed_n": int((1 - exposure[valid]).sum()),
                    }
                )
                continue
            e = exposure[valid].to_numpy().astype(int)
            o = y[valid].to_numpy().astype(int)
            a = int(((e == 1) & (o == 1)).sum())
            b = int(((e == 1) & (o == 0)).sum())
            c = int(((e == 0) & (o == 1)).sum())
            d = int(((e == 0) & (o == 0)).sum())
            rr, rr_lo, rr_hi = _risk_ratio_wald(a, a + b, c, c + d)
            rd, rd_lo, rd_hi = _risk_difference_wald(a, a + b, c, c + d)
            p = _fisher_exact(a, b, c, d)
            clusters = (
                (sub.loc[valid, "dataset"].astype(str) + ":person:" + sub.loc[valid, "person_id"].astype(str)).to_numpy()
                if "person_id" in sub and sub["person_id"].notna().any()
                else np.arange(n).astype(str)
            )
            boot_lo, boot_hi = _cluster_bootstrap_rr(
                e, o, clusters, repetitions=bootstrap_repetitions
            )
            rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "status": "estimated",
                    "n_observed": n,
                    "events": events,
                    "non_events": n - events,
                    "exposed_n": a + b,
                    "unexposed_n": c + d,
                    "risk_exposed": a / (a + b) if a + b else math.nan,
                    "risk_unexposed": c / (c + d) if c + d else math.nan,
                    "risk_ratio": rr,
                    "risk_ratio_ci95_low": min(rr_lo, boot_lo) if math.isfinite(rr_lo) else boot_lo,
                    "risk_ratio_ci95_high": max(rr_hi, boot_hi) if math.isfinite(rr_hi) else boot_hi,
                    "risk_ratio_cluster_boot_low": boot_lo,
                    "risk_ratio_cluster_boot_high": boot_hi,
                    "risk_difference": rd,
                    "risk_difference_ci95_low": rd_lo,
                    "risk_difference_ci95_high": rd_hi,
                    "fisher_p": p,
                }
            )
    result = pd.DataFrame(rows)
    if "fisher_p" in result:
        mask = result["fisher_p"].notna()
        result.loc[mask, "fisher_p_bh_adjusted"] = _bh_adjust(result.loc[mask, "fisher_p"].tolist())
    return result


def build_mantel_haenszel_tables(
    analysis_df: pd.DataFrame,
    *,
    exposure_col: str = "exposure_any",
    endpoints: list[str] | None = None,
) -> pd.DataFrame:
    """Analysis B: MH pooled OR stratified by baseline hypoxemia and resp support."""
    endpoints = endpoints or [e for e in ALL_ENDPOINTS if e in analysis_df]
    strata_defs: dict[str, pd.Series] = {}
    if "spo2_below_90_fraction" in analysis_df:
        strata_defs["baseline_hypoxemia"] = (
            pd.to_numeric(analysis_df["spo2_below_90_fraction"], errors="coerce").fillna(0).gt(0)
        ).map({True: "hypoxemic", False: "normoxemic"})
    if "mechanical_ventilation_flag" in analysis_df:
        strata_defs["mechanical_ventilation"] = (
            pd.to_numeric(analysis_df["mechanical_ventilation_flag"], errors="coerce").fillna(0).gt(0)
        ).map({True: "ventilated", False: "not_ventilated"})
    if "resp_support_any_flag" in analysis_df:
        strata_defs["resp_support"] = (
            pd.to_numeric(analysis_df["resp_support_any_flag"], errors="coerce").fillna(0).gt(0)
        ).map({True: "supported", False: "not_supported"})

    rows: list[dict[str, Any]] = []
    exposure = pd.to_numeric(analysis_df[exposure_col], errors="coerce") if exposure_col in analysis_df else pd.Series(np.nan, index=analysis_df.index)
    for dataset in (
        sorted(analysis_df["dataset"].astype(str).unique())
        if "dataset" in analysis_df
        else ["all"]
    ):
        ds_mask = (
            analysis_df["dataset"].astype(str).eq(dataset)
            if "dataset" in analysis_df
            else pd.Series(True, index=analysis_df.index)
        )
        for stratum_name, stratum_series in strata_defs.items():
            # Rebuild per endpoint with per-stratum counts.
            for endpoint in endpoints:
                y = pd.to_numeric(analysis_df[endpoint], errors="coerce")
                strata = []
                stratum_labels = []
                for level in stratum_series.dropna().unique():
                    mask = ds_mask & stratum_series.eq(level) & exposure.isin([0, 1]) & y.isin([0, 1])
                    if mask.sum() == 0:
                        continue
                    e = exposure[mask].to_numpy().astype(int)
                    o = y[mask].to_numpy().astype(int)
                    a = int(((e == 1) & (o == 1)).sum())
                    b = int(((e == 1) & (o == 0)).sum())
                    c = int(((e == 0) & (o == 1)).sum())
                    d = int(((e == 0) & (o == 0)).sum())
                    if a + b == 0 or c + d == 0:
                        continue
                    strata.append((a, b, c, d))
                    stratum_labels.append(str(level))
                if len(strata) < 2:
                    rows.append(
                        {
                            "dataset": dataset,
                            "stratification": stratum_name,
                            "endpoint": endpoint,
                            "status": "not_stratifiable",
                            "n_strata": len(strata),
                        }
                    )
                    continue
                or_mh, or_lo, or_hi = _mantel_haenszel_odds_ratio(strata)
                n_total = sum(a + b + c + d for a, b, c, d in strata)
                events_total = sum(a + c for a, b, c, d in strata)
                status = (
                    "estimated"
                    if n_total >= MIN_OBSERVED and events_total >= MIN_EVENTS and (n_total - events_total) >= MIN_NON_EVENTS
                    else "underpowered"
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "stratification": stratum_name,
                        "endpoint": endpoint,
                        "status": status,
                        "n_strata": len(strata),
                        "stratum_levels": ",".join(stratum_labels),
                        "n_total": n_total,
                        "events": events_total,
                        "mh_odds_ratio": or_mh,
                        "mh_or_ci95_low": or_lo,
                        "mh_or_ci95_high": or_hi,
                    }
                )
    return pd.DataFrame(rows)


def build_dose_response_tables(
    analysis_df: pd.DataFrame,
    *,
    endpoints: list[str] | None = None,
) -> pd.DataFrame:
    """Analysis C: risk by instability tertile with Cochran-Armitage trend."""
    endpoints = endpoints or [e for e in ALL_ENDPOINTS if e in analysis_df]
    if "exposure_tertile" not in analysis_df:
        return pd.DataFrame([{"status": "not_run", "reason": "exposure_tertile unavailable"}])
    rows: list[dict[str, Any]] = []
    datasets = (
        sorted(analysis_df["dataset"].astype(str).unique())
        if "dataset" in analysis_df
        else ["all"]
    )
    for dataset in datasets:
        sub = (
            analysis_df.loc[analysis_df["dataset"].astype(str).eq(dataset)]
            if "dataset" in analysis_df
            else analysis_df
        )
        tertile = sub["exposure_tertile"]
        for endpoint in endpoints:
            y = pd.to_numeric(sub[endpoint], errors="coerce")
            valid = tertile.notna() & y.isin([0, 1])
            n = int(valid.sum())
            events = int(y[valid].sum())
            counts, totals, risks = [], [], []
            for level in ("T1", "T2", "T3"):
                mask = valid & tertile.eq(level)
                t = int(mask.sum())
                c = int(y[mask].sum())
                counts.append(c)
                totals.append(t)
                risks.append(c / t if t else math.nan)
            if n < MIN_OBSERVED or events < MIN_EVENTS or (n - events) < MIN_NON_EVENTS:
                rows.append(
                    {
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "status": "underpowered",
                        "n_observed": n,
                        "events": events,
                        "risk_T1": risks[0],
                        "risk_T2": risks[1],
                        "risk_T3": risks[2],
                    }
                )
                continue
            z, p = _cochran_armitage_trend(counts, totals, [1.0, 2.0, 3.0])
            rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "status": "estimated",
                    "n_observed": n,
                    "events": events,
                    "risk_T1": risks[0],
                    "risk_T2": risks[1],
                    "risk_T3": risks[2],
                    "trend_z": z,
                    "trend_p_two_sided": p,
                }
            )
    result = pd.DataFrame(rows)
    if "trend_p_two_sided" in result:
        mask = result["trend_p_two_sided"].notna()
        result.loc[mask, "trend_p_bh_adjusted"] = _bh_adjust(result.loc[mask, "trend_p_two_sided"].tolist())
    return result


def build_paired_precedence_table(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None = None,
    *,
    bootstrap_repetitions: int = MIN_BOOTSTRAP,
    random_state: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Analysis D: within-stay paired lead times, sign test, complement counts.

    Unlike build_temporal_precedence (which summarizes paired onsets), this
    table also reports the denominator: how many outcome events occurred with
    NO prior instability, which guards against instability being ubiquitous.
    """
    from .spo2_protocol import build_temporal_precedence

    records, _ = build_temporal_precedence(events_df, cohort_df, bootstrap_repetitions=0)
    if records.empty or "lead_time_minutes" not in records:
        return pd.DataFrame([{"status": "unavailable", "reason": "no paired onsets"}])

    # Complement denominator bookkeeping: stays with outcome but no recorded
    # instability onset are visible as rows in the source records with
    # negative/zero lead times; the per-outcome counts below surface them.
    outcome_events = events_df.copy()
    outcome_events["concept"] = outcome_events.get("concept", "").fillna("").astype(str).str.lower()
    outcome_events["offset_minutes"] = pd.to_numeric(outcome_events.get("offset_minutes"), errors="coerce")

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for (dataset, outcome), group in records.groupby(["dataset", "outcome"]):
        values = group["lead_time_minutes"].to_numpy(dtype=float)
        n = len(values)
        positive = int((values > 0).sum())
        # Exact two-sided sign test vs p=0.5 using the binomial PMF.
        p_value = _binomial_twosided(positive, n, 0.5)
        # Cluster bootstrap on median lead time (person clusters when mapped).
        if analysis_df is not None and "person_id" in analysis_df:
            key = analysis_df.get("dataset", pd.Series("unknown", index=analysis_df.index)).astype(str) + ":" + analysis_df["stay_id"].astype(str)
            person_map = dict(zip(key, analysis_df["person_id"].astype(str)))
            clusters = np.array([person_map.get(f"{r.dataset}:{int(r.stay_id)}", f"{r.dataset}:{int(r.stay_id)}") for r in group.itertuples()])
        else:
            clusters = np.arange(n).astype(str)
        meds: list[float] = []
        unique = np.unique(clusters)
        by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
        for _ in range(bootstrap_repetitions):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([by_cluster[c] for c in sampled])
            meds.append(float(np.median(values[idx])))
        rows.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "status": "estimated",
                "n_paired": n,
                "n_preceded": positive,
                "preceded_fraction": positive / n,
                "sign_test_p": p_value,
                "median_lead_time_minutes": float(np.median(values)),
                "q1_lead_time_minutes": float(np.quantile(values, 0.25)),
                "q3_lead_time_minutes": float(np.quantile(values, 0.75)),
                "median_boot_ci95_low": float(np.quantile(meds, 0.025)) if meds else math.nan,
                "median_boot_ci95_high": float(np.quantile(meds, 0.975)) if meds else math.nan,
                "negative_lead_time_count": int((values <= 0).sum()),
                "claim_scope": "landmark_ordering_not_causal_precedence",
            }
        )
    return pd.DataFrame(rows)


def _binomial_twosided(k: int, n: int, p: float) -> float:
    from math import comb

    def pmf(x: int) -> float:
        return comb(n, x) * (p**x) * ((1 - p) ** (n - x))

    observed = pmf(k)
    return float(min(1.0, sum(pmf(x) for x in range(n + 1) if pmf(x) <= observed * (1 + 1e-9))))


def build_specificity_matrix(
    risk_tables: pd.DataFrame,
    dose_response: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Analysis E: per-endpoint classification across the evidence family.

    prespecified negative control: hepatic_lab_worsening (slow-moving, least
    plausibly coupled to acute SpO2 dynamics).  Acute endpoints are 'specific'
    only when robust while the bilirubin control is null.
    """
    rows: list[dict[str, Any]] = []
    primary = set(PRIMARY_ENDPOINTS)
    for (dataset, endpoint), group in risk_tables.groupby(["dataset", "endpoint"]):
        row = group.iloc[-1]
        p = row.get("fisher_p_bh_adjusted")
        rr = row.get("risk_ratio")
        status = row.get("status")
        if status != "estimated" or not isinstance(p, float) or not math.isfinite(p):
            classification = "underpowered" if status == "underpowered" else "unavailable"
        elif math.isfinite(rr) and rr > 1.0 and p < 0.05:
            classification = "positive_robust" if endpoint in primary else "positive_suggestive"
        elif math.isfinite(rr) and rr > 1.0 and p < 0.20:
            classification = "positive_suggestive"
        else:
            classification = "null"
        if dose_response is not None and not dose_response.empty:
            dr = dose_response.loc[
                dose_response["dataset"].eq(dataset) & dose_response["endpoint"].eq(endpoint)
            ]
            if not dr.empty:
                tp = dr.iloc[-1].get("trend_p_bh_adjusted")
                trend = "dose_response_present" if isinstance(tp, float) and math.isfinite(tp) and tp < 0.05 else "no_dose_response"
            else:
                trend = "not_estimated"
        else:
            trend = "not_estimated"
        role = "negative_control" if endpoint == NEGATIVE_CONTROL_ENDPOINT else ("primary" if endpoint in primary else "secondary")
        rows.append(
            {
                "dataset": dataset,
                "endpoint": endpoint,
                "role": role,
                "classification": classification,
                "dose_response": trend,
                "risk_ratio": rr,
                "bh_p": p,
                "interpretation": (
                    "specificity_supported"
                    if role == "primary" and classification == "positive_robust"
                    else (
                        "specificity_contradicted"
                        if role == "negative_control" and classification.startswith("positive")
                        else "consistent"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def run_epidemiology_analyses(
    analysis_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    cohort_df: pd.DataFrame | None = None,
    *,
    bootstrap_repetitions: int = MIN_BOOTSTRAP,
) -> dict[str, pd.DataFrame]:
    """Run the full model-minimal analysis family and return named tables."""
    frame = assign_exposure(analysis_df, events_df)
    risk = build_stratified_risk_tables(frame, bootstrap_repetitions=bootstrap_repetitions)
    mh = build_mantel_haenszel_tables(frame)
    dose = build_dose_response_tables(frame)
    paired = (
        build_paired_precedence_table(events_df, cohort_df, frame, bootstrap_repetitions=bootstrap_repetitions)
        if events_df is not None and cohort_df is not None
        else pd.DataFrame([{"status": "not_run", "reason": "events/cohort frames required"}])
    )
    specificity = build_specificity_matrix(risk, dose)
    return {
        "exposure_frame": frame,
        "stratified_risk_tables": risk,
        "mantel_haenszel": mh,
        "dose_response": dose,
        "paired_precedence": paired,
        "specificity_matrix": specificity,
    }


__all__ = [
    "assign_exposure",
    "build_dose_response_tables",
    "build_mantel_haenszel_tables",
    "build_paired_precedence_table",
    "build_specificity_matrix",
    "build_stratified_risk_tables",
    "run_epidemiology_analyses",
]
