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
# Bilirubin worsening is a slower-moving specificity comparator, not a true
# negative control: hypoxemia/shock can plausibly affect it.  Keep the legacy
# constant as an alias for downstream compatibility while labeling it honestly.
SPECIFICITY_COMPARATOR_ENDPOINT = "hepatic_lab_worsening_12h_flag"
NEGATIVE_CONTROL_ENDPOINT = SPECIFICITY_COMPARATOR_ENDPOINT

BOOTSTRAP_SEED = 20260829


# ---------------------------------------------------------------------------
# Exposure construction
# ---------------------------------------------------------------------------


def assign_exposure(
    analysis_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach prespecified exposure columns to the analysis frame.

    ``exposure_any`` is the dynamics-only primary exposure: any adjacent
    absolute change >= 4 percentage points inside [0, 240), restricted to
    dynamics-eligible signals. ``exposure_hypoxemia_or_dynamics`` retains the
    broader descriptive composite (SpO2 < 90% or a qualifying jump).  When the
    precomputed columns exist they are used directly; otherwise raw events are
    required.
    """
    frame = analysis_df.copy().reset_index(drop=True)
    frame = frame.drop(
        columns=[
            "exposure_any",
            "exposure_hypoxemia_or_dynamics",
            "exposure_tertile",
            "exposure_tertile_score",
        ],
        errors="ignore",
    )
    if "spo2_plausible_count" in frame:
        measured = pd.to_numeric(
            frame["spo2_plausible_count"], errors="coerce"
        ).fillna(0).gt(0)
    else:
        measurement_proxies = [
            column
            for column in (
                "spo2_min",
                "spo2_below_90_fraction",
                "spo2_abrupt_jump_rate_per_hr",
                "spo2_dynamics_proxy_score",
                "spo2_instability_proxy_score",
            )
            if column in frame
        ]
        measured = (
            frame[measurement_proxies].notna().any(axis=1)
            if measurement_proxies
            else pd.Series(False, index=frame.index)
        )
    if events_df is not None and not events_df.empty:
        raw = events_df.copy()
        if "dataset" not in raw:
            raw["dataset"] = "unknown"
        raw["offset_minutes"] = pd.to_numeric(raw.get("offset_minutes"), errors="coerce")
        raw["value_numeric"] = pd.to_numeric(raw.get("value_numeric"), errors="coerce")
        measured_keys = raw.loc[
            raw.get("concept", pd.Series("", index=raw.index)).astype(str).str.lower().eq("spo2")
            & raw["offset_minutes"].ge(0)
            & raw["offset_minutes"].lt(240)
            & raw["value_numeric"].between(50, 100),
            ["dataset", "stay_id"],
        ].drop_duplicates().assign(_measured_from_events=1)
        if "dataset" in frame:
            frame = frame.merge(measured_keys, on=["dataset", "stay_id"], how="left")
        else:
            frame = frame.merge(
                measured_keys[["stay_id", "_measured_from_events"]].drop_duplicates(),
                on="stay_id",
                how="left",
            )
        measured = measured | frame.pop("_measured_from_events").fillna(0).gt(0)
        dynamics_onset = _first_instability(events_df, include_hypoxemia=False)
        combined_onset = _first_instability(events_df, include_hypoxemia=True)
        if not dynamics_onset.empty:
            dynamics_onset = dynamics_onset.assign(exposure_any=1)
            if "dataset" in frame:
                frame = frame.merge(
                    dynamics_onset[["dataset", "stay_id", "exposure_any"]],
                    on=["dataset", "stay_id"],
                    how="left",
                )
            else:
                frame = frame.merge(
                    dynamics_onset[["stay_id", "exposure_any"]], on="stay_id", how="left"
                )
        else:
            frame["exposure_any"] = np.nan
        if not combined_onset.empty:
            combined_onset = combined_onset.assign(
                exposure_hypoxemia_or_dynamics=1
            )
            key_columns = ["dataset", "stay_id"] if "dataset" in frame else ["stay_id"]
            frame = frame.merge(
                combined_onset[[*key_columns, "exposure_hypoxemia_or_dynamics"]],
                on=key_columns,
                how="left",
            )
        else:
            frame["exposure_hypoxemia_or_dynamics"] = np.nan
        dynamics_eligible = (
            pd.to_numeric(frame["spo2_dynamics_eligible_flag"], errors="coerce").eq(1)
            if "spo2_dynamics_eligible_flag" in frame
            else measured
        )
        frame["exposure_any"] = pd.to_numeric(
            frame["exposure_any"], errors="coerce"
        ).fillna(0).where(dynamics_eligible)
        frame["exposure_hypoxemia_or_dynamics"] = pd.to_numeric(
            frame["exposure_hypoxemia_or_dynamics"], errors="coerce"
        ).fillna(0).where(measured)
    elif {"spo2_below_90_fraction", "spo2_abrupt_jump_rate_per_hr"}.issubset(frame):
        jump = pd.to_numeric(
            frame["spo2_abrupt_jump_rate_per_hr"], errors="coerce"
        ).gt(0)
        dynamics_eligible = (
            pd.to_numeric(frame["spo2_dynamics_eligible_flag"], errors="coerce").eq(1)
            if "spo2_dynamics_eligible_flag" in frame
            else frame["spo2_abrupt_jump_rate_per_hr"].notna()
        )
        low = pd.to_numeric(
            frame["spo2_below_90_fraction"], errors="coerce"
        ).gt(0)
        frame["exposure_any"] = jump.astype(float).where(dynamics_eligible)
        frame["exposure_hypoxemia_or_dynamics"] = (low | jump).astype(float).where(measured)
    elif {"spo2_min"}.issubset(frame):
        frame["exposure_any"] = np.nan
        frame["exposure_hypoxemia_or_dynamics"] = pd.to_numeric(
            frame["spo2_min"], errors="coerce"
        ).lt(90).astype(float).where(measured)
    else:
        frame["exposure_any"] = np.nan
        frame["exposure_hypoxemia_or_dynamics"] = np.nan

    score_column = (
        "spo2_dynamics_proxy_score"
        if "spo2_dynamics_proxy_score" in frame
        else "spo2_instability_proxy_score"
    )
    if score_column in frame:
        frame["exposure_tertile"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame["exposure_tertile_score"] = score_column
        dataset_series = (
            frame["dataset"].fillna("unknown").astype(str)
            if "dataset" in frame
            else pd.Series("all", index=frame.index)
        )
        for _, index in dataset_series.groupby(dataset_series).groups.items():
            scores = pd.to_numeric(
                frame.loc[index, score_column], errors="coerce"
            )
            if "spo2_dynamics_eligible_flag" in frame:
                scores = scores.where(
                    pd.to_numeric(
                        frame.loc[index, "spo2_dynamics_eligible_flag"],
                        errors="coerce",
                    ).eq(1)
                )
            valid = scores.dropna()
            if len(valid) < 30:
                continue
            # Preserve ties: identical dynamics scores must never be assigned
            # to different exposure categories merely to balance group sizes.
            q1, q2 = valid.quantile([1 / 3, 2 / 3]).tolist()
            if not (math.isfinite(q1) and math.isfinite(q2) and q1 < q2):
                continue
            assigned = pd.Series(
                np.select(
                    [valid.le(q1), valid.le(q2)],
                    ["T1", "T2"],
                    default="T3",
                ),
                index=valid.index,
                dtype="string",
            )
            if set(assigned.dropna().unique()) != {"T1", "T2", "T3"}:
                continue
            frame.loc[valid.index, "exposure_tertile"] = assigned
    return frame


# ---------------------------------------------------------------------------
# Classical statistics helpers (no sklearn)
# ---------------------------------------------------------------------------


def _fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    try:
        from scipy.stats import fisher_exact

        return float(fisher_exact([[a, b], [c, d]], alternative="two-sided").pvalue)
    except Exception:  # pragma: no cover - exact dependency-free fallback
        pass

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
    if len(unique) == 0 or repetitions <= 0:
        return math.nan, math.nan
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
    if len(rrs) < max(100, repetitions * 0.80):
        return math.nan, math.nan
    return float(np.quantile(rrs, 0.025)), float(np.quantile(rrs, 0.975))


def _cluster_bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    repetitions: int = MIN_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, int]:
    """Patient-cluster bootstrap CI for a Spearman rank correlation."""
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    if len(unique) == 0 or repetitions <= 0:
        return math.nan, math.nan, 0
    by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_cluster[cluster] for cluster in sampled])
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        estimate = pd.Series(x[idx]).rank().corr(pd.Series(y[idx]).rank())
        if pd.notna(estimate) and math.isfinite(float(estimate)):
            estimates.append(float(estimate))
    valid = len(estimates)
    if valid < max(100, math.ceil(repetitions * 0.80)):
        return math.nan, math.nan, valid
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        valid,
    )


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
                status = "unavailable" if n == 0 else "underpowered"
                rows.append(
                    {
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "status": status,
                        "n_observed": n,
                        "events": events,
                        "non_events": n - events,
                        "exposed_n": int(exposure[valid].sum()),
                        "unexposed_n": int((1 - exposure[valid]).sum()),
                        "reason": (
                            "no jointly observed binary exposure and endpoint values"
                            if status == "unavailable"
                            else "insufficient rows/events/non-events"
                        ),
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
            if "person_id" in sub and sub["person_id"].notna().any():
                person = sub.loc[valid, "person_id"].astype("string")
                if "stay_id" in sub:
                    fallback = "stay:" + sub.loc[valid, "stay_id"].astype(
                        "string"
                    )
                else:
                    fallback = pd.Series(
                        [f"row:{index}" for index in range(n)],
                        index=person.index,
                        dtype="string",
                    )
                person = person.where(
                    person.notna() & person.str.strip().ne(""), fallback
                )
                clusters = (
                    sub.loc[valid, "dataset"].astype(str)
                    + ":person:"
                    + person.astype(str)
                ).to_numpy()
            else:
                clusters = np.arange(n).astype(str)
            boot_lo, boot_hi = _cluster_bootstrap_rr(
                e, o, clusters, repetitions=bootstrap_repetitions
            )
            selected_rr_lo = boot_lo if math.isfinite(boot_lo) else rr_lo
            selected_rr_hi = boot_hi if math.isfinite(boot_hi) else rr_hi
            estimable = all(
                math.isfinite(value)
                for value in (rr, selected_rr_lo, selected_rr_hi, rd, rd_lo, rd_hi, p)
            )
            rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "status": "estimated" if estimable else "non_estimable",
                    "n_observed": n,
                    "events": events,
                    "non_events": n - events,
                    "n_bootstrap_clusters": int(np.unique(clusters).size),
                    "exposed_n": a + b,
                    "unexposed_n": c + d,
                    "risk_exposed": a / (a + b) if a + b else math.nan,
                    "risk_unexposed": c / (c + d) if c + d else math.nan,
                    "risk_ratio": rr,
                    "risk_ratio_ci95_low": selected_rr_lo,
                    "risk_ratio_ci95_high": selected_rr_hi,
                    "risk_ratio_ci_method": (
                        "patient_cluster_bootstrap"
                        if math.isfinite(boot_lo) and math.isfinite(boot_hi)
                        else "katz_log_fallback" if estimable else "unavailable"
                    ),
                    "risk_ratio_katz_ci95_low": rr_lo,
                    "risk_ratio_katz_ci95_high": rr_hi,
                    "risk_ratio_cluster_boot_low": boot_lo,
                    "risk_ratio_cluster_boot_high": boot_hi,
                    "risk_difference": rd,
                    "risk_difference_ci95_low": rd_lo,
                    "risk_difference_ci95_high": rd_hi,
                    "fisher_p": p if estimable else math.nan,
                    "reason": "" if estimable else "risk-ratio or confidence-interval inference non-finite",
                }
            )
    result = pd.DataFrame(rows)
    if "fisher_p" in result:
        for dataset, index in result.groupby("dataset").groups.items():
            mask = result.index.isin(index) & result["fisher_p"].notna()
            result.loc[mask, "fisher_p_bh_adjusted"] = _bh_adjust(
                result.loc[mask, "fisher_p"].tolist()
            )
            result.loc[mask, "multiplicity_scope"] = (
                f"all_prespecified_endpoints_within_{dataset}"
            )
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
        value = pd.to_numeric(
            analysis_df["spo2_below_90_fraction"], errors="coerce"
        )
        strata_defs["baseline_hypoxemia"] = value.gt(0).map(
            {True: "hypoxemic", False: "normoxemic"}
        ).where(value.notna())
    if "mechanical_ventilation_flag" in analysis_df:
        value = pd.to_numeric(
            analysis_df["mechanical_ventilation_flag"], errors="coerce"
        )
        strata_defs["mechanical_ventilation"] = value.gt(0).map(
            {True: "ventilated", False: "not_ventilated"}
        ).where(value.notna())
    if "resp_support_any_flag" in analysis_df:
        value = pd.to_numeric(
            analysis_df["resp_support_any_flag"], errors="coerce"
        )
        strata_defs["resp_support"] = value.gt(0).map(
            {True: "supported", False: "not_supported"}
        ).where(value.notna())

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
                joint_valid = ds_mask & exposure.isin([0, 1]) & y.isin([0, 1])
                joint_n = int(joint_valid.sum())
                joint_events = int(y.loc[joint_valid].sum())
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
                    status = "unavailable" if joint_n == 0 else "not_stratifiable"
                    rows.append(
                        {
                            "dataset": dataset,
                            "stratification": stratum_name,
                            "endpoint": endpoint,
                            "status": status,
                            "n_strata": len(strata),
                            "stratum_levels": ",".join(stratum_labels),
                            "n_total": joint_n,
                            "events": joint_events,
                            "reason": (
                                "no jointly observed binary exposure and endpoint values"
                                if status == "unavailable"
                                else "fewer than two usable exposure-by-stratum tables"
                            ),
                        }
                    )
                    continue
                or_mh, or_lo, or_hi = _mantel_haenszel_odds_ratio(strata)
                n_total = sum(a + b + c + d for a, b, c, d in strata)
                events_total = sum(a + c for a, b, c, d in strata)
                sample_adequate = (
                    n_total >= MIN_OBSERVED
                    and events_total >= MIN_EVENTS
                    and (n_total - events_total) >= MIN_NON_EVENTS
                )
                estimator_adequate = all(
                    math.isfinite(value) for value in (or_mh, or_lo, or_hi)
                )
                status = (
                    "estimated"
                    if sample_adequate and estimator_adequate
                    else "underpowered" if not sample_adequate else "non_estimable"
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
                        "reason": (
                            ""
                            if status == "estimated"
                            else "insufficient rows/events/non-events"
                            if status == "underpowered"
                            else "Mantel-Haenszel estimate or confidence interval non-finite"
                        ),
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
                status = "unavailable" if n == 0 else "underpowered"
                rows.append(
                    {
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "status": status,
                        "n_observed": n,
                        "events": events,
                        "risk_T1": risks[0],
                        "risk_T2": risks[1],
                        "risk_T3": risks[2],
                        "reason": (
                            "no jointly observed instability tertile and endpoint values"
                            if status == "unavailable"
                            else "insufficient rows/events/non-events"
                        ),
                    }
                )
                continue
            z, p = _cochran_armitage_trend(counts, totals, [1.0, 2.0, 3.0])
            estimable = (
                all(total > 0 for total in totals)
                and all(math.isfinite(risk) for risk in risks)
                and all(math.isfinite(value) for value in (z, p))
            )
            rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "status": "estimated" if estimable else "non_estimable",
                    "n_observed": n,
                    "events": events,
                    "risk_T1": risks[0],
                    "risk_T2": risks[1],
                    "risk_T3": risks[2],
                    "trend_z": z,
                    "trend_p_two_sided": p if estimable else math.nan,
                    "reason": "" if estimable else "empty tertile or non-finite trend inference",
                }
            )
    result = pd.DataFrame(rows)
    if "trend_p_two_sided" in result:
        for dataset, index in result.groupby("dataset").groups.items():
            mask = result.index.isin(index) & result[
                "trend_p_two_sided"
            ].notna()
            result.loc[mask, "trend_p_bh_adjusted"] = _bh_adjust(
                result.loc[mask, "trend_p_two_sided"].tolist()
            )
            result.loc[mask, "multiplicity_scope"] = (
                f"all_prespecified_endpoints_within_{dataset}"
            )
    return result


def _attach_event_exposure(
    frame: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    column: str,
    lower_bound: float | None,
    jump_threshold: float,
) -> pd.DataFrame:
    """Attach an event-derived exposure while preserving unmeasured as NaN."""
    result = frame.copy()
    events = events_df.copy()
    if "dataset" not in events:
        events["dataset"] = "unknown"
    events["offset_minutes"] = pd.to_numeric(events.get("offset_minutes"), errors="coerce")
    events["value_numeric"] = pd.to_numeric(events.get("value_numeric"), errors="coerce")
    concept = events.get("concept", pd.Series("", index=events.index)).astype(str).str.lower()
    measured_mask = (
        concept.eq("spo2")
        & events["offset_minutes"].ge(0)
        & events["offset_minutes"].lt(240)
        & events["value_numeric"].le(100)
    )
    if lower_bound is not None:
        measured_mask &= events["value_numeric"].ge(lower_bound)
    measured_keys = events.loc[measured_mask, ["dataset", "stay_id"]].drop_duplicates()
    onset = _first_instability(
        events,
        lower_bound=lower_bound,
        jump_threshold=jump_threshold,
    )
    key_cols = ["dataset", "stay_id"] if "dataset" in result else ["stay_id"]
    measured_keys = measured_keys[key_cols].drop_duplicates().assign(_measured=1)
    onset_keys = onset[key_cols].drop_duplicates().assign(_exposed=1)
    result = result.merge(measured_keys, on=key_cols, how="left")
    result = result.merge(onset_keys, on=key_cols, how="left")
    result[column] = pd.to_numeric(result.pop("_exposed"), errors="coerce").fillna(0).where(
        pd.to_numeric(result.pop("_measured"), errors="coerce").fillna(0).gt(0)
    )
    return result


def build_sensitivity_matrix(
    analysis_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    *,
    bootstrap_repetitions: int = MIN_BOOTSTRAP,
) -> pd.DataFrame:
    """Execute every prespecified S1-S12 sensitivity axis.

    Binary rows use directly interpretable exposed/unexposed risks and Fisher
    tests. Continuous lactate rows use Spearman rank association. Each missing
    source/stratifier is emitted as ``unavailable`` rather than disappearing.
    """
    frame = assign_exposure(analysis_df, events_df)
    if "dataset" not in frame:
        frame["dataset"] = "unknown"
    if events_df is not None and not events_df.empty:
        frame = _attach_event_exposure(
            frame,
            events_df,
            column="exposure_strict_gt70",
            lower_bound=np.nextafter(70.0, np.inf),
            jump_threshold=4.0,
        )
        frame = _attach_event_exposure(
            frame,
            events_df,
            column="exposure_raw",
            lower_bound=None,
            jump_threshold=4.0,
        )

    numeric = lambda column: pd.to_numeric(frame[column], errors="coerce")
    if "spo2_jump_3_count" in frame and "spo2_below_90_fraction" in frame:
        frame["exposure_jump3"] = numeric("spo2_jump_3_count").gt(0).astype(float).where(
            numeric("spo2_dynamics_eligible_flag").eq(1)
            if "spo2_dynamics_eligible_flag" in frame
            else numeric("spo2_jump_3_count").notna()
        )
    if "spo2_jump_5_count" in frame and "spo2_below_90_fraction" in frame:
        frame["exposure_jump5"] = numeric("spo2_jump_5_count").gt(0).astype(float).where(
            numeric("spo2_dynamics_eligible_flag").eq(1)
            if "spo2_dynamics_eligible_flag" in frame
            else numeric("spo2_jump_5_count").notna()
        )
    if "spo2_sustained_desaturation_episode_count" in frame:
        frame["exposure_sustained_low"] = numeric(
            "spo2_sustained_desaturation_episode_count"
        ).gt(0).astype(float).where(
            numeric("spo2_dynamics_eligible_flag").eq(1)
            if "spo2_dynamics_eligible_flag" in frame
            else True
        )
    if "spo2_sustained_abrupt_jump_episode_count" in frame:
        frame["exposure_sustained_jump"] = numeric(
            "spo2_sustained_abrupt_jump_episode_count"
        ).gt(0).astype(float).where(
            numeric("spo2_dynamics_eligible_flag").eq(1)
            if "spo2_dynamics_eligible_flag" in frame
            else True
        )
    for dataset, index in frame.groupby(frame["dataset"].astype(str)).groups.items():
        if "spo2_rmssd" in frame:
            values = numeric("spo2_rmssd").loc[index]
            median = values.median()
            frame.loc[index, "exposure_rmssd_high"] = values.ge(median).astype(float).where(values.notna())
        if "spo2_sampling_density_per_hr" in frame:
            values = numeric("spo2_sampling_density_per_hr").loc[index]
            median = values.median()
            frame.loc[index, "exposure_sampling_high"] = values.ge(median).astype(float).where(values.notna())

    endpoints = [endpoint for endpoint in ALL_ENDPOINTS if endpoint in frame]
    rows: list[dict[str, Any]] = []

    def unavailable(dataset: str, sensitivity: str, axis: str, reason: str) -> None:
        rows.append(
            {
                "dataset": dataset,
                "sensitivity": sensitivity,
                "axis": axis,
                "status": "unavailable",
                "reason": reason,
            }
        )

    def evaluate(
        dataset: str,
        sensitivity: str,
        axis: str,
        level: str,
        subset: pd.DataFrame,
        exposure_col: str,
        selected_endpoints: list[str] | None = None,
    ) -> None:
        if exposure_col not in subset:
            unavailable(dataset, sensitivity, axis, f"{exposure_col} unavailable")
            return
        exposure = pd.to_numeric(subset[exposure_col], errors="coerce")
        endpoint_list = endpoints if selected_endpoints is None else selected_endpoints
        for endpoint in endpoint_list:
            y = pd.to_numeric(subset[endpoint], errors="coerce")
            valid = exposure.isin([0, 1]) & y.isin([0, 1])
            n = int(valid.sum())
            event_count = int(y.loc[valid].sum())
            e = exposure.loc[valid].astype(int).to_numpy()
            o = y.loc[valid].astype(int).to_numpy()
            a = int(((e == 1) & (o == 1)).sum())
            b = int(((e == 1) & (o == 0)).sum())
            c = int(((e == 0) & (o == 1)).sum())
            d = int(((e == 0) & (o == 0)).sum())
            sample_adequate = (
                n >= MIN_OBSERVED
                and event_count >= MIN_EVENTS
                and n - event_count >= MIN_NON_EVENTS
                and a + b > 0
                and c + d > 0
            )
            rr, low, high = _risk_ratio_wald(a, a + b, c, c + d)
            estimator_adequate = all(math.isfinite(value) for value in (rr, low, high))
            adequate = sample_adequate and estimator_adequate
            reason = ""
            if n == 0:
                reason = "no jointly observed binary exposure and endpoint values"
            elif not sample_adequate:
                reason = "insufficient rows/events/non-events/exposure groups"
            elif not estimator_adequate:
                reason = "risk ratio unidentified because an exposure arm has zero events"
            rows.append(
                {
                    "dataset": dataset,
                    "sensitivity": sensitivity,
                    "axis": axis,
                    "level": level,
                    "exposure": exposure_col,
                    "endpoint": endpoint,
                    "analysis_type": "binary_risk_ratio",
                    "status": (
                        "estimated"
                        if adequate
                        else "unavailable"
                        if n == 0
                        else "underpowered_or_unidentified"
                    ),
                    "n": n,
                    "events": event_count,
                    "exposed_n": a + b,
                    "unexposed_n": c + d,
                    "risk_exposed": a / (a + b) if a + b else math.nan,
                    "risk_unexposed": c / (c + d) if c + d else math.nan,
                    "effect": rr,
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": _fisher_exact(a, b, c, d) if adequate else math.nan,
                    "reason": reason,
                }
            )

    for dataset in sorted(frame["dataset"].astype(str).unique()):
        ds = frame.loc[frame["dataset"].astype(str).eq(dataset)].copy()
        # S1: both fixed horizons.
        evaluate(dataset, "S1", "outcome_horizon", "12h_and_24h", ds, "exposure_any")
        # S2: lactate threshold/max variants and continuous delta.
        lactate_variants = [
            endpoint
            for endpoint in (
                "lactate_rise_12h_flag",
                "lactate_rise_24h_flag",
                "lactate_rise_1_0_12h_flag",
                "lactate_rise_1_0_24h_flag",
                "lactate_max_rise_12h_flag",
                "lactate_max_rise_24h_flag",
            )
            if endpoint in ds
        ]
        evaluate(dataset, "S2", "lactate_definition", "binary_variants", ds, "exposure_any", lactate_variants)
        for delta in ("lactate_delta_12h", "lactate_delta_24h"):
            score_column = (
                "spo2_dynamics_proxy_score"
                if "spo2_dynamics_proxy_score" in ds
                else "spo2_instability_proxy_score"
            )
            if delta not in ds or score_column not in ds:
                continue
            x = pd.to_numeric(ds[score_column], errors="coerce")
            y = pd.to_numeric(ds[delta], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() >= 20:
                try:
                    from scipy.stats import spearmanr

                    effect, p_value = spearmanr(x.loc[valid], y.loc[valid])
                except Exception:
                    effect = x.loc[valid].rank().corr(y.loc[valid].rank())
                    p_value = math.nan
                if "person_id" in ds and ds.loc[valid, "person_id"].notna().any():
                    person = ds.loc[valid, "person_id"].astype("string")
                    if "stay_id" in ds:
                        fallback = "stay:" + ds.loc[valid, "stay_id"].astype("string")
                    else:
                        fallback = pd.Series(
                            [f"row:{index}" for index in range(int(valid.sum()))],
                            index=person.index,
                            dtype="string",
                        )
                    person = person.where(
                        person.notna() & person.str.strip().ne(""), fallback
                    )
                    clusters = (
                        ds.loc[valid, "dataset"].astype(str)
                        + ":person:"
                        + person.astype(str)
                    ).to_numpy()
                else:
                    clusters = np.arange(int(valid.sum())).astype(str)
                ci_low, ci_high, bootstrap_valid = _cluster_bootstrap_spearman(
                    x.loc[valid].to_numpy(dtype=float),
                    y.loc[valid].to_numpy(dtype=float),
                    clusters,
                    repetitions=bootstrap_repetitions,
                )
                estimated = all(
                    math.isfinite(float(value))
                    for value in (effect, p_value, ci_low, ci_high)
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "sensitivity": "S2",
                        "axis": "lactate_definition",
                        "level": delta,
                        "endpoint": delta,
                        "analysis_type": "continuous_spearman",
                        "status": "estimated" if estimated else "underpowered_or_unidentified",
                        "n": int(valid.sum()),
                        "effect": float(effect),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "p_value": float(p_value) if estimated else math.nan,
                        "bootstrap_repetitions_requested": bootstrap_repetitions,
                        "bootstrap_repetitions_valid": bootstrap_valid,
                        "reason": "" if estimated else "cluster bootstrap confidence interval unavailable",
                    }
                )

        stratifiers: list[tuple[str, str, str, pd.Series | None]] = []
        if "spo2_sampling_density_per_hr" in ds:
            value = pd.to_numeric(ds["spo2_sampling_density_per_hr"], errors="coerce")
            stratifiers.extend(
                [("S3", "sampling_density", "below_median", value.lt(value.median())), ("S3", "sampling_density", "at_or_above_median", value.ge(value.median()))]
            )
        else:
            stratifiers.append(("S3", "sampling_density", "unavailable", None))
        for sid, column, axis in (
            ("S4", "mechanical_ventilation_flag", "mechanical_ventilation"),
            ("S5", "resp_support_any_flag", "respiratory_support"),
            ("S6", "spo2_below_90_fraction", "baseline_hypoxemia"),
            ("S7", "baseline_lactate", "baseline_lactate"),
            ("S8", "baseline_vasoactive_flag", "baseline_vasoactive"),
        ):
            if column not in ds:
                stratifiers.append((sid, axis, "unavailable", None))
                continue
            value = pd.to_numeric(ds[column], errors="coerce")
            if sid == "S6":
                stratifiers.extend([(sid, axis, "none", value.eq(0)), (sid, axis, "any", value.gt(0))])
            elif sid == "S7":
                stratifiers.extend([(sid, axis, "lte_2", value.le(2)), (sid, axis, "gt_2", value.gt(2))])
            else:
                stratifiers.extend([(sid, axis, "absent", value.eq(0)), (sid, axis, "present", value.eq(1))])
        for sid, axis, level, mask in stratifiers:
            if mask is None:
                unavailable(dataset, sid, axis, "stratifier unavailable")
            else:
                evaluate(dataset, sid, axis, level, ds.loc[mask], "exposure_any")
        if "fio2_max" in ds:
            fio2 = pd.to_numeric(ds["fio2_max"], errors="coerce")
            for level, mask in (
                ("fio2_21_29", fio2.between(21, 29.999)),
                ("fio2_30_49", fio2.between(30, 49.999)),
                ("fio2_ge_50", fio2.ge(50)),
            ):
                evaluate(dataset, "S5", "fio2_stratum", level, ds.loc[mask], "exposure_any")
        # S9: plausibility/artifact definitions.
        for level, exposure_col in (
            ("current_50_to_100", "exposure_any"),
            ("strict_gt70_to_100", "exposure_strict_gt70"),
            ("raw_le_100", "exposure_raw"),
        ):
            evaluate(dataset, "S9", "artifact_filter", level, ds, exposure_col)
        # S10: alternate instability signatures.
        for level, exposure_col in (
            ("jump_3pp_dynamics_only", "exposure_jump3"),
            ("jump_5pp_dynamics_only", "exposure_jump5"),
            ("hypoxemia_or_jump4", "exposure_hypoxemia_or_dynamics"),
            ("sustained_abrupt_jumps", "exposure_sustained_jump"),
            ("sustained_desaturation", "exposure_sustained_low"),
            ("rmssd_above_dataset_median", "exposure_rmssd_high"),
        ):
            evaluate(dataset, "S10", "instability_definition", level, ds, exposure_col)
        # S11: first eligible ICU stay per patient.
        if "first_stay_per_person_flag" in ds:
            first = pd.to_numeric(ds["first_stay_per_person_flag"], errors="coerce").eq(1)
            evaluate(dataset, "S11", "repeated_stays", "first_stay_only", ds.loc[first], "exposure_any")
        else:
            unavailable(dataset, "S11", "repeated_stays", "first-stay flag unavailable")
        # S12: sampling/missingness-only exposure versus physiology exposure.
        evaluate(dataset, "S12", "measurement_intensity_control", "physiological_instability", ds, "exposure_any")
        evaluate(dataset, "S12", "measurement_intensity_control", "sampling_density_high", ds, "exposure_sampling_high")

    result = pd.DataFrame(rows)
    if "p_value" in result:
        valid = pd.to_numeric(result["p_value"], errors="coerce").notna()
        result.loc[valid, "p_value_bh"] = _bh_adjust(
            pd.to_numeric(result.loc[valid, "p_value"], errors="coerce").tolist()
        )
        result.loc[valid, "multiplicity_scope"] = "entire_prespecified_sensitivity_matrix"
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

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for (dataset, outcome), group in records.groupby(["dataset", "outcome"]):
        preceded = pd.to_numeric(
            group.get("preceded_by_instability_flag"), errors="coerce"
        ).fillna(0).eq(1)
        paired_group = group.loc[preceded].copy()
        values = pd.to_numeric(
            paired_group["lead_time_minutes"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        paired_group = paired_group.loc[
            pd.to_numeric(paired_group["lead_time_minutes"], errors="coerce").notna()
        ]
        n_outcomes = len(group)
        n = len(values)
        positive = int(preceded.sum())
        # Exact two-sided sign test vs p=0.5 using the binomial PMF.
        p_value = _binomial_twosided(positive, n_outcomes, 0.5)
        # Cluster bootstrap on median lead time (person clusters when mapped).
        if analysis_df is not None and "person_id" in analysis_df:
            dataset_key = analysis_df.get(
                "dataset", pd.Series("unknown", index=analysis_df.index)
            ).fillna("unknown").astype(str)
            stay_key = analysis_df["stay_id"].astype(str)
            key = dataset_key + ":" + stay_key
            person = analysis_df["person_id"].astype("string")
            person = person.where(
                person.notna() & person.str.strip().ne(""),
                dataset_key.astype("string") + ":stay:" + stay_key.astype("string"),
            )
            person_map = dict(zip(key, person.astype(str)))
            clusters = np.array([person_map.get(f"{r.dataset}:{int(r.stay_id)}", f"{r.dataset}:{int(r.stay_id)}") for r in paired_group.itertuples()])
        else:
            clusters = np.arange(n).astype(str)
        meds: list[float] = []
        unique = np.unique(clusters)
        by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
        if n and len(unique) and bootstrap_repetitions > 0:
            for _ in range(bootstrap_repetitions):
                sampled = rng.choice(unique, size=len(unique), replace=True)
                idx = np.concatenate([by_cluster[c] for c in sampled])
                meds.append(float(np.median(values[idx])))
        enough_bootstrap = (
            bootstrap_repetitions > 0
            and len(meds) >= max(100, math.ceil(bootstrap_repetitions * 0.8))
        )
        status = "estimated" if n > 0 and enough_bootstrap else "unavailable"
        rows.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "status": status,
                "n_outcome_events": n_outcomes,
                "n_paired": n,
                "n_preceded": positive,
                "n_outcome_without_prior_instability": n_outcomes - positive,
                "preceded_fraction": positive / n_outcomes if n_outcomes else math.nan,
                "sign_test_p": p_value,
                "sign_test_interpretation": "descriptive_only_no_justified_50_percent_null",
                "median_lead_time_minutes": float(np.median(values)) if n else math.nan,
                "q1_lead_time_minutes": float(np.quantile(values, 0.25)) if n else math.nan,
                "q3_lead_time_minutes": float(np.quantile(values, 0.75)) if n else math.nan,
                "median_boot_ci95_low": float(np.quantile(meds, 0.025)) if meds else math.nan,
                "median_boot_ci95_high": float(np.quantile(meds, 0.975)) if meds else math.nan,
                "bootstrap_repetitions_requested": int(bootstrap_repetitions),
                "bootstrap_repetitions_valid": int(len(meds)),
                "negative_lead_time_count": int((values <= 0).sum()),
                "claim_scope": "landmark_ordering_design_enforced_not_causal_precedence",
                "reason": (
                    ""
                    if status == "estimated"
                    else "no paired prior-instability onset or fewer than 80% valid cluster bootstraps"
                ),
            }
        )
    return pd.DataFrame(rows)


def _binomial_twosided(k: int, n: int, p: float) -> float:
    """Exact two-sided binomial test for small n; normal approximation for large n.

    The exact path evaluates the PMF in log-space (math.lgamma), so comb(n, x)
    never materializes as an integer; beyond n=10,000 the exact enumeration is
    both unnecessary and numerically meaningless, and the normal approximation
    with continuity correction is used instead.
    """
    from math import comb, exp, lgamma, log, sqrt

    if n <= 0:
        return 1.0
    if n > 10_000:
        mean = n * p
        std = sqrt(n * p * (1 - p))
        if std == 0:
            return 0.0 if k != mean else 1.0
        z = (abs(k - mean) - 0.5) / std
        # Two-sided normal tail: erfc(z/sqrt(2)).
        return float(math.erfc(z / sqrt(2.0)))

    log_pmf = lambda x: (
        lgamma(n + 1) - lgamma(x + 1) - lgamma(n - x + 1)
        + x * log(p) + (n - x) * log(1 - p)
    )
    observed = log_pmf(k)
    threshold = observed + log(1 + 1e-9)  # pmf <= pmf(k)*(1+1e-9)  <=>  log_pmf <= log_pmf(k) + log(1+eps)
    total = sum(exp(log_pmf(x)) for x in range(n + 1) if log_pmf(x) <= threshold)
    return float(min(1.0, total))


def build_specificity_matrix(
    risk_tables: pd.DataFrame,
    dose_response: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Analysis E: per-endpoint classification across the evidence family.

    Prespecified specificity comparator: hepatic_lab_worsening. It is not a
    true negative control because hypoxemia/shock can plausibly affect hepatic
    injury. Acute associations are more endpoint-specific when robust while the
    comparator is null, but this pattern is not causal evidence.
    """
    rows: list[dict[str, Any]] = []
    primary = set(PRIMARY_ENDPOINTS)
    for (dataset, endpoint), group in risk_tables.groupby(["dataset", "endpoint"]):
        row = group.iloc[-1]
        p = row.get("fisher_p_bh_adjusted")
        rr = row.get("risk_ratio")
        rr_low = row.get("risk_ratio_ci95_low")
        rr_method = row.get("risk_ratio_ci_method")
        status = row.get("status")
        p_numeric = pd.to_numeric(pd.Series([p]), errors="coerce").iloc[0]
        rr_numeric = pd.to_numeric(pd.Series([rr]), errors="coerce").iloc[0]
        rr_low_numeric = pd.to_numeric(
            pd.Series([rr_low]), errors="coerce"
        ).iloc[0]
        if status != "estimated" or not math.isfinite(p_numeric):
            classification = "underpowered" if status == "underpowered" else "unavailable"
        elif (
            math.isfinite(rr_numeric)
            and rr_numeric > 1.0
            and math.isfinite(rr_low_numeric)
            and rr_low_numeric > 1.0
            and rr_method == "patient_cluster_bootstrap"
            and p_numeric < 0.05
        ):
            classification = "positive_robust" if endpoint in primary else "positive_suggestive"
        elif (
            math.isfinite(rr_numeric)
            and rr_numeric > 1.0
            and p_numeric < 0.20
        ):
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
        role = "specificity_comparator" if endpoint == SPECIFICITY_COMPARATOR_ENDPOINT else ("primary" if endpoint in primary else "secondary")
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
                        if role == "specificity_comparator" and classification.startswith("positive")
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
    sensitivity = build_sensitivity_matrix(frame, events_df)
    return {
        "exposure_frame": frame,
        "stratified_risk_tables": risk,
        "mantel_haenszel": mh,
        "dose_response": dose,
        "paired_precedence": paired,
        "specificity_matrix": specificity,
        "sensitivity_matrix": sensitivity,
    }


__all__ = [
    "assign_exposure",
    "build_dose_response_tables",
    "build_mantel_haenszel_tables",
    "build_paired_precedence_table",
    "build_specificity_matrix",
    "build_sensitivity_matrix",
    "build_stratified_risk_tables",
    "run_epidemiology_analyses",
]
