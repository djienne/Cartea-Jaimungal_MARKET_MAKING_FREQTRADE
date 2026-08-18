"""Calibration knobs added for the CASHCAT one-sided-edge investigation.

Context (docs/market_viability_report.json): on CASHCAT the buy-MO curve turns
profitable past ~14 bps of depth and reaches +10.6 bps at 40 bps, while the
sell-MO curve is negative at every depth out to 34 bps. The model quotes the bid
at 1/kappa- + eps- = 10.9 bps, i.e. inside the losing zone, so two calibration
choices are on trial: epsilon's 200 ms measurement horizon and kappa's
whole-distribution survival fit. Both are now sweepable PER SIDE.

The point of this file is that the sweep must be able to move those knobs
without moving anything the live estimator container writes. So every test here
is one of two kinds:

1. "the default did not move" -- the live estimator passes none of the new
   flags, and the numbers it produces must be identical to before they existed;
2. "the new knob does what it says, and only where it is pointed".
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import estimate_all  # noqa: E402
import get_epsilon  # noqa: E402
from estimator_common import (  # noqa: E402
    fit_kappa_survival,
    load_market_window,
    normalize_trades,
    parse_window_bound_ms,
    select_shards_for_bounds,
)
from get_epsilon import (  # noqa: E402
    resolve_post_horizon_ms,
    resolve_post_horizon_ms_pair,
    run_epsilon_for_crypto,
)
from get_kappa import run_kappa_for_crypto  # noqa: E402


# ---------------------------------------------------------------------------
# (a) the default kappa fit is bit-identical to the pre-support-bound fit
# ---------------------------------------------------------------------------


def _kappa_fit_as_of_before_support_lower(depths: np.ndarray, support_quantile: float = 0.99) -> dict:
    """Frozen copy of fit_kappa_survival as it stood before the lower bound.

    Kept verbatim (not imported, not parameterised) so the equality test below
    compares against the actual shipped algorithm rather than against the new
    one calling itself with different arguments.
    """
    empty = {
        "kappa": float("nan"),
        "r_squared": float("nan"),
        "n_points": 0,
        "depth_p95": float("nan"),
        "depth_max_fitted": float("nan"),
        "survival_intercept": float("nan"),
    }
    depths = np.asarray(depths, dtype=float)
    depths = depths[np.isfinite(depths)]
    n_total = len(depths)
    if n_total < 10:
        return empty

    depth_p95 = float(np.percentile(depths, 95))
    support_cap = float(np.quantile(depths, support_quantile))
    sorted_depths = np.sort(depths)

    grid = np.unique(sorted_depths[sorted_depths <= support_cap])
    if len(grid) < 3:
        return empty

    tail_counts = n_total - np.searchsorted(sorted_depths, grid, side="left")
    survival = tail_counts.astype(float) / float(n_total)
    mask = (survival > 0) & (tail_counts >= 2)
    grid = grid[mask]
    survival = survival[mask]
    tail_counts = tail_counts[mask]
    if len(grid) < 3:
        return empty

    y = np.log(survival)
    weights = np.sqrt(tail_counts.astype(float))
    design = np.column_stack((np.ones_like(grid), -grid))
    try:
        coef, _, _, _ = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)
    except np.linalg.LinAlgError:
        return empty
    intercept, kappa = float(coef[0]), float(coef[1])

    y_pred = intercept - kappa * grid
    ss_res = float(np.sum((weights * (y - y_pred)) ** 2))
    y_mean = float(np.average(y, weights=weights**2))
    ss_tot = float(np.sum((weights * (y - y_mean)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "kappa": max(kappa, 0.0),
        "r_squared": r_squared,
        "n_points": int(len(grid)),
        "depth_p95": depth_p95,
        "depth_max_fitted": float(grid.max()),
        "survival_intercept": float(np.exp(intercept)),
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_default_kappa_fit_is_bit_identical_to_the_old_implementation(seed):
    """The live estimator calls fit_kappa_survival(depths) with no support args.

    Every one of those calls must return exactly the float it returned before
    the lower bound existed -- not "close", the same bits: kappa is EMA-blended
    across cycles, so even a last-place difference would propagate.
    """
    rng = np.random.default_rng(seed)
    # A mixture, so the fit is not trivially well conditioned: most MOs are
    # shallow, a few walk deep. That is the shape the CASHCAT depths have.
    depths = np.concatenate(
        [
            rng.exponential(1.0 / 8000.0, 900),
            rng.exponential(1.0 / 1500.0, 100),
        ]
    )
    old = _kappa_fit_as_of_before_support_lower(depths)
    new = fit_kappa_survival(depths)

    for key, expected in old.items():
        assert key in new, f"{key} disappeared from the fit dict"
        if isinstance(expected, float) and np.isnan(expected):
            assert np.isnan(new[key])
        else:
            assert new[key] == expected, f"{key} moved: {expected!r} -> {new[key]!r}"


def test_default_kappa_fit_reports_the_default_support_range():
    rng = np.random.default_rng(11)
    depths = rng.exponential(0.1, 2000)
    fit = fit_kappa_survival(depths)
    assert fit["support_quantile_lower"] == 0.0
    assert fit["support_quantile_upper"] == 0.99
    assert np.isnan(fit["support_depth_lower"])  # no lower bound was applied
    assert fit["support_depth_upper"] == pytest.approx(float(np.quantile(depths, 0.99)))


def test_kappa_lower_support_bound_refits_over_the_deep_tail_only():
    """Two depth regimes, as the suspicion about CASHCAT describes: the bulk of
    market orders barely leave the touch, and a thin minority walks deep enough
    that a quote out there would have earned. Fitting everything hands back the
    shallow slope; raising the lower support bound recovers the deep one."""
    rng = np.random.default_rng(3)
    shallow = rng.exponential(1.0 / 20000.0, 9500)  # kappa ~= 20000, 95% of MOs
    deep = rng.exponential(1.0 / 300.0, 500) + 0.0005  # kappa ~= 300, out in the tail
    depths = np.concatenate([shallow, deep])

    full = fit_kappa_survival(depths)
    tail = fit_kappa_survival(depths, support_quantile_lower=0.90)

    assert tail["kappa"] < full["kappa"], "a deeper support must give a looser kappa"
    assert tail["support_quantile_lower"] == 0.90
    assert tail["support_depth_lower"] == pytest.approx(float(np.quantile(depths, 0.90)))
    assert tail["depth_min_fitted"] >= tail["support_depth_lower"]
    # 1/kappa is what the quote is made of: the whole point is that this moves,
    # and by a lot -- here the model half-spread quadruples.
    assert (1.0 / tail["kappa"]) > 3.0 * (1.0 / full["kappa"])
    # ...and it lands near the deep regime that generated those fills.
    assert abs(tail["kappa"] - 300.0) < abs(full["kappa"] - 300.0)


def test_kappa_support_bounds_are_validated():
    depths = np.random.default_rng(1).exponential(0.1, 500)
    with pytest.raises(ValueError):
        fit_kappa_survival(depths, support_quantile=0.5, support_quantile_lower=0.5)
    with pytest.raises(ValueError):
        fit_kappa_survival(depths, support_quantile_lower=-0.1)


# ---------------------------------------------------------------------------
# horizon resolution
# ---------------------------------------------------------------------------


def test_post_horizon_defaults_to_200ms_on_both_sides(monkeypatch):
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_PLUS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_MINUS", raising=False)
    assert resolve_post_horizon_ms_pair() == (200, 200)
    assert resolve_post_horizon_ms() == 200


def test_post_horizon_scalar_is_the_shorthand_for_both_sides(monkeypatch):
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_PLUS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_MINUS", raising=False)
    assert resolve_post_horizon_ms_pair(cli_scalar=1500) == (1500, 1500)
    monkeypatch.setenv("EPSILON_POST_HORIZON_MS", "750")
    assert resolve_post_horizon_ms_pair() == (750, 750)


def test_post_horizon_per_side_overrides_scalar_and_env(monkeypatch):
    monkeypatch.setenv("EPSILON_POST_HORIZON_MS", "300")
    monkeypatch.setenv("EPSILON_POST_HORIZON_MS_MINUS", "4000")
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_PLUS", raising=False)
    # plus falls through to the scalar env, minus takes its own env
    assert resolve_post_horizon_ms_pair() == (300, 4000)
    # a per-side CLI flag beats every env var
    assert resolve_post_horizon_ms_pair(cli_plus=200) == (200, 4000)
    # the scalar CLI beats env but loses to the per-side CLI
    assert resolve_post_horizon_ms_pair(cli_scalar=1000, cli_minus=5000) == (1000, 5000)


# ---------------------------------------------------------------------------
# synthetic market fixtures
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _write_market(
    data_dir: Path,
    *,
    symbol: str = "SYN",
    n_buy: int = 120,
    n_sell: int = 120,
    buy_impact: float = 0.0,
    sell_impact: float = 0.0,
    buy_decay_ms: int | None = None,
    sell_decay_ms: int | None = None,
    seed: int = 5,
    start_ms: int = 0,
    duration_ms: int = 600_000,
    duplicate_shards: bool = False,
) -> None:
    """A BBO stream on a 100 ms grid plus market orders that move the mid.

    ``*_impact`` is the size of the jump each MO puts into the mid; with a
    ``*_decay_ms`` the jump reverts after that long, which is how a 200 ms
    horizon and a 5 s horizon are made to disagree on purpose.
    """
    rng = np.random.default_rng(seed)
    grid = np.arange(start_ms, start_ms + duration_ms + 1, 100)

    buy_ts = ((np.linspace(start_ms + 5_000, start_ms + duration_ms - 20_000, n_buy) // 100) * 100 + 50).astype(int)
    sell_ts = ((np.linspace(start_ms + 7_000, start_ms + duration_ms - 22_000, n_sell) // 100) * 100 + 50).astype(int)

    def mid_at(ts: float) -> float:
        m = 100.0
        if buy_impact:
            if buy_decay_ms is None:
                m += buy_impact * int(np.sum(buy_ts < ts))
            else:
                m += buy_impact * int(np.sum((buy_ts < ts) & (ts <= buy_ts + buy_decay_ms)))
        if sell_impact:
            if sell_decay_ms is None:
                m -= sell_impact * int(np.sum(sell_ts < ts))
            else:
                m -= sell_impact * int(np.sum((sell_ts < ts) & (ts <= sell_ts + sell_decay_ms)))
        return m

    price_rows = []
    for ts in grid:
        m = mid_at(float(ts))
        price_rows.append({"timestamp": ts / 1000.0, "price": m - 0.05, "size": 1.0, "side": "bid", "exchange_timestamp": int(ts)})
        price_rows.append({"timestamp": ts / 1000.0, "price": m + 0.05, "size": 1.0, "side": "ask", "exchange_timestamp": int(ts)})

    trade_rows = []
    for ts, depth in zip(buy_ts, rng.exponential(0.05, len(buy_ts))):
        trade_rows.append({"timestamp": ts / 1000.0, "price": mid_at(float(ts)) + float(depth), "size": 0.01,
                           "side": "buy", "exchange_timestamp": int(ts), "trade_id": f"b{ts}"})
    for ts, depth in zip(sell_ts, rng.exponential(0.05, len(sell_ts))):
        trade_rows.append({"timestamp": ts / 1000.0, "price": mid_at(float(ts)) - float(depth), "size": 0.01,
                           "side": "sell", "exchange_timestamp": int(ts), "trade_id": f"s{ts}"})

    trades = pd.DataFrame(trade_rows).sort_values("timestamp")
    prices = pd.DataFrame(price_rows)

    # Shard names carry the flush timestamp and BOTH shard selectors read it --
    # the trailing one relative to the newest shard, the absolute one against the
    # requested bounds. Write one shard per minute named at the end of the data
    # it holds, the way the collector does, so a windowed load exercises the
    # selection rather than trivially finding one giant file.
    shard_ms = 60_000
    for stream, frame in (("prices", prices), ("trades", trades)):
        ts_ms = frame["exchange_timestamp"].to_numpy(dtype=float)
        for chunk_start in range(start_ms, start_ms + duration_ms + 1, shard_ms):
            chunk_end = chunk_start + shard_ms
            part = frame[(ts_ms >= chunk_start) & (ts_ms < chunk_end)]
            if part.empty:
                continue
            _write_parquet(data_dir / symbol / stream / f"{stream}_{chunk_end}.parquet", part)
            if duplicate_shards and stream == "trades":
                # Exactly the 2026-08-16 failure: a second collector on the same
                # output directory writes the same public feed under a different
                # shard name, so only the local receive `timestamp` differs
                # between the two copies.
                second = part.copy()
                second["timestamp"] = second["timestamp"] + 0.004
                _write_parquet(
                    data_dir / symbol / stream / f"{stream}_{chunk_end + 1}.parquet", second
                )


# ---------------------------------------------------------------------------
# (b) default epsilon output is unchanged when no new flag is passed
# ---------------------------------------------------------------------------


def test_default_epsilon_matches_the_single_horizon_path(tmp_path, monkeypatch):
    """No flags -> the 200 ms number, computed by the same one-pass code.

    The comparison value comes from compute_mo_impacts at a single horizon,
    which is exactly what run_epsilon_for_crypto did before the split.
    """
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_PLUS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_MINUS", raising=False)

    ds = tmp_path / "data"
    _write_market(ds, buy_impact=0.5, sell_impact=0.25)
    eps_file = tmp_path / "epsilon.json"
    run_epsilon_for_crypto("SYN", minutes=10, epsilon_file=str(eps_file), data_dir=ds)
    entry = json.loads(eps_file.read_text(encoding="utf-8"))["SYN"]

    window = load_market_window("SYN", 10, data_dir=ds)
    mos = get_epsilon.attach_pre_mid(get_epsilon.aggregate_market_orders(window.trades), window.mids)
    reference = get_epsilon.compute_mo_impacts(mos, window.mids, 200, window.window_end_ms)
    ref_plus, ref_minus, _ = get_epsilon._floored_trimmed_means(reference)

    assert entry["window_ms"] == 200
    assert entry["window_ms_plus"] == 200
    assert entry["window_ms_minus"] == 200
    assert entry["epsilon+_raw"] == pytest.approx(ref_plus, rel=0, abs=0)
    assert entry["epsilon-_raw"] == pytest.approx(ref_minus, rel=0, abs=0)
    assert entry["trades_analyzed"] == reference["trades_analyzed"]
    assert entry["trades_skipped"] == reference["trades_skipped"]


def test_explicit_200ms_flags_reproduce_the_default_entry(tmp_path, monkeypatch):
    """--post-horizon-ms 200 and --post-horizon-ms-plus/-minus 200 must all be
    the same run: the per-side split is a new degree of freedom, not a new
    default."""
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_PLUS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_MINUS", raising=False)
    ds = tmp_path / "data"
    _write_market(ds, buy_impact=0.5, sell_impact=0.25)

    volatile = {"generated_at", "ema_seeded"}

    def _run(name: str, **kwargs) -> dict:
        path = tmp_path / name
        run_epsilon_for_crypto("SYN", minutes=10, epsilon_file=str(path), data_dir=ds, **kwargs)
        entry = json.loads(path.read_text(encoding="utf-8"))["SYN"]
        return {k: v for k, v in entry.items() if k not in volatile}

    baseline = _run("a.json")
    scalar = _run("b.json", post_horizon_ms=200)
    per_side = _run("c.json", post_horizon_ms_plus=200, post_horizon_ms_minus=200)

    assert scalar == baseline
    assert per_side == baseline


def test_default_kappa_entry_records_the_default_support(tmp_path):
    ds = tmp_path / "data"
    _write_market(ds)
    kappa_file = tmp_path / "kappa.json"
    lambda_file = tmp_path / "lambda.json"
    run_kappa_for_crypto("SYN", minutes=10, kappa_file=str(kappa_file), lambda_file=str(lambda_file), data_dir=ds)
    entry = json.loads(kappa_file.read_text(encoding="utf-8"))["SYN"]

    assert entry["support_quantile_lower_plus"] == 0.0
    assert entry["support_quantile_lower_minus"] == 0.0
    assert entry["support_quantile_upper_plus"] == 0.99
    assert entry["support_quantile_upper_minus"] == 0.99
    # Every key the runner validates is still there.
    for key in ("kappa+", "kappa-", "lambda+", "lambda-", "depth_p95_plus", "depth_max_fitted_minus"):
        assert key in entry


def test_kappa_support_flags_are_applied_per_side(tmp_path):
    ds = tmp_path / "data"
    _write_market(ds, n_buy=400, n_sell=400)
    kappa_file = tmp_path / "kappa.json"
    lambda_file = tmp_path / "lambda.json"
    run_kappa_for_crypto(
        "SYN", minutes=10, kappa_file=str(kappa_file), lambda_file=str(lambda_file), data_dir=ds,
        support_quantile_lower_plus=0.5, ema_tau=0.0,
    )
    entry = json.loads(kappa_file.read_text(encoding="utf-8"))["SYN"]
    assert entry["support_quantile_lower_plus"] == 0.5
    assert entry["support_quantile_lower_minus"] == 0.0  # untouched side keeps the default
    assert entry["support_depth_lower_plus"] is not None
    assert entry["support_depth_lower_minus"] is None
    assert entry["depth_min_fitted_plus"] >= entry["support_depth_lower_plus"]


# ---------------------------------------------------------------------------
# (c) per-side horizons genuinely produce different eps+ and eps-
# ---------------------------------------------------------------------------


def test_per_side_horizons_produce_different_epsilons(tmp_path, monkeypatch):
    """Buy MOs move the mid permanently; sell MOs move it and it comes back
    after 400 ms. At a shared 200 ms horizon both sides see their jump. Measure
    the minus side at 5 s instead and it collapses, while the plus side is
    untouched -- which is the asymmetry the CASHCAT question turns on."""
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_PLUS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_MINUS", raising=False)

    ds = tmp_path / "data"
    _write_market(ds, buy_impact=0.5, sell_impact=0.5, sell_decay_ms=400)

    shared = tmp_path / "shared.json"
    run_epsilon_for_crypto("SYN", minutes=10, epsilon_file=str(shared), data_dir=ds)
    shared_entry = json.loads(shared.read_text(encoding="utf-8"))["SYN"]

    split = tmp_path / "split.json"
    run_epsilon_for_crypto(
        "SYN", minutes=10, epsilon_file=str(split), data_dir=ds,
        post_horizon_ms_plus=200, post_horizon_ms_minus=5000,
    )
    split_entry = json.loads(split.read_text(encoding="utf-8"))["SYN"]

    assert split_entry["window_ms_plus"] == 200
    assert split_entry["window_ms_minus"] == 5000
    assert split_entry["window_ms"] == 5000  # compat key reports the longest in use

    # The ask side is measured identically in both runs.
    assert split_entry["epsilon+_raw"] == pytest.approx(shared_entry["epsilon+_raw"], rel=1e-12)
    # The bid side is not: its jump has decayed by 5 s.
    assert shared_entry["epsilon-_raw"] > 0.3
    assert split_entry["epsilon-_raw"] < 0.1
    # And the per-side primary agrees with the matching diagnostic horizon.
    assert split_entry["epsilon-_raw"] == pytest.approx(split_entry["epsilon_5s_minus"], rel=1e-12)


def test_diagnostics_are_never_mislabelled_under_split_horizons(tmp_path, monkeypatch):
    """The 5s diagnostic must be a 5s measurement on BOTH sides even when one
    primary side is already at 5s -- otherwise the reuse shortcut would publish
    a 200 ms number under a 5s label."""
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_PLUS", raising=False)
    monkeypatch.delenv("EPSILON_POST_HORIZON_MS_MINUS", raising=False)
    ds = tmp_path / "data"
    _write_market(ds, buy_impact=0.5, buy_decay_ms=400, sell_impact=0.5)

    split = tmp_path / "split.json"
    run_epsilon_for_crypto(
        "SYN", minutes=10, epsilon_file=str(split), data_dir=ds,
        post_horizon_ms_plus=200, post_horizon_ms_minus=5000,
    )
    entry = json.loads(split.read_text(encoding="utf-8"))["SYN"]
    # plus decays: its 200 ms primary is large, its 5 s diagnostic is not.
    assert entry["epsilon+_raw"] > 0.3
    assert entry["epsilon_5s_plus"] < 0.1


# ---------------------------------------------------------------------------
# (d) trade_id de-duplication
# ---------------------------------------------------------------------------


def test_normalize_trades_drops_duplicate_trade_ids():
    trades = pd.DataFrame(
        {
            "timestamp": [1.000, 1.004, 2.000],
            "exchange_timestamp": [1000, 1000, 2000],
            "price": [100.0, 100.0, 101.0],
            "size": [1.0, 1.0, 2.0],
            "side": ["buy", "buy", "sell"],
            "trade_id": ["a", "a", "b"],
        }
    )
    out = normalize_trades(trades, "exchange")
    assert len(out) == 2


def test_normalize_trades_keeps_rows_with_no_usable_id():
    """The collector writes str(trade.get("tid")), so a feed message with no id
    lands as the literal "None". Those are distinct trades sharing a
    placeholder and must never collapse into one."""
    trades = pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 3.0],
            "exchange_timestamp": [1000, 2000, 3000],
            "price": [100.0, 100.0, 100.0],
            "size": [1.0, 1.0, 1.0],
            "side": ["buy", "buy", "buy"],
            "trade_id": ["None", "None", ""],
        }
    )
    assert len(normalize_trades(trades, "exchange")) == 3


def test_duplicated_shards_do_not_inflate_n_trades_or_lambda(tmp_path):
    """Regression for 2026-08-16: two collectors on one directory doubled every
    trade under different shard filenames, inflating n_trades and lambda+/- ~2x
    (2104 rows for 1055 unique trade ids) with nothing upstream noticing."""
    single = tmp_path / "single"
    doubled = tmp_path / "doubled"
    _write_market(single, n_buy=150, n_sell=150, seed=21)
    _write_market(doubled, n_buy=150, n_sell=150, seed=21, duplicate_shards=True)

    w_single = load_market_window("SYN", 10, data_dir=single)
    w_doubled = load_market_window("SYN", 10, data_dir=doubled)

    assert len(w_doubled.trades) == len(w_single.trades)
    assert w_doubled.meta["duplicate_trade_ids_dropped"] == 300
    assert w_single.meta["duplicate_trade_ids_dropped"] == 0

    def _lambdas(data_dir: Path, name: str) -> dict:
        kappa_file = tmp_path / f"kappa_{name}.json"
        run_kappa_for_crypto(
            "SYN", minutes=10, kappa_file=str(kappa_file),
            lambda_file=str(tmp_path / f"lambda_{name}.json"), data_dir=data_dir, ema_tau=0.0,
        )
        return json.loads(kappa_file.read_text(encoding="utf-8"))["SYN"]

    a = _lambdas(single, "single")
    b = _lambdas(doubled, "doubled")
    assert b["n_trades"] == a["n_trades"]
    assert b["lambda+"] == pytest.approx(a["lambda+"], rel=1e-12)
    assert b["lambda-"] == pytest.approx(a["lambda-"], rel=1e-12)


# ---------------------------------------------------------------------------
# emit mode: the live snapshots must not move
# ---------------------------------------------------------------------------


def _freeze(paths: list[Path]) -> dict[Path, tuple[bytes, float]]:
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}


def _assert_unchanged(before: dict[Path, tuple[bytes, float]]) -> None:
    for path, (payload, mtime) in before.items():
        assert path.read_bytes() == payload, f"{path} content changed"
        assert path.stat().st_mtime_ns == mtime, f"{path} mtime changed"


def _seed_live_snapshots(tmp_path: Path) -> list[Path]:
    live = {
        "kappa.json": {"SYN": {"kappa+": 1.0, "kappa-": 2.0, "status": "ok", "schema_version": 3}},
        "lambda.json": {"SYN": {"lambda+": 0.1, "lambda-": 0.2, "status": "ok", "schema_version": 3}},
        "epsilon.json": {"SYN": {"epsilon+": 0.01, "epsilon-": 0.02, "status": "ok", "schema_version": 3}},
    }
    paths = []
    for name, payload in live.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload, indent=4, sort_keys=True), encoding="utf-8")
        paths.append(path)
    return paths


def test_emit_mode_writes_only_the_emit_file_epsilon(tmp_path):
    ds = tmp_path / "data"
    _write_market(ds, buy_impact=0.3, sell_impact=0.3)
    live = _seed_live_snapshots(tmp_path)
    before = _freeze(live)

    out = tmp_path / "sweep_eps.json"
    run_epsilon_for_crypto(
        "SYN", minutes=10, epsilon_file=str(tmp_path / "epsilon.json"), data_dir=ds,
        post_horizon_ms_minus=5000, emit_params_json=out,
    )

    _assert_unchanged(before)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["crypto"] == "SYN"
    assert payload["calibration"]["epsilon_post_horizon_ms_minus"] == 5000
    assert payload["calibration"]["ema_applied"] is False
    entry = payload["epsilon"]
    # The emitted block is the entry that would have been written, key for key.
    assert entry["epsilon+"] == entry["epsilon+_raw"]  # no EMA against live state
    assert entry["window_ms_minus"] == 5000
    assert entry["schema_version"] == 3


def test_emit_mode_writes_only_the_emit_file_kappa(tmp_path):
    ds = tmp_path / "data"
    _write_market(ds, n_buy=300, n_sell=300)
    live = _seed_live_snapshots(tmp_path)
    before = _freeze(live)

    out = tmp_path / "sweep_kappa.json"
    run_kappa_for_crypto(
        "SYN", minutes=10, kappa_file=str(tmp_path / "kappa.json"),
        lambda_file=str(tmp_path / "lambda.json"), data_dir=ds,
        support_quantile_lower_minus=0.4, emit_params_json=out,
    )

    _assert_unchanged(before)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["calibration"]["kappa_support_quantile_lower_minus"] == 0.4
    assert payload["kappa"]["kappa+"] == payload["kappa"]["kappa+_raw"]
    assert payload["lambda"]["lambda_source"] == "mo_survival_fit"


def test_emit_mode_combined_cycle_writes_one_file(tmp_path):
    ds = tmp_path / "data"
    _write_market(ds, n_buy=300, n_sell=300, buy_impact=0.2, sell_impact=0.2)
    live = _seed_live_snapshots(tmp_path)
    (tmp_path / "lambda_trades.json").write_text("{}", encoding="utf-8")
    before = _freeze(live + [tmp_path / "lambda_trades.json"])

    out = tmp_path / "sweep_all.json"
    result = estimate_all.run_all(
        "SYN", minutes=10, data_dir=str(ds),
        post_horizon_ms_minus=5000, support_quantile_lower_plus=0.3,
        emit_params_json=out,
    )

    _assert_unchanged(before)
    assert result["emitted_to"] == str(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["calibration"]) >= {
        "kappa_support_quantile_lower_plus",
        "epsilon_post_horizon_ms_minus",
    }
    assert payload["kappa"] is not None and payload["lambda"] is not None
    assert payload["epsilon"] is not None
    assert payload["epsilon"]["window_ms_minus"] == 5000
    # Toxicity in the emitted set uses the kappa from this same slice, not the
    # stale 1.0/2.0 sitting in the seeded kappa.json next door.
    assert payload["epsilon"]["toxicity_plus"] == pytest.approx(
        payload["kappa"]["kappa+"] * payload["epsilon"]["epsilon+_raw"], rel=1e-9
    )


@pytest.mark.parametrize(
    "script,extra_flags,protected",
    [
        ("get_epsilon.py", ["--post-horizon-ms-minus", "5000"], ("epsilon.json",)),
        ("get_kappa.py", ["--support-quantile-lower-plus", "0.4"], ("kappa.json", "lambda.json")),
    ],
)
def test_emit_mode_from_the_cli_leaves_the_snapshots_untouched(tmp_path, script, extra_flags, protected):
    """The property that actually protects the running bot, exercised through
    the same entry point an operator would type.

    The scripts resolve their snapshot filenames relative to the CWD, so the
    subprocess runs in a directory holding a full set of them; if emit mode ever
    fell through to a save, this is where it would show.
    """
    ds = tmp_path / "data"
    _write_market(ds, buy_impact=0.3, sell_impact=0.3, n_buy=300, n_sell=300)
    cwd = tmp_path / "run"
    cwd.mkdir()
    live = _seed_live_snapshots(cwd)
    before = _freeze(live)

    out = tmp_path / f"cli_emit_{script}.json"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPTS / script),
            "--crypto", "SYN", "--minutes", "10",
            "--data-dir", str(ds),
            "--emit-params-json", str(out),
            *extra_flags,
        ],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Error" not in proc.stdout, proc.stdout
    _assert_unchanged(before)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["crypto"] == "SYN"
    for name in protected:
        # The emitted file carries the numbers; the live file still carries the
        # seeded placeholder values, untouched.
        assert json.loads((cwd / name).read_text(encoding="utf-8"))["SYN"]["status"] == "ok"


# ---------------------------------------------------------------------------
# explicit window bounds
# ---------------------------------------------------------------------------


def test_parse_window_bound_accepts_iso_and_epoch():
    expected = datetime(2026, 8, 17, 18, 7, 40, tzinfo=timezone.utc).timestamp() * 1000.0
    assert parse_window_bound_ms("2026-08-17T18:07:40Z") == expected
    # Naive means UTC: every timestamp this stack writes is UTC, including the
    # window_start/window_end an operator would copy out of a snapshot.
    assert parse_window_bound_ms("2026-08-17T18:07:40") == expected
    assert parse_window_bound_ms(expected / 1000.0) == expected  # epoch seconds
    assert parse_window_bound_ms(expected) == expected  # epoch ms
    assert parse_window_bound_ms(None) is None
    with pytest.raises(ValueError):
        parse_window_bound_ms("not a time")


def test_select_shards_for_bounds_keeps_a_historical_slice():
    files = [Path(f"trades_{1_000_000 + i * 60_000}.parquet") for i in range(100)]
    kept = select_shards_for_bounds(files, 2_000_000, 2_500_000, margin_ms=0.0)
    assert kept  # the trailing-window rule would have kept only the newest ones
    assert all(2_000_000 <= float(p.stem.rsplit("_", 1)[1]) <= 2_500_000 for p in kept)
    # Unparseable names are kept: fail open.
    assert Path("weird.parquet") in select_shards_for_bounds(
        files + [Path("weird.parquet")], 2_000_000, 2_500_000
    )


def test_window_bounds_select_a_train_slice(tmp_path):
    ds = tmp_path / "data"
    # 20 minutes of data starting at epoch 1_800_000_000_000.
    base = 1_800_000_000_000
    _write_market(ds, start_ms=base, duration_ms=1_200_000, n_buy=200, n_sell=200)

    full = load_market_window("SYN", 60, data_dir=ds)
    half = load_market_window(
        "SYN", 60, data_dir=ds,
        window_start=base + 0, window_end=base + 600_000,
    )
    assert half.window_start_ms == base
    assert half.window_end_ms == base + 600_000
    assert 0 < len(half.trades) < len(full.trades)
    assert half.window_seconds == pytest.approx(600.0)
    # An end beyond the data is clamped rather than inflating the covered span.
    clamped = load_market_window("SYN", 60, data_dir=ds, window_end=base + 10_000_000)
    assert clamped.window_end_ms <= base + 1_200_000


def test_window_bounds_reject_an_inverted_range(tmp_path):
    ds = tmp_path / "data"
    _write_market(ds)
    with pytest.raises(ValueError):
        load_market_window("SYN", 10, data_dir=ds, window_start=2_000, window_end=1_000)


def test_emit_on_a_train_slice_differs_from_the_full_window(tmp_path):
    """The sweep's actual use: fit on one slice, keep the live files alone."""
    ds = tmp_path / "data"
    base = 1_800_000_000_000
    _write_market(ds, start_ms=base, duration_ms=1_200_000, n_buy=300, n_sell=300, buy_impact=0.2)
    live = _seed_live_snapshots(tmp_path)
    before = _freeze(live)

    train = tmp_path / "train.json"
    test = tmp_path / "test.json"
    estimate_all.run_all("SYN", minutes=60, data_dir=str(ds), emit_params_json=train,
                         window_start=base, window_end=base + 600_000)
    estimate_all.run_all("SYN", minutes=60, data_dir=str(ds), emit_params_json=test,
                         window_start=base + 600_000, window_end=base + 1_200_000)

    _assert_unchanged(before)
    a = json.loads(train.read_text(encoding="utf-8"))
    b = json.loads(test.read_text(encoding="utf-8"))
    assert a["window"]["end"] != b["window"]["end"]
    assert a["kappa"]["n_market_orders_plus"] > 0
    assert b["kappa"]["n_market_orders_plus"] > 0
