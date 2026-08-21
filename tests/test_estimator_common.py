from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from estimator_common import (  # noqa: E402
    aggregate_market_orders,
    attach_pre_mid,
    build_bbo_mid,
    choose_time_source,
    fit_kappa_survival,
    load_market_window,
    mo_depths,
    realized_sigma2_per_sec,
)
def test_aggregate_market_orders_collapses_same_timestamp_prints():
    trades = pd.DataFrame(
        {
            "ts_ms": [1000, 1000, 1000, 2000, 3000],
            "side": ["buy", "buy", "buy", "buy", "sell"],
            "price": [100.0, 100.1, 100.3, 100.0, 99.8],
            "size": [1.0, 2.0, 0.5, 1.0, 4.0],
        }
    )
    mos = aggregate_market_orders(trades)

    assert len(mos) == 3
    sweep = mos[mos["ts_ms"] == 1000].iloc[0]
    assert sweep["price_extreme"] == 100.3  # deepest print of the buy sweep
    assert sweep["size"] == 3.5
    assert sweep["n_prints"] == 3
    sell = mos[mos["side"] == "sell"].iloc[0]
    assert sell["price_extreme"] == 99.8


def test_aggregate_market_orders_keeps_opposite_sides_separate():
    trades = pd.DataFrame(
        {
            "ts_ms": [1000, 1000],
            "side": ["buy", "sell"],
            "price": [100.1, 99.9],
            "size": [1.0, 1.0],
        }
    )
    mos = aggregate_market_orders(trades)
    assert len(mos) == 2


def test_attach_pre_mid_is_strictly_before():
    mos = pd.DataFrame({"ts_ms": [2000.0], "side": ["buy"], "price_extreme": [100.5], "size": [1.0], "n_prints": [1]})
    mids = pd.DataFrame({"ts_ms": [1000.0, 2000.0], "mid": [100.0, 101.0]})
    merged = attach_pre_mid(mos, mids)
    # The mid stamped at exactly the MO timestamp may already reflect the trade.
    assert merged["pre_mid"].iloc[0] == 100.0


def test_mo_depths_truncates_negatives_and_counts_skips():
    mos = pd.DataFrame(
        {
            "ts_ms": [1.0, 2.0, 3.0, 4.0],
            "side": ["buy", "buy", "sell", "sell"],
            "price_extreme": [100.5, 99.5, 99.5, 100.5],
            "size": [1.0, 1.0, 1.0, 1.0],
            "n_prints": [1, 1, 1, 1],
            "pre_mid": [100.0, 100.0, 100.0, np.nan],
        }
    )
    result = mo_depths(mos)
    assert result["skipped_no_pre_mid"] == 1
    np.testing.assert_allclose(result["buy_depths"], [0.5, 0.0])
    np.testing.assert_allclose(result["sell_depths"], [0.5])
    assert result["negative_depth_truncated_buy"] == 1
    assert result["negative_depth_truncated_sell"] == 0


def test_fit_kappa_survival_recovers_known_kappa():
    rng = np.random.default_rng(7)
    true_kappa = 5.0
    depths = rng.exponential(scale=1.0 / true_kappa, size=4000)
    fit = fit_kappa_survival(depths)

    assert abs(fit["kappa"] - true_kappa) / true_kappa < 0.10
    assert fit["r_squared"] > 0.95
    assert fit["n_points"] >= 100
    assert fit["depth_p95"] > 0
    assert fit["depth_max_fitted"] >= fit["depth_p95"] * 0.8


def test_fit_kappa_survival_insufficient_data_returns_nan():
    fit = fit_kappa_survival(np.array([0.1, 0.2, 0.3]))
    assert np.isnan(fit["kappa"])
    assert fit["n_points"] == 0


def test_realized_sigma2_recovers_step_variance():
    rng = np.random.default_rng(11)
    steps = rng.normal(0.0, 0.5, size=600)  # variance 0.25 per 1s step
    mids = pd.DataFrame(
        {
            "ts_ms": (np.arange(601) * 1000.0),
            "mid": 4000.0 + np.concatenate(([0.0], np.cumsum(steps))),
        }
    )
    sigma2 = realized_sigma2_per_sec(mids)
    assert sigma2 is not None
    assert abs(sigma2 - 0.25) / 0.25 < 0.30


def test_realized_sigma2_drops_increments_across_gaps():
    rng = np.random.default_rng(13)
    steps = rng.normal(0.0, 0.5, size=600)
    mid = 4000.0 + np.concatenate(([0.0], np.cumsum(steps)))
    ts = np.arange(601) * 1000.0
    # 60s outage in the middle with a 50-USDC level shift across it.
    keep = (ts < 300_000) | (ts > 360_000)
    mid_gapped = mid.copy()
    mid_gapped[ts > 360_000] += 50.0
    mids = pd.DataFrame({"ts_ms": ts[keep], "mid": mid_gapped[keep]})
    sigma2 = realized_sigma2_per_sec(mids)
    assert sigma2 is not None
    # The 50-USDC jump would dominate the variance if gap increments were kept.
    assert sigma2 < 1.0


def test_realized_sigma2_requires_min_samples():
    mids = pd.DataFrame({"ts_ms": np.arange(10) * 1000.0, "mid": np.full(10, 4000.0)})
    assert realized_sigma2_per_sec(mids) is None


def test_choose_time_source_requires_full_coverage():
    with_ts = pd.DataFrame({"timestamp": [1.0], "exchange_timestamp": [1000]})
    without_ts = pd.DataFrame({"timestamp": [1.0]})
    assert choose_time_source(with_ts, with_ts) == "exchange"
    assert choose_time_source(with_ts, without_ts) == "local"


def test_build_bbo_mid_pivots_one_row_per_side():
    prices = pd.DataFrame(
        {
            "timestamp": [1.0, 1.0, 2.0],
            "price": [99.0, 101.0, 99.5],
            "size": [1.0, 1.0, 1.0],
            "side": ["bid", "ask", "bid"],
            "exchange_timestamp": [1000, 1000, 2000],
        }
    )
    mids = build_bbo_mid(prices, "exchange")
    assert list(mids["ts_ms"]) == [1000, 2000]
    assert list(mids["mid"]) == [100.0, 100.25]  # ask ffilled at 2000


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def test_load_market_window_prefers_prices_stream(tmp_path):
    base = tmp_path / "TST"
    prices = pd.DataFrame(
        {
            "timestamp": [10.0, 10.0, 20.0, 20.0],
            "price": [99.0, 101.0, 99.4, 101.4],
            "size": [1.0] * 4,
            "side": ["bid", "ask", "bid", "ask"],
            "exchange_timestamp": [10_000, 10_000, 20_000, 20_000],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": [15.0],
            "price": [101.2],
            "size": [0.5],
            "side": ["buy"],
            "exchange_timestamp": [15_000],
            "trade_id": ["t1"],
        }
    )
    _write_parquet(base / "prices" / "p.parquet", prices)
    _write_parquet(base / "trades" / "t.parquet", trades)

    window = load_market_window("TST", minutes=60, data_dir=tmp_path)
    assert window.mid_source == "prices"
    assert window.ts_source == "exchange"
    assert len(window.mids) == 1  # clipped to overlap window ending at trade ts
    assert len(window.trades) == 1
    assert window.window_seconds == 3600.0


def test_load_market_window_falls_back_to_orderbooks(tmp_path):
    base = tmp_path / "TST"
    orderbooks = pd.DataFrame(
        {
            "timestamp": [10.0, 20.0],
            "exchange_timestamp": [10_000, 20_000],
            "bid_price_0": [99.0, 99.2],
            "ask_price_0": [101.0, 101.2],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": [15.0],
            "price": [101.0],
            "size": [0.5],
            "side": ["sell"],
            "exchange_timestamp": [15_000],
            "trade_id": ["t1"],
        }
    )
    _write_parquet(base / "orderbooks" / "o.parquet", orderbooks)
    _write_parquet(base / "trades" / "t.parquet", trades)

    window = load_market_window("TST", minutes=60, data_dir=tmp_path)
    assert window.mid_source == "orderbooks_fallback"
    assert not window.mids.empty


# ---------------------------------------------------------------------------
# End-to-end estimator pipeline on synthetic parquet datasets
# ---------------------------------------------------------------------------

from get_epsilon import run_epsilon_for_crypto  # noqa: E402
from get_kappa import run_kappa_for_crypto  # noqa: E402


def _write_synthetic_market(
    data_dir: Path,
    *,
    buy_depths: np.ndarray,
    sell_depths: np.ndarray,
    impact: float = 0.0,
    transient: bool = False,
    symbol: str = "SYN",
) -> None:
    """600s of synthetic BBO + trades. Buy MOs optionally move the mid by
    ``impact`` (permanently, or reverting after 300 ms when transient)."""
    duration_ms = 600_000
    base_mid = 100.0
    update_ts = np.arange(0, duration_ms + 1, 100)

    n_buy, n_sell = len(buy_depths), len(sell_depths)
    buy_ts = ((np.linspace(5_000, 580_000, n_buy) // 100) * 100 + 50).astype(int)
    sell_ts = ((np.linspace(7_000, 582_000, n_sell) // 100) * 100 + 50).astype(int)

    def mid_at(ts_ms: float) -> float:
        m = base_mid
        if impact:
            if transient:
                m += impact * int(np.sum((buy_ts < ts_ms) & (ts_ms <= buy_ts + 300)))
            else:
                m += impact * int(np.sum(buy_ts < ts_ms))
        return m

    price_rows = []
    for ts in update_ts:
        m = mid_at(float(ts))
        price_rows.append({"timestamp": ts / 1000.0, "price": m - 0.05, "size": 1.0, "side": "bid", "exchange_timestamp": int(ts)})
        price_rows.append({"timestamp": ts / 1000.0, "price": m + 0.05, "size": 1.0, "side": "ask", "exchange_timestamp": int(ts)})

    trade_rows = []
    for ts, depth in zip(buy_ts, buy_depths):
        pre = mid_at(float(ts))
        trade_rows.append({"timestamp": ts / 1000.0, "price": pre + float(depth), "size": 0.01, "side": "buy", "exchange_timestamp": int(ts), "trade_id": f"b{ts}"})
    for ts, depth in zip(sell_ts, sell_depths):
        pre = mid_at(float(ts))
        trade_rows.append({"timestamp": ts / 1000.0, "price": pre - float(depth), "size": 0.01, "side": "sell", "exchange_timestamp": int(ts), "trade_id": f"s{ts}"})

    _write_parquet(data_dir / symbol / "prices" / "p.parquet", pd.DataFrame(price_rows))
    _write_parquet(data_dir / symbol / "trades" / "t.parquet", pd.DataFrame(trade_rows).sort_values("timestamp"))


def test_kappa_pipeline_publishes_direct_window_estimates(tmp_path):
    rng = np.random.default_rng(5)
    kappa_file = tmp_path / "kappa.json"
    lambda_file = tmp_path / "lambda.json"

    ds1 = tmp_path / "run1"
    _write_synthetic_market(
        ds1,
        buy_depths=rng.exponential(0.1, 300),  # kappa ~= 10
        sell_depths=rng.exponential(0.1, 300),
    )
    run_kappa_for_crypto("SYN", minutes=10, kappa_file=str(kappa_file), lambda_file=str(lambda_file), data_dir=ds1)

    import json

    first = json.loads(kappa_file.read_text(encoding="utf-8"))["SYN"]
    assert first["status"] == "ok"
    assert first["lambda_source"] == "mo_survival_fit"
    assert first["kappa+"] == first["kappa+_raw"]
    assert 7.0 < first["kappa+"] < 13.0
    assert 0.4 < first["lambda+"] < 0.65  # ~300 MOs / ~582s
    assert first["depth_p95_plus"] > 0
    assert first["sigma2_per_sec"] is not None

    # Second run with kappa ~= 20: the primary must move directly to this
    # window's estimate rather than blending with the first run.
    ds2 = tmp_path / "run2"
    _write_synthetic_market(
        ds2,
        buy_depths=rng.exponential(0.05, 300),
        sell_depths=rng.exponential(0.05, 300),
    )
    run_kappa_for_crypto("SYN", minutes=10, kappa_file=str(kappa_file), lambda_file=str(lambda_file), data_dir=ds2)
    second = json.loads(kappa_file.read_text(encoding="utf-8"))["SYN"]
    assert second["kappa+_raw"] > 15.0
    assert second["kappa+"] == second["kappa+_raw"]
    assert abs(second["kappa+"] - first["kappa+"]) > 2.0


def test_epsilon_pipeline_distinguishes_permanent_from_transient_impact(tmp_path):
    rng = np.random.default_rng(9)
    depths = rng.exponential(0.05, 60)

    permanent_dir = tmp_path / "permanent"
    _write_synthetic_market(permanent_dir, buy_depths=depths, sell_depths=depths, impact=0.5, transient=False)
    eps_file = tmp_path / "epsilon_permanent.json"
    run_epsilon_for_crypto("SYN", minutes=10, epsilon_file=str(eps_file), data_dir=permanent_dir)

    import json

    entry = json.loads(eps_file.read_text(encoding="utf-8"))["SYN"]
    assert entry["status"] == "ok"
    assert entry["n_buy_events"] >= 50
    assert entry["window_ms"] == 200
    assert abs(entry["epsilon+"] - 0.5) < 0.05  # arrival jump measured at 200ms
    assert entry["epsilon-"] == 0.0
    # A genuinely permanent step is still there at the long diagnostic horizons.
    assert abs(entry["epsilon_5s_plus"] - 0.5) < 0.05

    transient_dir = tmp_path / "transient"
    _write_synthetic_market(transient_dir, buy_depths=depths, sell_depths=depths, impact=0.5, transient=True)
    eps_file2 = tmp_path / "epsilon_transient.json"
    run_epsilon_for_crypto("SYN", minutes=10, epsilon_file=str(eps_file2), data_dir=transient_dir)

    entry2 = json.loads(eps_file2.read_text(encoding="utf-8"))["SYN"]
    # KNOWN TRADE-OFF of the 200 ms primary horizon: it measures the jump at the
    # arrival instant, which is what eq. 10.22 defines, but it therefore cannot
    # tell a permanent step from a mechanical BBO reaction that decays. The long
    # diagnostics are what distinguish them, and they still do.
    assert entry2["epsilon+"] > 0.3          # arrival jump is visible...
    assert entry2["epsilon_1s_plus"] < 0.05  # ...but it did not persist
    assert entry2["epsilon_5s_plus"] < 0.05
    # Operators comparing epsilon+ against epsilon_5s_plus can see the decay; a
    # large gap means the shipped epsilon is pricing in transient reaction. The
    # principled fix is a drift-debiased long-horizon markout
    # (eps = markout - horizon * (lambda+ eps+ - lambda- eps-)), which needs
    # lambda inside get_epsilon and is deliberately not done here.


def test_epsilon_pipeline_flags_insufficient_events(tmp_path):
    rng = np.random.default_rng(15)
    depths = rng.exponential(0.05, 20)  # below MIN_EPSILON_EVENTS=50
    ds = tmp_path / "thin"
    _write_synthetic_market(ds, buy_depths=depths, sell_depths=depths)
    eps_file = tmp_path / "epsilon_thin.json"
    run_epsilon_for_crypto("SYN", minutes=10, epsilon_file=str(eps_file), data_dir=ds)

    import json

    entry = json.loads(eps_file.read_text(encoding="utf-8"))["SYN"]
    assert entry["status"] == "insufficient_data"


def test_load_market_window_uses_local_clock_when_exchange_ts_missing(tmp_path):
    base = tmp_path / "TST"
    prices = pd.DataFrame(
        {
            "timestamp": [10.0, 10.0],
            "price": [99.0, 101.0],
            "size": [1.0, 1.0],
            "side": ["bid", "ask"],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": [10.5],
            "price": [101.0],
            "size": [0.5],
            "side": ["buy"],
        }
    )
    _write_parquet(base / "prices" / "p.parquet", prices)
    _write_parquet(base / "trades" / "t.parquet", trades)

    window = load_market_window("TST", minutes=60, data_dir=tmp_path)
    assert window.ts_source == "local"
    assert window.mids["ts_ms"].iloc[0] == 10_000.0
