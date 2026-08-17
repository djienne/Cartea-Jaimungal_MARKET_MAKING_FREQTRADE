"""Behavioural pins for scripts/mm_core.py.

mm_core is about to become the single implementation of every piece of quote
arithmetic in the project (today the strategy and the replay each carry their
own copy). These tests pin what it does *before* anything imports it, so the
migration has a fixed reference to be checked against rather than a moving one.

The properties that matter most, and why:
  - a disabled side is None, never a floor-clamped quote (the HJB's way of
    saying "do not quote here" must survive assembly),
  - the fee cushion is one maker fee per side and the multiplier scales only
    the model term,
  - bids round down and asks round up, never toward the touch.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mm_core import (  # noqa: E402
    HalfSpread,
    JsonlEventLogger,
    QuoteConfig,
    assemble_half_spread,
    compute_quotes,
    effective_phi,
    finite_float_or_none,
    inventory_to_q,
    maker_safe,
    merged_params,
    parse_utc_timestamp,
    select_delta,
    solve_hjb,
    resolve_net_inventory,
    route_sides,
    validate_param_snapshot,
)

MID = 1650.0

LIVE_PARAMS = {
    "kappa+": 2.0,
    "kappa-": 2.0,
    "epsilon+": 0.05,
    "epsilon-": 0.05,
    "lambda+": 0.1,
    "lambda-": 0.1,
}


def _snapshot(now: datetime | None = None, **overrides):
    """A schema-v3 parameter snapshot that passes every validation gate."""
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    kappa = {
        "schema_version": 3,
        "status": "ok",
        "generated_at": stamp,
        "kappa+": 2.0,
        "kappa-": 2.0,
        "n_points_plus": 40,
        "n_points_minus": 40,
        "r2_plus": 0.9,
        "r2_minus": 0.9,
    }
    epsilon = {
        "schema_version": 3,
        "status": "ok",
        "generated_at": stamp,
        "epsilon+": 0.05,
        "epsilon-": 0.05,
        "n_buy_events": 500,
        "n_sell_events": 500,
    }
    lambda_ = {
        "schema_version": 3,
        "status": "ok",
        "generated_at": stamp,
        "lambda+": 0.1,
        "lambda-": 0.1,
        "lambda_source": "mo_survival_fit",
    }
    for target, patch in overrides.items():
        {"kappa": kappa, "epsilon": epsilon, "lambda_": lambda_}[target].update(patch)
    return kappa, epsilon, lambda_


# --------------------------------------------------------------------------
# assemble_half_spread
# --------------------------------------------------------------------------


def test_disabled_side_is_none_not_a_floor_clamped_quote():
    """An infinite model depth must not become a quote at the floor.

    This is the single most important property in the module: the HJB returns
    +inf at the inventory boundary meaning "do not quote", and clamping that
    into [min, max] bps would silently turn it into an aggressive quote.
    """
    config = QuoteConfig()
    assert assemble_half_spread(float("inf"), MID, config) is None
    assert assemble_half_spread(float("nan"), MID, config) is None
    assert assemble_half_spread(None, MID, config) is None


def test_invalid_mid_yields_no_quote():
    config = QuoteConfig()
    for bad_mid in (0.0, -1.0, float("nan"), float("inf")):
        assert assemble_half_spread(0.5, bad_mid, config) is None


def test_fee_cushion_is_one_maker_fee_per_side():
    config = QuoteConfig(maker_fee_rate=0.00015, spread_multiplier=1.0)
    spread = assemble_half_spread(0.5, MID, config)
    assert spread is not None
    assert spread.fee_cushion == pytest.approx(0.00015 * MID)
    assert spread.delta_pre_clamp == pytest.approx(0.5 + 0.00015 * MID)


def test_multiplier_scales_only_the_model_term():
    """Widening defensively must not inflate the fee compensation."""
    base = QuoteConfig(spread_multiplier=1.0, maker_fee_rate=0.00015)
    tripled = QuoteConfig(spread_multiplier=3.0, maker_fee_rate=0.00015)
    a = assemble_half_spread(0.5, MID, base)
    b = assemble_half_spread(0.5, MID, tripled)
    assert a is not None and b is not None
    assert b.delta_pre_clamp - a.delta_pre_clamp == pytest.approx(2 * 0.5)
    assert a.fee_cushion == b.fee_cushion


@pytest.mark.parametrize(
    "delta_model,expected_clamp",
    [
        (1e-9, "floor"),   # far below the 3 bps floor
        (100.0, "cap"),    # far above the 80 bps cap
        (0.5, None),       # ~3 bps + fee, inside the band
    ],
)
def test_clamping_is_reported(delta_model, expected_clamp):
    # Explicit floor rather than the shipped one: the default floor is anchored
    # to the maker fee, so a zero-depth quote lands exactly ON it and the clamp
    # never fires. This test is about the clamp MECHANISM, not the constant.
    config = QuoteConfig(min_half_spread_bps=3.0)
    spread = assemble_half_spread(delta_model, MID, config)
    assert spread is not None
    assert spread.clamped == expected_clamp
    # `bps` is recomputed as delta/mid after delta was derived from bps*mid, so
    # the round trip can land an ulp outside the band (3.0 bps comes back as
    # 2.9999999999999996). Any gate asserting the bound exactly would flake.
    tol = 1e-9
    assert config.min_half_spread_bps - tol <= spread.bps <= config.max_half_spread_bps + tol


def test_depth_p95_flags_extrapolation_beyond_calibrated_range():
    config = QuoteConfig()
    near = assemble_half_spread(0.5, MID, config, depth_p95=10.0)
    far = assemble_half_spread(0.5, MID, config, depth_p95=0.05)
    assert near is not None and far is not None
    assert near.outside_calibrated_range is False
    assert far.outside_calibrated_range is True
    # No p95 supplied -> unknown, not False.
    assert assemble_half_spread(0.5, MID, config).outside_calibrated_range is None


def test_half_spread_as_dict_keys_are_stable():
    """Gate evaluators read these keys by name."""
    spread = assemble_half_spread(0.5, MID, QuoteConfig())
    assert spread is not None
    assert set(spread.as_dict()) == {
        "delta_total",
        "delta_model",
        "delta_pre_clamp",
        "fee_cushion",
        "bps",
        "clamped",
        "depth_p95",
        "quote_outside_calibrated_range",
    }


# --------------------------------------------------------------------------
# QuoteConfig validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_half_spread_bps": 80.0, "max_half_spread_bps": 3.0},
        {"min_half_spread_bps": 5.0, "max_half_spread_bps": 5.0},
        {"spread_multiplier": 0.0},
        {"spread_multiplier": -1.0},
        {"spread_multiplier": float("inf")},
        {"q_max": 0},
        {"q_max": -1},
    ],
)
def test_quote_config_rejects_incoherent_settings(kwargs):
    with pytest.raises(ValueError):
        QuoteConfig(**kwargs)


# --------------------------------------------------------------------------
# inventory_to_q
# --------------------------------------------------------------------------


def test_inventory_to_q_is_signed_when_shorting_allowed():
    config = QuoteConfig(inventory_unit_base=0.01, q_max=3, allow_short=True)
    assert inventory_to_q(0.0, config) == 0
    assert inventory_to_q(0.02, config) == 2
    assert inventory_to_q(-0.02, config) == -2
    # Clamped at both ends, not wrapped.
    assert inventory_to_q(99.0, config) == 3
    assert inventory_to_q(-99.0, config) == -3


def test_inventory_to_q_clamps_to_zero_when_long_only():
    """allow_short=False reproduces the legacy strategy's [0, q_max] clamp."""
    config = QuoteConfig(inventory_unit_base=0.01, q_max=3, allow_short=False)
    assert inventory_to_q(-0.05, config) == 0
    assert inventory_to_q(0.02, config) == 2
    assert inventory_to_q(99.0, config) == 3


def test_inventory_to_q_rounds_to_nearest_unit():
    config = QuoteConfig(inventory_unit_base=0.01, q_max=6, allow_short=True)
    assert inventory_to_q(0.014, config) == 1
    assert inventory_to_q(0.016, config) == 2


def test_inventory_to_q_treats_garbage_as_flat():
    config = QuoteConfig()
    for bad in (None, float("nan"), float("inf"), "abc"):
        assert inventory_to_q(bad, config) == 0


# --------------------------------------------------------------------------
# maker_safe
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "side,price,ok,reason",
    [
        ("bid", 99.0, True, "ok"),
        ("bid", 101.0, False, "bid_crosses_ask"),
        ("bid", 100.5, False, "bid_crosses_ask"),   # equal to ask is a cross
        ("ask", 101.0, True, "ok"),
        ("ask", 99.0, False, "ask_crosses_bid"),
        ("ask", 99.5, False, "ask_crosses_bid"),    # equal to bid is a cross
    ],
)
def test_maker_safe_rejects_anything_that_would_take(side, price, ok, reason):
    assert maker_safe(side, price, 99.5, 100.5) == (ok, reason)


def test_maker_safe_rejects_invalid_or_crossed_book():
    assert maker_safe("bid", 100.0, 101.0, 100.0)[1] == "crossed_or_invalid_book"
    assert maker_safe("bid", -1.0, 99.5, 100.5)[1] == "invalid_rate"
    assert maker_safe("bid", float("nan"), 99.5, 100.5)[1] == "invalid_rate"


# --------------------------------------------------------------------------
# solve_hjb / select_delta / compute_quotes
# --------------------------------------------------------------------------


def test_compute_quotes_returns_both_sides_at_interior_inventory():
    """The whole point of the rework: one call, both sides, same mid and q."""
    config = QuoteConfig(spread_multiplier=1.0)
    hjb = solve_hjb(LIVE_PARAMS, config)
    pair = compute_quotes(MID, 0, hjb, config)

    assert pair.bid_price is not None
    assert pair.ask_price is not None
    assert pair.bid is not None and pair.ask is not None
    assert pair.bid_price < MID < pair.ask_price


def test_compute_quotes_disables_the_bid_at_max_long():
    """At +q_max the model says stop buying; the ask must stay live."""
    config = QuoteConfig(spread_multiplier=1.0)
    hjb = solve_hjb(LIVE_PARAMS, config)
    pair = compute_quotes(MID, config.q_max, hjb, config)

    assert pair.bid_price is None
    assert pair.bid is None
    assert pair.ask_price is not None


def test_compute_quotes_disables_the_ask_at_max_short():
    config = QuoteConfig(spread_multiplier=1.0)
    hjb = solve_hjb(LIVE_PARAMS, config)
    pair = compute_quotes(MID, -config.q_max, hjb, config)

    assert pair.ask_price is None
    assert pair.ask is None
    assert pair.bid_price is not None


def test_inventory_skew_tilts_quotes_toward_unwinding():
    """Long inventory should make selling likelier than buying, per the book."""
    config = QuoteConfig(spread_multiplier=1.0)
    hjb = solve_hjb(LIVE_PARAMS, config)
    flat = compute_quotes(MID, 0, hjb, config)
    long_ = compute_quotes(MID, 2, hjb, config)

    assert flat.bid is not None and flat.ask is not None
    assert long_.bid is not None and long_.ask is not None
    # Holding inventory: the ask tightens and the bid widens.
    assert long_.ask.delta_model < flat.ask.delta_model
    assert long_.bid.delta_model > flat.bid.delta_model


def test_quotes_round_maker_safe_away_from_the_touch():
    config = QuoteConfig(spread_multiplier=1.0)
    hjb = solve_hjb(LIVE_PARAMS, config)
    unrounded = compute_quotes(MID, 0, hjb, config)
    rounded = compute_quotes(MID, 0, hjb, config, price_tick_size=0.1)

    assert rounded.bid_price is not None and rounded.ask_price is not None
    # Bid rounds down, ask rounds up -- never toward the mid.
    assert rounded.bid_price <= unrounded.bid_price
    assert rounded.ask_price >= unrounded.ask_price
    assert rounded.bid_price == pytest.approx(round(rounded.bid_price, 1))
    assert rounded.ask_price == pytest.approx(round(rounded.ask_price, 1))


def test_select_delta_maps_bid_to_delta_minus():
    """A bid fill increases inventory, so the bid is priced by delta_minus."""
    config = QuoteConfig()
    hjb = solve_hjb(LIVE_PARAMS, config)
    idx = int(np.argmin(np.abs(np.asarray(hjb["q_grid"]) - 1)))
    assert select_delta(hjb, 1, "bid") == pytest.approx(hjb["delta_minus"][idx])
    assert select_delta(hjb, 1, "ask") == pytest.approx(hjb["delta_plus"][idx])


def test_select_delta_clamps_beyond_the_grid():
    config = QuoteConfig()
    hjb = solve_hjb(LIVE_PARAMS, config)
    assert select_delta(hjb, 999, "ask") == pytest.approx(hjb["delta_plus"][-1])
    assert select_delta(hjb, -999, "bid") == pytest.approx(hjb["delta_minus"][0])


def test_select_delta_without_a_surface_is_none():
    assert select_delta({}, 0, "bid") is None


def test_solve_hjb_reports_the_phi_channel():
    config = QuoteConfig()
    quiet = solve_hjb(LIVE_PARAMS, config, sigma2_per_sec=None)
    loud = solve_hjb(LIVE_PARAMS, config, sigma2_per_sec=4.0)
    assert quiet["phi_source"] == "phi_base_fallback"
    assert quiet["phi_effective"] == pytest.approx(config.hjb_phi)
    assert loud["phi_source"] == "sigma2_channel"
    assert loud["phi_effective"] > config.hjb_phi


def test_effective_phi_falls_back_rather_than_stopping_the_quoter():
    config = QuoteConfig()
    for bad in (None, float("nan"), -1.0, "abc"):
        phi, source = effective_phi(config, bad)
        assert phi == pytest.approx(config.hjb_phi)
        assert source == "phi_base_fallback"


# --------------------------------------------------------------------------
# validate_param_snapshot
# --------------------------------------------------------------------------


def test_valid_snapshot_passes():
    config = QuoteConfig()
    assert validate_param_snapshot(*_snapshot(), config) == (True, "ok")


def test_lambda_must_come_from_the_survival_fit():
    """The raw trades/sec monitor differs by orders of magnitude; only one is
    the model's arrival rate."""
    config = QuoteConfig()
    snap = _snapshot(lambda_={"lambda_source": "trades_per_second"})
    assert validate_param_snapshot(*snap, config) == (False, "invalid_lambda_source")


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"kappa": {"schema_version": 2}}, "param_schema_unsupported"),
        ({"kappa": {"schema_version": None}}, "param_schema_unsupported"),
        ({"kappa": {"status": "stale"}}, "param_status_not_ok"),
        ({"kappa": {"generated_at": None}}, "missing_param_timestamp"),
        ({"kappa": {"generated_at": "not-a-date"}}, "invalid_param_timestamp"),
        ({"kappa": {"kappa+": 0.0}}, "invalid_kappa"),
        ({"kappa": {"kappa+": -1.0}}, "invalid_kappa"),
        ({"lambda_": {"lambda+": -1.0}}, "invalid_lambda"),
        ({"epsilon": {"epsilon+": -1.0}}, "invalid_epsilon"),
        ({"kappa": {"n_points_plus": 1}}, "insufficient_kappa_diagnostics"),
        ({"kappa": {"r2_plus": 0.01}}, "insufficient_kappa_diagnostics"),
        ({"epsilon": {"n_buy_events": 0}}, "insufficient_epsilon_diagnostics"),
    ],
)
def test_snapshot_rejection_reasons_are_stable(overrides, reason):
    """Gate evaluators count these strings, so they are a contract."""
    config = QuoteConfig()
    assert validate_param_snapshot(*_snapshot(**overrides), config)[1] == reason


def test_missing_and_nonfinite_params_are_named():
    config = QuoteConfig()
    kappa, epsilon, lambda_ = _snapshot()
    del kappa["kappa+"]
    assert validate_param_snapshot(kappa, epsilon, lambda_, config)[1] == "missing_kappa+"

    kappa, epsilon, lambda_ = _snapshot()
    kappa["kappa+"] = float("nan")
    assert validate_param_snapshot(kappa, epsilon, lambda_, config)[1] == "nonfinite_kappa+"


def test_stale_and_future_snapshots_are_rejected():
    config = QuoteConfig(max_param_age_seconds=90.0)
    now = datetime.now(timezone.utc)

    old = _snapshot(now=now - timedelta(seconds=600))
    assert validate_param_snapshot(*old, config, now=now)[1] == "stale_params"

    ahead = _snapshot(now=now + timedelta(seconds=600))
    assert validate_param_snapshot(*ahead, config, now=now)[1] == "param_timestamp_future"


def test_toxicity_gate_rejects_hostile_flow():
    """kappa*epsilon is the adverse-selection scale; past max_toxicity the
    model's own edge is gone."""
    config = QuoteConfig(max_toxicity=1.5)
    snap = _snapshot(kappa={"kappa+": 100.0, "kappa-": 100.0})
    assert validate_param_snapshot(*snap, config)[1] == "toxicity_too_high"


def test_incomplete_snapshot_is_rejected():
    config = QuoteConfig()
    kappa, epsilon, _ = _snapshot()
    assert validate_param_snapshot(kappa, epsilon, None, config)[1] == "param_schema_unsupported"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def test_finite_float_or_none():
    assert finite_float_or_none("1.5") == 1.5
    for bad in (None, "abc", float("nan"), float("inf"), [1]):
        assert finite_float_or_none(bad) is None


def test_parse_utc_timestamp_accepts_iso_seconds_and_millis():
    expected = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
    assert parse_utc_timestamp("2026-08-17T12:00:00Z") == expected
    assert parse_utc_timestamp(expected.timestamp()) == expected
    assert parse_utc_timestamp(expected.timestamp() * 1000.0) == expected
    # Naive input is assumed UTC rather than rejected.
    assert parse_utc_timestamp(datetime(2026, 8, 17, 12, 0)) == expected
    for bad in (None, "", "nonsense"):
        assert parse_utc_timestamp(bad) is None


def test_merged_params_flattens_and_tolerates_none():
    merged = merged_params({"a": 1}, None, {"b": 2})
    assert merged == {"a": 1, "b": 2}


def test_jsonl_logger_writes_envelope_and_rotates(tmp_path):
    """The "ts"/"event" envelope is a contract with the gate evaluators."""
    import json

    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(path, max_bytes=200)
    logger.emit("quote", {"side": "bid", "price": 1.0})
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["event"] == "quote"
    assert record["side"] == "bid"
    assert "ts" in record

    for i in range(200):
        logger.emit("quote", {"i": i})
    assert path.exists()
    assert path.stat().st_size <= 200 * 4  # rotation kept it bounded


def test_disabled_logger_writes_nothing(tmp_path):
    path = tmp_path / "events.jsonl"
    JsonlEventLogger(path, enabled=False).emit("quote", {"a": 1})
    assert not path.exists()


# --------------------------------------------------------------------------
# Two-instance side routing
# --------------------------------------------------------------------------


def test_routing_quotes_both_sides_when_flat():
    assert route_sides(0, 0, 3) == {"long": "bid", "short": "ask"}


def test_routing_stays_two_sided_at_interior_inventory():
    """Every interior state must present a bid AND an ask -- that is the whole
    point of running two instances."""
    for q_long in range(0, 4):
        for q_short in range(-3, 1):
            q_net = q_long + q_short
            if abs(q_net) >= 3:
                continue
            sides = route_sides(q_long, q_short, 3)
            assert {sides["long"], sides["short"]} == {"bid", "ask"}, (q_long, q_short)


def test_routing_prefers_unwinding_gross_inventory():
    """When both legs hold something, reduce rather than add -- otherwise gross
    inventory ratchets while net stays near zero."""
    assert route_sides(1, -1, 3) == {"long": "ask", "short": "bid"}
    assert route_sides(3, -3, 3) == {"long": "ask", "short": "bid"}
    assert route_sides(2, -1, 3) == {"long": "ask", "short": "bid"}


def test_routing_posts_one_side_at_the_net_boundary():
    """At +q_max the model disables the bid, so only an ask should be posted --
    once, by the leg that is reducing."""
    sides = route_sides(3, 0, 3)
    assert sides["long"] == "ask"
    assert sides["short"] is None

    sides = route_sides(0, -3, 3)
    assert sides["short"] == "bid"
    assert sides["long"] is None


def test_routing_never_posts_the_same_side_twice():
    """Two orders on one side would double size at one price and leave the book
    one-sided -- the failure the routing rule exists to prevent."""
    for q_long in range(0, 5):
        for q_short in range(-4, 1):
            sides = route_sides(q_long, q_short, 4)
            if sides["long"] is not None and sides["short"] is not None:
                assert sides["long"] != sides["short"], (q_long, q_short)


def test_routing_never_adds_beyond_the_boundary():
    """No bid at max long, no ask at max short, from either leg."""
    for q_long in range(0, 4):
        for q_short in range(-3, 1):
            q_net = q_long + q_short
            sides = set(route_sides(q_long, q_short, 3).values())
            if q_net >= 3:
                assert "bid" not in sides, (q_long, q_short)
            if q_net <= -3:
                assert "ask" not in sides, (q_long, q_short)


# --------------------------------------------------------------------------
# Net inventory resolution
# --------------------------------------------------------------------------


def _peer(q_own=-1, fingerprint="abc", age_seconds=0.0):
    stamp = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return {"q_own": q_own, "param_fingerprint": fingerprint, "published_at": stamp.isoformat()}


def test_net_inventory_sums_both_legs():
    assert resolve_net_inventory(2, _peer(q_own=-1), "abc") == (1, "ok")
    assert resolve_net_inventory(0, _peer(q_own=-3), "abc") == (-3, "ok")


def test_missing_peer_fails_closed_rather_than_assuming_flat():
    """A missing peer must not be read as q=0: flat is a plausible value that
    would silently mis-price every quote."""
    assert resolve_net_inventory(2, None, "abc") == (None, "peer_inventory_missing")


def test_stale_peer_fails_closed():
    q, reason = resolve_net_inventory(2, _peer(age_seconds=600), "abc", max_peer_age_seconds=30)
    assert q is None and reason == "peer_inventory_stale"


def test_param_fingerprint_mismatch_fails_closed():
    q, reason = resolve_net_inventory(2, _peer(fingerprint="other"), "abc")
    assert q is None and reason == "peer_param_fingerprint_mismatch"


def test_fingerprint_check_is_skipped_when_either_side_is_unknown():
    """Content-addressed comparison only; a missing fingerprint is not a
    mismatch, or the pair could never start up."""
    assert resolve_net_inventory(2, _peer(fingerprint=None), "abc")[1] == "ok"
    assert resolve_net_inventory(2, _peer(fingerprint="abc"), None)[1] == "ok"


def test_malformed_peer_payloads_fail_closed():
    bad_ts = {"q_own": 1, "param_fingerprint": "abc", "published_at": "nonsense"}
    assert resolve_net_inventory(0, bad_ts, "abc")[1] == "peer_inventory_timestamp_invalid"

    bad_q = {"q_own": "x", "param_fingerprint": "abc",
             "published_at": datetime.now(timezone.utc).isoformat()}
    assert resolve_net_inventory(0, bad_q, "abc")[1] == "peer_inventory_invalid"
