"""The tape and the configuration grid the replay's bit-identity check runs on.

Kept out of the test module so the SAME definitions can be replayed against an
older copy of ``replay_market_maker`` to regenerate the reference. If this file
and the reference JSON were ever regenerated together from the current code the
check would be circular and worthless, so:

    tests/data/replay_reference_metrics.json.gz WAS RECORDED FROM THE PRE-OPTIMISATION
    REPLAY (git 85bb564, 2026-08-17) AND MUST NOT BE REGENERATED FROM THE CURRENT
    ONE.

If a change genuinely has to move a number, the honest move is to say which
number and why in the commit, not to refresh the file and let the diff vanish.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_replay import synthetic_tape  # noqa: E402

# Gzipped: the metrics carry one markout sample per fill per horizon, so the
# grid is ~3.6 MB of JSON and ~0.3 MB compressed. Storing the whole payload
# rather than a digest of it means a failure can be read, not just detected.
REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "replay_reference_metrics.json.gz"


def load_reference() -> dict[str, Any]:
    import gzip
    import json

    with gzip.open(REFERENCE_PATH, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def dump_reference(payload: dict[str, Any]) -> None:
    import gzip
    import json

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(REFERENCE_PATH, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)

# Small enough that the whole grid runs in a few seconds, long enough that every
# branch fires: fills on both sides, episode rolls, queue decay, markouts.
REFERENCE_ROWS = 4_000
REFERENCE_SEED = 20260817

REFERENCE_PARAMS = {
    "kappa+": 10179.501680696987,
    "kappa-": 10131.639846298318,
    "lambda+": 0.18384454888412738,
    "lambda-": 0.2720996837612637,
    "epsilon+": 2.0594489447746147e-05,
    "epsilon-": 1.5018482890699448e-05,
}

_BASE: dict[str, Any] = dict(
    symbol="CASHCAT",
    data_dir=SCRIPTS / "HL_data",
    mid_fallback=0.1015,
    inventory_unit_base=2092.0,
    price_tick_size=1e-6,
    maker_fee=0.00015,
    taker_fee=0.00045,
    starting_equity_usdc=1000.0,
    leverage=1.0,
)

# The axes the harness is actually swept along. Both latency extremes, both ends
# of the requote interval, a phi that clamps and one that does not, the two-sided
# and the long-only inventory domain, and both time modes.
_LATENCIES = {"lat50": (25, 25, 25), "lat500": (250, 250, 250)}
_REFRESHES = {"ref100": 100, "ref30000": 30000}
_PHIS = {"phi10": (10.0, 50.0), "phi300": (300.0, 300.0)}
_INVENTORY = {"twosided6": dict(q_max=6, q_min=-6), "longonly3": dict(q_max=3, q_min=0)}
_TIME_MODES = {
    "episodic": dict(hjb_time_mode="episodic", hjb_horizon_seconds=30.0),
    "stationary": dict(hjb_time_mode="stationary", hjb_horizon_seconds=30.0),
}


def reference_configs() -> list[tuple[str, dict[str, Any]]]:
    """The grid, as (name, ReplayConfig kwargs). Deterministic order."""
    out: list[tuple[str, dict[str, Any]]] = []
    for lat, ref, phi, inventory, mode in itertools.product(
        _LATENCIES, _REFRESHES, _PHIS, _INVENTORY, _TIME_MODES
    ):
        decision, ack, cancel = _LATENCIES[lat]
        phi_kappa_t, phi_max = _PHIS[phi]
        kwargs = dict(_BASE)
        kwargs.update(
            decision_latency_ms=decision,
            order_ack_latency_ms=ack,
            cancel_latency_ms=cancel,
            quote_refresh_interval_ms=_REFRESHES[ref],
            hjb_phi_kappa_t=phi_kappa_t,
            hjb_phi_kappa_t_max=phi_max,
            hjb_alpha_kappa=0.05,
            **_INVENTORY[inventory],
            **_TIME_MODES[mode],
        )
        out.append((f"{lat}_{ref}_{phi}_{inventory}_{mode}", kwargs))

    def variant(name: str, **overrides: Any) -> None:
        kwargs = dict(_BASE)
        kwargs.update(
            decision_latency_ms=250,
            order_ack_latency_ms=250,
            cancel_latency_ms=250,
            quote_refresh_interval_ms=1000,
            hjb_phi_kappa_t=10.0,
            hjb_phi_kappa_t_max=50.0,
            hjb_alpha_kappa=0.05,
            q_max=6,
            q_min=-6,
            hjb_time_mode="episodic",
            hjb_horizon_seconds=30.0,
        )
        kwargs.update(overrides)
        out.append((name, kwargs))

    # Branches the cross above never reaches.
    #
    # join_touch pins the half-spread to the value the synthetic book quotes at
    # (see benchmark_replay.JOIN_HALF_SPREAD_BPS), so the quote rests ON the
    # touch. Without it nothing here would ever call is_joining_best true, and
    # the queue-ahead lookup and the whole queue-decay accumulation would be
    # unexercised -- a third of the fill rule, silently untested.
    # The cap does the pinning: phi*kappa*T = 300 puts the model depth around
    # 40 bps, so every quote clamps to max_half_spread_bps exactly. (min == max
    # is rejected by QuoteConfig, which is why this is a cap and not a band.)
    from benchmark_replay import JOIN_HALF_SPREAD_BPS  # noqa: E402

    join = dict(
        hjb_phi_kappa_t=300.0,
        hjb_phi_kappa_t_max=300.0,
        max_half_spread_bps=JOIN_HALF_SPREAD_BPS,
    )
    variant("join_touch_episodic", **join)
    variant("join_touch_slow_decay", **join, queue_decay_per_second=0.5, quote_refresh_interval_ms=5000)
    variant("join_touch_stationary", **join, hjb_time_mode="stationary")
    variant("funding", funding_rate_per_hour=0.0001)
    variant("hjb_qmin0", hjb_q_min=0, q_min=0, q_max=3)
    variant("sigma2", sigma2_per_sec=1e-8)
    variant("amount_step", amount_step_size=1000.0)
    variant("raw_phi", hjb_phi_kappa_t=0.0, hjb_alpha_kappa=0.0, hjb_phi=0.0001, hjb_alpha=0.001)
    variant("max_price_events", max_price_events=1500)
    variant("no_reset_on_flat", episode_reset_on_flat=False)
    variant("no_queue_decay", queue_decay_per_second=0.0)
    variant("no_tick", price_tick_size=0.0)
    return out


def reference_tape(module: Any) -> Any:
    """The pinned tape, as ``module``'s own ReplayTape.

    Takes the module so the frames can be handed to an older copy of the replay
    for regeneration; the frames themselves are built by one generator, so the
    two copies score the same bytes.
    """
    tape = synthetic_tape(REFERENCE_ROWS, seed=REFERENCE_SEED)
    return module.ReplayTape(
        prices=tape.prices,
        trades=tape.trades,
        orderbooks=tape.orderbooks,
        input_files=dict(tape.input_files),
    )
