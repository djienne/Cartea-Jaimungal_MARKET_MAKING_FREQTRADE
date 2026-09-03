#!/usr/bin/env python3
"""Event replay harness for the market-making model.

This is intentionally conservative. It is not a candle backtest: quotes that
cross the book are post-only rejects, quotes away from touch wait for the book to
move through them, and quotes at touch need observed traded volume before fill.

Four things a reader of any artifact this harness produces has to know:

- **The solve is the Python parity solve.** ``solve_replay_hjb`` routes through
  ``mm_core.solve_hjb``, which derives phi and alpha from the DIMENSIONLESS
  targets ``hjb_phi_kappa_t`` / ``hjb_alpha_kappa`` and the fitted kappa. Sweeping the
  raw phi/alpha means nothing across symbols (eq. 10.28: they are not
  kappa-invariant). Rust implements the corresponding solve independently and
  parity tests compare selected surfaces and quotes.
- **Latency is a headline axis, not a sensitivity.** A quote decided at a book
  event only rests ``decision_latency_ms + order_ack_latency_ms`` later and stays
  cancellable for ``cancel_latency_ms`` after it goes stale. The 250 + 250 ms
  default is retained to reproduce early artifacts; it is not the current Rust
  dry-run profile. State the latency of a run, never infer it from defaults.
- **Staleness is set by the slower of two channels**: order latency and
  ``quote_refresh_interval_ms``. At the 1 s refresh default, dropping latency
  500 -> 50 ms only moves total staleness from ~1.5 s to ~1.05 s, so a
  latency-only sweep reads nearly flat and invites the wrong conclusion. Sweep
  them together only as explicitly labelled joint scenarios, and vary latency
  alone when the goal is to isolate latency. A joint scenario is not causal
  evidence for either component.
- **The clock is the exchange's**, not the collector's receive clock; see
  :func:`apply_event_clock`. And a tape can only resolve latency down to its own
  inter-update cadence -- ``price_event_cadence_ms`` in the metrics reports it,
  so no reader can over-read a rung that sits below it.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from flow_guard import (
    FlowGuard,
    FlowGuardConfig,
    MidWindow,
    VpinTracker,
    bucket_volume_from_totals,
)
from hjb import compute_h_asymmetric, compute_h_symmetric
import mm_core
from mm_core import QuoteConfig, finite_float_or_none


MAKER_FEE = 0.00015
TAKER_FEE = 0.00045
# Quote-assembly defaults. mm_core is the single implementation these
# delegate to; tests/test_quote_assembly.py asserts the delegation.
MIN_HALF_SPREAD_BPS = 1.5
MAX_HALF_SPREAD_BPS = 80.0
MARKOUT_HORIZONS_MS = (100, 1_000, 5_000, 30_000)
PARAMETER_SERIES_UNIT = {
    "kappa": "1/USDC",
    "lambda": "events/second",
    "epsilon": "USDC",
}


@dataclass
class ReplayConfig:
    symbol: str
    data_dir: Path
    mid_fallback: float
    inventory_unit_base: float = 0.01
    q_max: int = 3
    # 0 reproduces the long-only agent the harness shipped with; set to -q_max
    # to simulate the two-sided market maker the book actually describes.
    q_min: int = 0
    # Inventory domain the HJB is SOLVED on, which is deliberately separate from
    # the clamp above. None keeps the legacy symmetric [-q_max, q_max] solve --
    # which, with q_min=0, is the known defect: the boundary is part of the
    # optimisation (eq. 10.4), so a long-only agent priced off a symmetric solve
    # is using the wrong control. Keeping both knobs lets the A/B measure it.
    hjb_q_min: int | None = None
    spread_multiplier: float = 1.0
    min_half_spread_bps: float = MIN_HALF_SPREAD_BPS
    max_half_spread_bps: float = MAX_HALF_SPREAD_BPS
    # Latency, in three parts, all sweepable from the CLI. decision+ack is when
    # the quote starts resting; cancel is how long it keeps resting after it
    # should have been pulled. The 250+250 = 500 ms default is deliberately the
    # historical end -- early artifacts used it, so it stays the default for
    # reproducibility rather than as a current performance claim.
    decision_latency_ms: int = 250
    order_ack_latency_ms: int = 250
    cancel_latency_ms: int = 250
    # The OTHER staleness channel, and usually the dominant one: at 1000 ms the
    # quote is re-priced once a second no matter how fast the exchange acks. A
    # latency sweep at a fixed refresh interval measures mostly the interval.
    # The current Rust runtime is event-driven; this interval is a replay-model
    # parameter retained for historical scenario reproduction.
    quote_refresh_interval_ms: int = 1000
    maker_fee: float = MAKER_FEE
    taker_fee: float = TAKER_FEE
    funding_rate_per_hour: float = 0.0
    starting_equity_usdc: float = 1000.0
    leverage: float = 1.0
    maintenance_margin_rate: float = 0.05
    queue_decay_per_second: float = 0.05
    price_tick_size: float = 0.0
    amount_step_size: float = 0.0
    fill_calibration_path: Path | None = None
    newest_per_stream: int | None = None
    max_price_events: int | None = None
    # HJB risk/horizon knobs, previously buried as params.get() defaults inside
    # compute_hjb_cache where no caller could reach them. They have to be
    # sweepable: T is the episode length in episodic mode, and phi/alpha are the
    # two penalties whose RATIO decides whether the terminal condition matters.
    hjb_alpha: float = 0.001
    hjb_phi: float = 0.0001
    # The DIMENSIONLESS targets the live path actually configures (see the
    # comments above QuoteConfig.hjb_phi_kappa_t). phi and alpha are not
    # kappa-invariant, so these -- not the raw values above -- are what a sweep
    # can carry from one symbol to another. Both default to 0, which disables the
    # derivation and leaves mm_core.solve_hjb using the raw hjb_phi/hjb_alpha, so
    # every historical CLI run (docs/time_mode_ab.md) still reproduces. Live sets
    # 10.0 / 0.05.
    hjb_phi_kappa_t: float = 0.0
    hjb_alpha_kappa: float = 0.0
    # Ceiling on the same dimensionless product, tracked from QuoteConfig so
    # replay clamps where live clamps. Sweeping hjb_phi_kappa_t past this without
    # raising it produces a flat curve, because every setting above the ceiling
    # solves the identical problem.
    hjb_phi_kappa_t_max: float = QuoteConfig.hjb_phi_kappa_t_max
    gamma_inventory_risk: float = QuoteConfig.gamma_inventory_risk
    # Volatility channel: phi_effective = phi + gamma*sigma2*inventory_unit_base.
    # None reproduces a run with the channel quiet, which is also what the live
    # path does when the estimator publishes no sigma2.
    sigma2_per_sec: float | None = None
    hjb_horizon_seconds: float = 60.0
    # "episodic" runs the book's delta*(t,q) on a simulated clock; "stationary"
    # reads only the t=0 slice, which is what this harness did before.
    hjb_time_mode: str = "episodic"
    # End the episode early once flat, but only after this fraction of T, so a
    # single round trip cannot pin the clock at t=0 forever.
    episode_reset_on_flat: bool = True
    episode_min_elapsed_fraction: float = 0.25
    # Which of the collector's two clocks orders the event stream. See
    # apply_event_clock: "exchange" is correct, "local" reproduces artifacts
    # recorded before 2026-08-17.
    event_clock: str = "exchange"
    # Toxic-flow guard, on by default so the replay models what the live bot
    # actually runs. Until 2026-09-02 the Python replay had no guard at all,
    # which biased every sweep: the 08-22 cascade sits in the sweep's TRAIN
    # slice, so selection was tuning the inventory penalty to avoid a cascade
    # the guard already bounds. Semantics and defaults are the Rust ones
    # (rust_live/crates/mm-runtime/src/flow_guard.rs).
    flow_guard_enabled: bool = True
    flow_guard_fast_move_window_ms: float = FlowGuardConfig.fast_move_window_ms
    flow_guard_fast_move_threshold_bps: float = FlowGuardConfig.fast_move_threshold_bps
    flow_guard_vpin_buckets_per_day: int = FlowGuardConfig.vpin_buckets_per_day
    flow_guard_vpin_window_buckets: int = FlowGuardConfig.vpin_window_buckets
    flow_guard_vpin_threshold: float = FlowGuardConfig.vpin_threshold
    flow_guard_cooldown_ms: float = FlowGuardConfig.cooldown_ms
    flow_guard_warmup_reentry_cooldown_ms: float = FlowGuardConfig.warmup_reentry_cooldown_ms
    flow_guard_warmup_reentry_calm_fraction: float = FlowGuardConfig.warmup_reentry_calm_fraction
    # Aggressive flatten. None keeps the shipped behaviour: hold inventory until
    # an offsetting MAKER fill arrives, which takes 6.5 s median and eats the
    # full adverse-selection accrual -- eps(d) grows from 12.84 bps at 200 ms to
    # 29.77 bps at 6.6 s, which is the whole reason the strategy loses. Setting
    # this crosses the spread to go flat once a lot has been held that long,
    # paying half-spread plus taker fee to truncate the markout.
    flatten_after_ms: float | None = None
    # What crossing actually costs beyond the touch. Our lot is 2.5-2.9x the
    # median touch size and exceeds it on 81-88% of events, so a flatten walks
    # the book; measured from the 20-level snapshots the volume-weighted fill
    # sits 1.6-2.1 bps (median) past the touch. Charging zero here is the single
    # most flattering assumption this policy could make.
    flatten_slippage_bps: float = 2.5
    # How much resting size a sweep must clear before it reaches our quote.
    #   "touch_only"  -- the shipped model: the visible touch size when we join
    #                    the best, and ZERO for anything behind it. A quote at
    #                    60 bps against a 5.6 bps half-spread therefore fills on
    #                    the first print that reaches its price, as though we
    #                    were always first in queue at that level.
    #   "book_level"  -- size resting at OUR price level, from the 20-level
    #                    snapshots (median ~2652 units at 60 bps out, against our
    #                    2092-unit lot). This is the correct pairing with the
    #                    fill rule, which decrements the queue only on prints at
    #                    our price or beyond.
    #   "book_depth"  -- cumulative at-or-better (~51k units at 60 bps). Charges
    #                    the whole ladder against prints that can only be a
    #                    fraction of it, so it over-penalises by ~19x. Kept
    #                    deliberately as a stress bound, not as the model.
    queue_model: str = "touch_only"


@dataclass
class ReplayMetrics:
    input_files: dict[str, int] = field(default_factory=dict)
    input_rows: dict[str, int] = field(default_factory=dict)
    data_start: str | None = None
    data_end: str | None = None
    data_span_seconds: float = 0.0
    price_event_count: int = 0
    price_events_per_day: float = 0.0
    max_price_gap_seconds: float | None = None
    p95_price_gap_seconds: float | None = None
    # Inter-arrival of the BBO stream this run actually scored. The tape cannot
    # resolve a latency shorter than its own cadence: if total latency is under
    # one inter-update gap the quote goes live before the book has moved at all,
    # so two such rungs produce the same fills. Measured on CASHCAT at
    # p10 67 ms / p50 ~203 ms / p90 ~0.85 s, which is why a 50 ms rung is an
    # UPPER BOUND on this tape and not a forecast: the real book moves faster
    # than the collector recorded it.
    price_gap_ms_p10: float | None = None
    price_gap_ms_p50: float | None = None
    price_gap_ms_p90: float | None = None
    event_clock: str = ""
    event_clock_by_stream: dict[str, str] = field(default_factory=dict)
    decision_latency_ms: int = 0
    order_ack_latency_ms: int = 0
    cancel_latency_ms: int = 0
    quote_attempts: int = 0
    quote_decision_events: int = 0
    post_only_rejects: int = 0
    maker_fills: int = 0
    taker_fills: int = 0
    stale_quote_cancels: int = 0
    quote_refresh_interval_ms: int = 0
    consumed_trade_events: int = 0
    calibration_rejected_fills: int = 0
    price_tick_size: float = 0.0
    amount_step_size: float = 0.0
    price_rounding_adjustments: int = 0
    amount_rounding_rejects: int = 0
    calibration_attempts_by_key: dict[str, int] = field(default_factory=dict)
    calibration_fills_by_key: dict[str, int] = field(default_factory=dict)
    fill_calibration: dict[str, Any] = field(default_factory=dict)
    realized_spread_usdc: float = 0.0
    fees_usdc: float = 0.0
    cash_usdc: float = 0.0
    inventory_base: float = 0.0
    final_mid: float | None = None
    mark_to_market_pnl_usdc: float = 0.0
    funding_usdc: float = 0.0
    starting_equity_usdc: float = 0.0
    equity_usdc: float = 0.0
    min_equity_usdc: float | None = None
    notional_exposure_usdc: float = 0.0
    margin_used_usdc: float = 0.0
    max_margin_used_usdc: float = 0.0
    maintenance_margin_usdc: float = 0.0
    min_liquidation_buffer_usdc: float | None = None
    liquidation_breach_events: int = 0
    # Toxic-flow guard observability. `flow_guard_trips` counts fresh trips
    # (a re-arm inside an open trip is not a new one); `withheld` counts quote
    # decisions the guard blanked, which is the analogue of a Rust
    # QuoteReason::ToxicFlow publication.
    flow_guard_enabled: bool = False
    flow_guard_trips: int = 0
    flow_guard_withheld_decisions: int = 0
    flow_guard_vpin_bucket_volume: float = 0.0
    flow_guard_vpin_buckets_seen: int = 0
    # How long inventory is actually held, FIFO-matched fill against offsetting
    # fill. This exists to CHOOSE THE MARKOUT HORIZON rather than assume one:
    # the fill-conditional adverse selection eps(d) grows steeply with the
    # horizon it is measured over (scripts/epsilon_conditional.py -- on the
    # held-out slice the per-fill edge at 26 bps runs +12.23 bps at 200 ms,
    # +0.96 at 1 s and -4.10 at 5 s), so "how long do we hold" decides whether
    # the strategy has an edge at all. Weighted by size, since a partial fill
    # held for an hour is not the same evidence as a full lot held for a second.
    holding_time_seconds_weighted: float = 0.0
    holding_time_base_matched: float = 0.0
    holding_time_pairs: int = 0
    holding_time_base_unmatched: float = 0.0
    # Aggressive-flatten accounting, kept apart from maker economics so the two
    # legs stay separable: this is what the truncation COST.
    flatten_events: int = 0
    flatten_base: float = 0.0
    flatten_cost_usdc: float = 0.0
    queue_decay_base: float = 0.0
    holding_time_samples: list[tuple[float, float]] = field(default_factory=list)
    time_at_q_boundary: dict[str, int] = field(default_factory=lambda: {"q_min": 0, "q_max": 0})
    inventory_histogram: dict[int, int] = field(default_factory=dict)
    # Fractional inventory the integer grid cannot represent. The book's q is a
    # unit-jump count (eq. 10.2); partial fills are not, so this is the size of
    # the gap between the model's state and the real one.
    q_residual_abs_sum: float = 0.0
    q_residual_samples: int = 0
    episodes: int = 0
    pnl_by_side: dict[str, float] = field(default_factory=lambda: {"bid": 0.0, "ask": 0.0})
    # Per-side attribution. The edge on CASHCAT is strongly one-sided, and WHICH
    # side is not stable: the depth study found the resting ask profitable past
    # ~14 bps and the resting bid negative at every depth out to 34 bps, while
    # the first replay to carry these fields (3.4 h tape, 2026-08-17) came back
    # with the signs reversed -- 30 s markout +0.29 USDC/fill on the bid against
    # -0.27 on the ask, on a tape that trended up. That is the whole argument for
    # measuring it per run: a total alone cannot distinguish a working ask
    # carrying a losing bid from neither side working. Depth is the mean over
    # fills of the quote's distance from mid at the decision that placed it.
    fills_by_side: dict[str, int] = field(default_factory=lambda: {"bid": 0, "ask": 0})
    fill_depth_bps_sum_by_side: dict[str, float] = field(default_factory=lambda: {"bid": 0.0, "ask": 0.0})
    markout_usdc_by_side: dict[str, dict[int, float]] = field(
        default_factory=lambda: {"bid": {}, "ask": {}}
    )
    markout_fills_by_side: dict[str, dict[int, int]] = field(
        default_factory=lambda: {"bid": {}, "ask": {}}
    )
    quote_attempts_by_depth: dict[str, int] = field(default_factory=dict)
    fills_by_depth: dict[str, int] = field(default_factory=dict)
    markout_samples: list[dict[str, Any]] = field(default_factory=list)
    parameter_series: list[dict[str, Any]] = field(default_factory=list)
    toxicity_series: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        attempts = max(self.quote_attempts, 1)
        fills = self.maker_fills + self.taker_fills
        fill_ratio_by_depth = {
            key: self.fills_by_depth.get(key, 0) / max(attempts_at_depth, 1)
            for key, attempts_at_depth in sorted(self.quote_attempts_by_depth.items())
        }
        total_quote_latency_ms = int(self.decision_latency_ms) + int(self.order_ack_latency_ms)
        # How long the quote actually rests: it goes live at total latency, is
        # replaced at the refresh interval, and lingers one cancel latency past
        # whichever came later. Note that CUTTING latency at a fixed refresh
        # interval LENGTHENS this -- the quote goes live sooner and is still
        # pulled at refresh+cancel -- which is why the two must be swept together.
        quote_exposure_ms = max(0, int(self.quote_refresh_interval_ms) - total_quote_latency_ms) + int(
            self.cancel_latency_ms
        )
        mean_fill_depth_bps_by_side = {
            side: (
                self.fill_depth_bps_sum_by_side.get(side, 0.0) / self.fills_by_side[side]
                if self.fills_by_side.get(side, 0)
                else None
            )
            for side in ("bid", "ask")
        }
        markout_by_side = {
            side: {
                str(horizon_ms): {
                    "count": int(self.markout_fills_by_side.get(side, {}).get(horizon_ms, 0)),
                    "sum_usdc": float(self.markout_usdc_by_side.get(side, {}).get(horizon_ms, 0.0)),
                    "mean_usdc": (
                        float(self.markout_usdc_by_side[side][horizon_ms])
                        / int(self.markout_fills_by_side[side][horizon_ms])
                        if int(self.markout_fills_by_side.get(side, {}).get(horizon_ms, 0))
                        else None
                    ),
                }
                for horizon_ms in MARKOUT_HORIZONS_MS
            }
            for side in ("bid", "ask")
        }
        return {
            "latency": {
                "decision_latency_ms": int(self.decision_latency_ms),
                "order_ack_latency_ms": int(self.order_ack_latency_ms),
                "cancel_latency_ms": int(self.cancel_latency_ms),
                "total_quote_latency_ms": total_quote_latency_ms,
                "quote_refresh_interval_ms": int(self.quote_refresh_interval_ms),
                "quote_exposure_ms": quote_exposure_ms,
                # True when the quote is live before the tape's fastest decile of
                # book updates: the rung is then bounded by data cadence, not by
                # the simulator, and reads as an upper bound.
                "below_tape_resolution": (
                    None
                    if self.price_gap_ms_p10 is None
                    else bool(total_quote_latency_ms < float(self.price_gap_ms_p10))
                ),
            },
            "price_event_cadence_ms": {
                "p10": self.price_gap_ms_p10,
                "p50": self.price_gap_ms_p50,
                "p90": self.price_gap_ms_p90,
            },
            "event_clock": self.event_clock,
            "event_clock_by_stream": self.event_clock_by_stream,
            "fills_by_side": self.fills_by_side,
            "mean_fill_depth_bps_by_side": mean_fill_depth_bps_by_side,
            "markout_by_side": markout_by_side,
            "input_files": self.input_files,
            "input_rows": self.input_rows,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "data_span_seconds": self.data_span_seconds,
            "price_event_count": self.price_event_count,
            "price_events_per_day": self.price_events_per_day,
            "max_price_gap_seconds": self.max_price_gap_seconds,
            "p95_price_gap_seconds": self.p95_price_gap_seconds,
            "quote_attempts": self.quote_attempts,
            "quote_decision_events": self.quote_decision_events,
            # to_dict enumerates its keys, so a new ReplayMetrics field is NOT
            # carried across automatically. The guard counters were added
            # without this line and every sweep artifact recorded trips=0 while
            # the guard was demonstrably firing.
            "flow_guard_enabled": self.flow_guard_enabled,
            "flow_guard_trips": self.flow_guard_trips,
            "flow_guard_withheld_decisions": self.flow_guard_withheld_decisions,
            "flow_guard_vpin_bucket_volume": self.flow_guard_vpin_bucket_volume,
            "flow_guard_vpin_buckets_seen": self.flow_guard_vpin_buckets_seen,
            # The horizon evidence. `mean` is size-weighted; `unmatched_base` is
            # inventory still open when the tape ended, which is censored rather
            # than short-lived and is therefore excluded from the statistics
            # instead of being counted as a fast round trip.
            "flatten": {
                "events": self.flatten_events,
                "base": self.flatten_base,
                "cost_usdc": self.flatten_cost_usdc,
            },
            "holding_time": {
                "pairs": self.holding_time_pairs,
                "matched_base": self.holding_time_base_matched,
                "unmatched_base": self.holding_time_base_unmatched,
                "mean_seconds": (
                    self.holding_time_seconds_weighted / self.holding_time_base_matched
                    if self.holding_time_base_matched > 0.0
                    else None
                ),
                "p50_seconds": holding_time_percentile(self.holding_time_samples, 0.50),
                "p90_seconds": holding_time_percentile(self.holding_time_samples, 0.90),
            },
            "post_only_rejects": self.post_only_rejects,
            "post_only_reject_ratio": self.post_only_rejects / attempts,
            "maker_fills": self.maker_fills,
            "taker_fills": self.taker_fills,
            "maker_ratio": self.maker_fills / max(fills, 1),
            "stale_quote_cancels": self.stale_quote_cancels,
            "stale_quote_cancel_ratio": self.stale_quote_cancels / attempts,
            "quote_refresh_interval_ms": self.quote_refresh_interval_ms,
            "consumed_trade_events": self.consumed_trade_events,
            "calibration_rejected_fills": self.calibration_rejected_fills,
            "price_tick_size": self.price_tick_size,
            "amount_step_size": self.amount_step_size,
            "price_rounding_adjustments": self.price_rounding_adjustments,
            "amount_rounding_rejects": self.amount_rounding_rejects,
            "calibration_attempts_by_key": self.calibration_attempts_by_key,
            "calibration_fills_by_key": self.calibration_fills_by_key,
            "fill_calibration": self.fill_calibration,
            "realized_spread_usdc": self.realized_spread_usdc,
            "fees_usdc": self.fees_usdc,
            "cash_usdc": self.cash_usdc,
            "inventory_base": self.inventory_base,
            "final_mid": self.final_mid,
            "mark_to_market_pnl_usdc": self.mark_to_market_pnl_usdc,
            "funding_usdc": self.funding_usdc,
            "starting_equity_usdc": self.starting_equity_usdc,
            "equity_usdc": self.equity_usdc,
            "min_equity_usdc": self.min_equity_usdc,
            "notional_exposure_usdc": self.notional_exposure_usdc,
            "margin_used_usdc": self.margin_used_usdc,
            "max_margin_used_usdc": self.max_margin_used_usdc,
            "maintenance_margin_usdc": self.maintenance_margin_usdc,
            "min_liquidation_buffer_usdc": self.min_liquidation_buffer_usdc,
            "liquidation_breach_events": self.liquidation_breach_events,
            "queue_decay_base": self.queue_decay_base,
            "time_at_q_boundary": self.time_at_q_boundary,
            "inventory_histogram": self.inventory_histogram,
            "mean_abs_q_residual": (
                self.q_residual_abs_sum / self.q_residual_samples
                if self.q_residual_samples
                else None
            ),
            "episodes": self.episodes,
            "pnl_by_side": self.pnl_by_side,
            "quote_attempts_by_depth": self.quote_attempts_by_depth,
            "fills_by_depth": self.fills_by_depth,
            "fill_ratio_by_depth": fill_ratio_by_depth,
            "markout_samples": self.markout_samples,
            "parameter_series": self.parameter_series,
            "toxicity_series": self.toxicity_series,
        }


def selected_parquet_files(path: Path, newest_per_stream: int | None = None) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if newest_per_stream is not None and newest_per_stream > 0:
        files = sorted(sorted(files, key=lambda file: file.stat().st_mtime, reverse=True)[:newest_per_stream])
    return files


def drop_duplicate_trades(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse trades recorded twice by two collectors sharing one output dir.

    Shards are concatenated blindly, so a second collector pointed at the same
    directory produces two rows per trade that differ only in the local receive
    `timestamp` — which would hand the replay twice the fill opportunities and
    flatter the result. That happened on 2026-08-16 (hl-collector +
    hl-collector2 both writing scripts/HL_data). tid is the exchange's own
    per-trade id, so this is exact and a no-op for a correct single-writer
    directory. The prices/orderbooks streams carry no trade_id and pass through
    untouched — they have no comparably reliable unique key, and collapsing them
    on their exchange fields would risk merging genuinely distinct events
    whenever exchange_timestamp is sparse.
    """
    if frame.empty or "trade_id" not in frame.columns:
        return frame
    tid = frame["trade_id"].astype(str)
    # The collector stores str(trade.get("tid")), so a feed message with no id
    # lands as the literal "None". Those rows are distinct trades and must never
    # be collapsed into one, so only de-duplicate genuine ids.
    has_id = ~tid.isin({"", "None", "nan", "NaT", "<NA>"})
    return frame[~(has_id & tid.duplicated(keep="first"))].reset_index(drop=True)


def load_parquet_dir(path: Path, newest_per_stream: int | None = None) -> tuple[pd.DataFrame, int]:
    files = selected_parquet_files(path, newest_per_stream)
    if not files:
        return pd.DataFrame(), 0
    # selected_parquet_files() returns shard names sorted, and the name carries
    # the flush timestamp, so keep="first" deterministically keeps the earliest
    # copy of a duplicated trade.
    frame = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
    return drop_duplicate_trades(frame), len(files)


def normalize_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "timestamp" not in frame:
        return frame
    frame = frame.copy()
    series = frame["timestamp"]
    if pd.api.types.is_numeric_dtype(series):
        finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            frame["timestamp"] = pd.to_datetime(series, utc=True, errors="coerce")
            return frame
        median_abs = float(finite.abs().median())
        if median_abs >= 1e18:
            unit = "ns"
        elif median_abs >= 1e15:
            unit = "us"
        elif median_abs >= 1e12:
            unit = "ms"
        else:
            unit = "s"
        frame["timestamp"] = pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
    else:
        frame["timestamp"] = pd.to_datetime(series, utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"])


def normalize_price_bbo(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    if {"bid", "ask"}.issubset(prices.columns):
        return prices
    if not {"timestamp", "side", "price"}.issubset(prices.columns):
        return prices

    frame = prices.copy()
    frame["side"] = frame["side"].astype(str).str.lower()
    frame = frame[frame["side"].isin({"bid", "ask"})]
    if frame.empty:
        return prices

    price_pivot = frame.pivot_table(index="timestamp", columns="side", values="price", aggfunc="last")
    size_pivot = frame.pivot_table(index="timestamp", columns="side", values="size", aggfunc="last") if "size" in frame else pd.DataFrame(index=price_pivot.index)
    bbo = pd.DataFrame(index=price_pivot.index)
    if "bid" in price_pivot:
        bbo["bid"] = price_pivot["bid"]
    if "ask" in price_pivot:
        bbo["ask"] = price_pivot["ask"]
    if "bid" in size_pivot:
        bbo["bid_size"] = size_pivot["bid"]
    if "ask" in size_pivot:
        bbo["ask_size"] = size_pivot["ask"]
    bbo = bbo.sort_index().ffill().dropna(subset=["bid", "ask"]).reset_index()
    return bbo


def apply_event_clock(frame: pd.DataFrame, event_clock: str) -> tuple[pd.DataFrame, str]:
    """Re-key one stream onto the exchange's clock, and say which clock won.

    The collector records two: ``timestamp`` is when the local process received
    the message, ``exchange_timestamp`` is when Hyperliquid stamped it. This
    harness ordered events by the RECEIVE clock until 2026-08-17, which is fine
    for anything measured in seconds and wrong for anything measured in
    milliseconds. On the CASHCAT tape the receive lag runs 349/397/559 ms
    (p10/p50/p90) on prices but 425/470/671 ms on orderbooks, so the streams are
    smeared against each other by more than the latency a sweep is trying to
    resolve: a 50 ms rung read off the receive clock is measuring collector
    jitter, not the market. The lag itself is harmless (it shifts every stream
    alike); the per-message and per-stream VARIANCE in it is not.

    All-or-nothing per stream, and it falls back rather than failing: an older
    tape with no exchange_timestamp still replays, it just cannot resolve tens of
    milliseconds. Mixing the two clocks in one column would also break the unit
    inference in :func:`normalize_timestamp_column`, which reads the magnitude.
    """
    if str(event_clock) != "exchange":
        return frame, "local"
    if frame.empty or "exchange_timestamp" not in frame.columns:
        return frame, "local_no_exchange_timestamp"
    exchange = pd.to_numeric(frame["exchange_timestamp"], errors="coerce")
    if exchange.isna().any() or bool((exchange <= 0).any()):
        return frame, "local_invalid_exchange_timestamp"
    frame = frame.copy()
    frame["timestamp"] = exchange
    return frame, "exchange"


def load_symbol_data(config: ReplayConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    base = config.data_dir / config.symbol
    prices, price_files = load_parquet_dir(base / "prices", config.newest_per_stream)
    trades, trade_files = load_parquet_dir(base / "trades", config.newest_per_stream)
    orderbooks, orderbook_files = load_parquet_dir(base / "orderbooks", config.newest_per_stream)
    # Pick the clock BEFORE normalising: normalize_timestamp_column infers the
    # unit from the column's magnitude, so the swap has to happen while the
    # column is still raw. (This replaced a sort loop that rebound its own loop
    # variable and therefore normalised nothing -- the real normalisation has
    # always been the lines below.)
    prices, prices_clock = apply_event_clock(prices, config.event_clock)
    trades, trades_clock = apply_event_clock(trades, config.event_clock)
    orderbooks, orderbooks_clock = apply_event_clock(orderbooks, config.event_clock)
    prices = normalize_price_bbo(normalize_timestamp_column(prices))
    trades = normalize_timestamp_column(trades)
    orderbooks = normalize_timestamp_column(orderbooks)
    for frame, clock in ((prices, prices_clock), (trades, trades_clock), (orderbooks, orderbooks_clock)):
        if not frame.empty and "timestamp" in frame:
            frame.sort_values("timestamp", inplace=True)
        # Rides along on the frame so the metrics can state which clock scored
        # the run without load_symbol_data having to grow a fifth return value
        # (tests monkeypatch it with a four-tuple).
        frame.attrs["event_clock"] = clock
    return prices, trades, orderbooks, {
        "prices": price_files,
        "trades": trade_files,
        "orderbooks": orderbook_files,
    }


@dataclass(frozen=True)
class ReplayTape:
    """One immutable load of the collected data, scored by many configs.

    The collector keeps writing shards while a sweep runs, so calling
    ``load_symbol_data`` per variant scores DIFFERENT data every few minutes.
    That is not hypothetical: it invalidated a phi sweep where one setting came
    back at both -3.98 and -19.39 USDC, and docs/time_mode_ab.md had to
    monkeypatch the loader to work around it. Load once, pass the tape to every
    run, and the only thing that differs between rows of a sweep is the setting.

    ``run_replay`` never mutates these frames -- it re-indexes and filters, both
    of which copy -- so one tape is safely reusable for any number of runs.
    """

    prices: pd.DataFrame
    trades: pd.DataFrame
    orderbooks: pd.DataFrame
    input_files: dict[str, int]

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
        return self.prices, self.trades, self.orderbooks, dict(self.input_files)


def load_tape(config: ReplayConfig) -> ReplayTape:
    """Load the data ONCE so a sweep can pin it. See :class:`ReplayTape`."""
    prices, trades, orderbooks, input_files = load_symbol_data(config)
    return ReplayTape(prices=prices, trades=trades, orderbooks=orderbooks, input_files=input_files)


def post_only_check(side: str, price: float, best_bid: float, best_ask: float) -> tuple[bool, str]:
    try:
        price, best_bid, best_ask = (float(price), float(best_bid), float(best_ask))
    except (TypeError, ValueError):
        return False, "crossed_or_invalid_book"
    # math.isfinite, not the generator over np.isfinite this used to run: the
    # three values are already Python floats by here, so the answer is the same
    # and the generator plus three numpy scalar dispatches per quote was costing
    # about a second per pass over a 148k-row tape.
    if not (math.isfinite(price) and math.isfinite(best_bid) and math.isfinite(best_ask)):
        return False, "crossed_or_invalid_book"
    if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
        return False, "crossed_or_invalid_book"
    if side == "bid" and price >= best_ask:
        return False, "bid_crosses_ask"
    if side == "ask" and price <= best_bid:
        return False, "ask_crosses_bid"
    return True, "ok"


def _inventory_config(unit: float, q_max: int, q_min: int = 0) -> QuoteConfig:
    return QuoteConfig(
        inventory_unit_base=max(float(unit), 1e-12),
        q_max=max(int(q_max), 1),
        allow_short=int(q_min) < 0,
    )


def inventory_q(inventory_base: float, unit: float, q_max: int, q_min: int = 0) -> int:
    """Map a base position onto the integer inventory grid.

    ``q_min`` defaults to 0, reproducing the long-only clamp this harness
    shipped with. Pass a negative ``q_min`` to simulate the two-sided agent.
    Delegates to mm_core so replay and live share one mapping.
    """
    return mm_core.inventory_to_q(inventory_base, _inventory_config(unit, q_max, q_min))


def inventory_q_exact(inventory_base: float, unit: float, q_max: int, q_min: int = 0) -> float:
    """The same mapping WITHOUT the rounding -- what the depths price off.

    This harness fills partially by construction (``fill_size`` is capped by the
    matched trade's size), so its inventory sits between grid points more often
    than not. That is the same gap the live path has, which makes this the place
    to measure it.
    """
    return mm_core.inventory_to_q_exact(
        inventory_base, _inventory_config(unit, q_max, q_min)
    )


def hjb_quote_config(config: ReplayConfig) -> QuoteConfig:
    """The QuoteConfig the replay's solve runs on -- the live object, live values.

    Built from ReplayConfig so a swept setting means the same thing here as
    everywhere else that constructs a QuoteConfig. Note
    ``inventory_unit_base`` and ``gamma_inventory_risk``
    matter to the SOLVE, not just to sizing: mm_core.effective_phi multiplies
    sigma2 by both.
    """
    return QuoteConfig(
        maker_fee_rate=float(config.maker_fee),
        spread_multiplier=float(config.spread_multiplier),
        min_half_spread_bps=float(config.min_half_spread_bps),
        max_half_spread_bps=float(config.max_half_spread_bps),
        inventory_unit_base=max(float(config.inventory_unit_base), 1e-12),
        q_max=max(int(config.q_max), 1),
        allow_short=int(config.q_min) < 0,
        hjb_phi_kappa_t=float(config.hjb_phi_kappa_t),
        hjb_alpha_kappa=float(config.hjb_alpha_kappa),
        hjb_phi_kappa_t_max=float(config.hjb_phi_kappa_t_max),
        hjb_alpha=float(config.hjb_alpha),
        hjb_phi=float(config.hjb_phi),
        hjb_horizon_seconds=float(config.hjb_horizon_seconds),
        gamma_inventory_risk=float(config.gamma_inventory_risk),
        hjb_time_mode=str(config.hjb_time_mode),
    )


def solve_hjb_on_domain(
    params: dict[str, float],
    solver_config: QuoteConfig,
    *,
    q_min: int | None = None,
    sigma2_per_sec: float | None = None,
) -> dict:
    """mm_core.solve_hjb, extended to this harness's asymmetric inventory domain.

    Everything that decides WHAT is solved -- deriving phi and alpha from the
    dimensionless targets and live kappa, the hjb_phi_kappa_t_max ceiling, the
    sigma2 volatility channel, the step count -- belongs to mm_core, because a
    replay that re-derives any of it is a second implementation that will drift
    from the quoter. So the symmetric solve is always mm_core's.

    ``q_min`` is the one thing mm_core.solve_hjb cannot express: it always solves
    the symmetric [-q_max, q_max] grid, and the long-only A/B needs [0, q_max],
    where the disabled side at the bottom is the ask ("cannot sell what you do
    not hold") rather than the ask at maximum short. Rather than edit mm_core --
    the live path has no asymmetric domain and does not need one -- the
    asymmetric case re-solves on the reachable grid with the penalties mm_core
    just derived. That costs one extra solve (~0.3 s at q_max=6, T=150 s), once
    per run, and only when an asymmetric domain is actually asked for; the
    default q_min=None never pays it.
    """
    solved = mm_core.solve_hjb(params, solver_config, sigma2_per_sec=sigma2_per_sec)
    q_max = int(solver_config.q_max)
    if q_min is None or int(q_min) == -q_max:
        return solved

    solver = compute_h_asymmetric if solver_config.use_asymmetric_kappa else compute_h_symmetric
    asymmetric = dict(
        solver(
            lambda_plus=float(params["lambda+"]),
            lambda_minus=float(params["lambda-"]),
            epsilon_plus=float(params["epsilon+"]),
            epsilon_minus=float(params["epsilon-"]),
            kappa_plus=float(params["kappa+"]),
            kappa_minus=float(params["kappa-"]),
            alpha=float(solved["alpha_effective"]),
            phi=float(solved["phi_effective"]),
            T_seconds=float(solver_config.hjb_horizon_seconds),
            q_max=q_max,
            q_min=int(q_min),
            n_steps=mm_core.hjb_n_steps(solver_config),
            return_surface=solver_config.hjb_time_mode == "episodic",
        )
    )
    for key in (
        "hjb_time_mode",
        "phi_effective",
        "phi_source",
        "phi_base",
        "alpha_effective",
        "kappa_avg",
        "sigma2_per_sec",
    ):
        asymmetric[key] = solved[key]
    return asymmetric


def solve_replay_hjb(config: ReplayConfig, params: dict[str, float]) -> dict:
    """The surface run_replay quotes off. One line, on purpose: it is the whole
    parity claim -- same params, same QuoteConfig, same mm_core.solve_hjb the
    strategy calls."""
    return solve_hjb_on_domain(
        params,
        hjb_quote_config(config),
        q_min=config.hjb_q_min,
        sigma2_per_sec=config.sigma2_per_sec,
    )


def compute_hjb_cache(
    params: dict[str, float],
    q_max: int,
    q_min: int | None = None,
    *,
    alpha: float | None = None,
    phi: float | None = None,
    T_seconds: float | None = None,
    time_mode: str = "stationary",
    max_dt_seconds: float = 0.25,
) -> dict:
    """Solve on the requested domain from RAW alpha/phi, via mm_core.

    This is the pre-dimensionless entry point, kept because raw phi/alpha are
    still the fallback the CLI exposes and what docs/time_mode_ab.md was run
    with. It reaches the solver through the same mm_core.solve_hjb the live path
    uses, with the kappa-relative targets disabled (0), which is exactly the
    branch in solve_hjb that passes the raw values through untouched.

    ``q_min=None`` keeps the symmetric grid. Passing 0 solves the long-only
    problem, where the disabled side at the bottom of the grid is the ask.

    ``time_mode="episodic"`` also returns the full delta*(t,q) surface so the
    caller can quote off its actual time-to-go rather than the t=0 slice.
    """
    horizon = float(params.get("T_seconds", 60.0) if T_seconds is None else T_seconds)
    solver_config = QuoteConfig(
        q_max=max(int(q_max), 1),
        hjb_phi_kappa_t=0.0,
        hjb_alpha_kappa=0.0,
        hjb_alpha=float(params.get("alpha", 0.001) if alpha is None else alpha),
        hjb_phi=float(params.get("phi", 0.0001) if phi is None else phi),
        hjb_horizon_seconds=horizon,
        hjb_max_dt_seconds=float(max_dt_seconds),
        hjb_time_mode=str(time_mode),
    )
    return solve_hjb_on_domain(params, solver_config, q_min=q_min)


def timestamp_iso(value: Any) -> str | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.isoformat()


def replay_parameter_snapshot(
    config: ReplayConfig,
    params: dict[str, float],
    ts: Any,
    *,
    data_start: str | None,
    data_end: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ts": timestamp_iso(ts),
        "source": "static_replay_params",
        "symbol": config.symbol,
        "unit": dict(PARAMETER_SERIES_UNIT),
        "data_start": data_start,
        "data_end": data_end,
        "kappa_plus": finite_float_or_none(params.get("kappa+")),
        "kappa_minus": finite_float_or_none(params.get("kappa-")),
        "lambda_plus": finite_float_or_none(params.get("lambda+")),
        "lambda_minus": finite_float_or_none(params.get("lambda-")),
        "epsilon_plus": finite_float_or_none(params.get("epsilon+")),
        "epsilon_minus": finite_float_or_none(params.get("epsilon-")),
    }


def replay_toxicity_snapshot(
    config: ReplayConfig,
    params: dict[str, float],
    ts: Any,
    *,
    data_start: str | None,
    data_end: str | None,
) -> dict[str, Any]:
    kappa_plus = finite_float_or_none(params.get("kappa+")) or 0.0
    kappa_minus = finite_float_or_none(params.get("kappa-")) or 0.0
    epsilon_plus = finite_float_or_none(params.get("epsilon+")) or 0.0
    epsilon_minus = finite_float_or_none(params.get("epsilon-")) or 0.0
    return {
        "schema_version": 1,
        "ts": timestamp_iso(ts),
        "source": "static_replay_params",
        "symbol": config.symbol,
        "unit": "kappa_times_epsilon",
        "formula": "kappa * epsilon",
        "data_start": data_start,
        "data_end": data_end,
        "toxicity_plus": kappa_plus * epsilon_plus,
        "toxicity_minus": kappa_minus * epsilon_minus,
    }


def half_spread_quote_config(
    *,
    spread_multiplier: float = 1.0,
    maker_fee: float = MAKER_FEE,
    min_half_spread_bps: float = MIN_HALF_SPREAD_BPS,
    max_half_spread_bps: float = MAX_HALF_SPREAD_BPS,
) -> QuoteConfig:
    """The QuoteConfig :func:`assemble_half_spread` prices off.

    Split out so ``run_replay`` can build it ONCE instead of once per side per
    quote decision -- it is a function of the run's configuration, not of the
    event -- while leaving exactly one place that decides which fields go in.
    """
    return QuoteConfig(
        maker_fee_rate=float(maker_fee),
        spread_multiplier=float(spread_multiplier),
        min_half_spread_bps=float(min_half_spread_bps),
        max_half_spread_bps=float(max_half_spread_bps),
    )


def assemble_half_spread(
    delta_model: float,
    mid: float,
    *,
    spread_multiplier: float = 1.0,
    maker_fee: float = MAKER_FEE,
    min_half_spread_bps: float = MIN_HALF_SPREAD_BPS,
    max_half_spread_bps: float = MAX_HALF_SPREAD_BPS,
    quote_config: QuoteConfig | None = None,
) -> float | None:
    """Final half-spread for one side, delegated to mm_core:

        delta_total = clamp(delta_model * spread_multiplier + maker_fee*mid,
                            min_half_spread_bps, max_half_spread_bps)

    Returns None for a disabled side (non-finite model delta). mm_core is the
    single implementation shared with the live path, so replay simulates
    literally the arithmetic that will quote.

    ``quote_config`` supplies a pre-built config; it must be the one
    :func:`half_spread_quote_config` would have built from the same keywords.
    """
    config = (
        quote_config
        if quote_config is not None
        else half_spread_quote_config(
            spread_multiplier=spread_multiplier,
            maker_fee=maker_fee,
            min_half_spread_bps=min_half_spread_bps,
            max_half_spread_bps=max_half_spread_bps,
        )
    )
    spread = mm_core.assemble_half_spread(delta_model, mid, config)
    return None if spread is None else float(spread.delta)


def compute_quotes(
    mid: float,
    q: int,
    params: dict[str, float],
    q_max: int,
    hjb: dict | None = None,
    *,
    maker_fee: float = MAKER_FEE,
    spread_multiplier: float = 1.0,
    min_half_spread_bps: float = MIN_HALF_SPREAD_BPS,
    max_half_spread_bps: float = MAX_HALF_SPREAD_BPS,
    tau_remaining: float | None = None,
    q_exact: float | None = None,
    quote_config: QuoteConfig | None = None,
) -> tuple[float | None, float | None, dict]:
    if hjb is None:
        hjb = compute_hjb_cache(params, q_max)
    if quote_config is None:
        quote_config = half_spread_quote_config(
            spread_multiplier=spread_multiplier,
            maker_fee=maker_fee,
            min_half_spread_bps=min_half_spread_bps,
            max_half_spread_bps=max_half_spread_bps,
        )
    # Read the surface through mm_core rather than indexing it here: this used
    # to do its own argmin over q_grid, which is precisely the kind of second
    # implementation this module exists to avoid. mm_core also handles the two
    # things a bare argmin cannot -- the time slice and fractional q.
    q_price = float(q) if q_exact is None else float(q_exact)
    bid_total = assemble_half_spread(
        mm_core.select_delta(hjb, q_price, "bid", tau_remaining=tau_remaining),
        mid,
        quote_config=quote_config,
    )
    ask_total = assemble_half_spread(
        mm_core.select_delta(hjb, q_price, "ask", tau_remaining=tau_remaining),
        mid,
        quote_config=quote_config,
    )
    bid = None if bid_total is None else mid - bid_total
    ask = None if ask_total is None else mid + ask_total
    return bid, ask, hjb


def mid_from_price_row(row: pd.Series, fallback: float) -> tuple[float, float, float]:
    best_bid = float(row.get("bid", row.get("best_bid", np.nan)))
    best_ask = float(row.get("ask", row.get("best_ask", np.nan)))
    if not np.isfinite(best_bid) or not np.isfinite(best_ask):
        return fallback, np.nan, np.nan
    return (best_bid + best_ask) / 2.0, best_bid, best_ask


# ---------------------------------------------------------------------------
# Column views
#
# run_replay used to read every field through pandas scalar access -- iterrows
# for the price stream, a full-frame boolean mask per quote to cut the trade
# window, .iloc on an 83-column orderbook frame per quote. That is microseconds
# per field on work that is a handful of flops, and it made one pass over a
# 148k-row tape take about five minutes; the sweep driver runs hundreds of such
# passes. Everything below hoists a column into a numpy array ONCE per run so
# the loop indexes it.
#
# The rule these helpers are written to: the arithmetic is not allowed to
# change, only the access. Where the row-wise version used min/max/float those
# calls are kept verbatim, because reassociating them would move fills.
# tests/test_replay_performance.py pins each helper against the row-wise
# function it replaces, and pins whole-run outputs against a reference recorded
# before any of this existed.
# ---------------------------------------------------------------------------

_NS_PER_MS = 1_000_000
_NS_PER_SECOND = 1_000_000_000

# Column names first_level_size consults, per side, in priority order.
_FIRST_LEVEL_SIZE_KEYS: dict[str, tuple[str, ...]] = {
    "bid": ("bid_size", "best_bid_size", "bid_size_0", "bids"),
    "ask": ("ask_size", "best_ask_size", "ask_size_0", "asks"),
}
_NESTED_LEVEL_KEYS = frozenset({"bids", "asks"})


def total_seconds_from_ns(delta_ns: int) -> float:
    """``pd.Timedelta(delta_ns, "ns").total_seconds()`` without the Timedelta.

    Reproduced rather than approximated. The episode clock, the funding accrual
    and the queue decay all read this number, so a different rounding would move
    fills, and pandas does NOT compute what either obvious shortcut computes: it
    truncates to whole microseconds and then adds the microsecond part to the
    integer-second part in floating point. 1904529860 ns is 1.9045290000000001 s
    to pandas, 1.90452986 s to ``ns / 1e9`` and 1.904529 s to the exact
    ``ns // 1000 / 1e6``. tests/test_replay_performance.py checks the identity
    over 200k samples spanning both signs and seventeen decades.
    """
    return (delta_ns // _NS_PER_SECOND) + ((delta_ns % _NS_PER_SECOND) // 1000) / 1e6


def timestamps_as_ns(
    frame: pd.DataFrame, column: str = "timestamp", *, coerce: bool = False
) -> np.ndarray | None:
    """UTC nanoseconds as int64, or None when the column is not a datetime one.

    Any resolution is widened to nanoseconds, which is lossless and leaves
    Timedelta's microsecond truncation (see :func:`total_seconds_from_ns`)
    untouched. Returning None rather than raising lets the caller decide: an
    absent column is legitimate on some streams and fatal on others.

    ``coerce`` converts a present-but-not-datetime column element by element,
    exactly as the row-wise code's ``pd.Timestamp(cell)`` did. It is off by
    default because the whole point here is the one cheap vectorised conversion;
    the public DataFrame adapters turn it on so they keep accepting frames the
    replay itself would have rejected earlier.
    """
    if frame is None or column not in getattr(frame, "columns", ()):
        return None
    series = frame[column]
    if not pd.api.types.is_datetime64_any_dtype(series):
        if not coerce:
            return None
        return np.array([timestamp_as_ns(value) for value in series], dtype="int64")
    return series.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)


def timestamp_as_ns(value: Any) -> int | None:
    """One timestamp as UTC nanoseconds, matching :func:`timestamps_as_ns`."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if stamp.unit != "ns":
        stamp = stamp.as_unit("ns")
    return int(stamp.value)


def is_monotonic_ns(values: np.ndarray | None) -> bool:
    """Is this timestamp array non-decreasing?

    Only the TRADE stream needs the answer. The original cut its window with a
    boolean mask, which does not care about order, so a searchsorted range is
    only equivalent when the column is sorted -- and a monkeypatched loader in a
    test is free to hand over an unsorted frame. The price and orderbook streams
    were already read with ``searchsorted`` before this rewrite, so they carry
    the same assumption they always did and need no guard.
    """
    if values is None or len(values) < 2:
        return True
    return bool(np.all(values[1:] >= values[:-1]))


def price_event_arrays(prices: pd.DataFrame, fallback: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`mid_from_price_row` for a whole frame, column-wise.

    Same column preference (``bid`` then ``best_bid``) -- which is a property of
    the frame's columns, not of any row, so resolving it once is exact -- and
    the same "either side non-finite disables the row" rule. The mid is an IEEE
    double add and divide either way, so the values are bit-identical to the
    per-row function.
    """
    n = len(prices)
    columns = prices.columns

    def column(primary: str, secondary: str) -> np.ndarray:
        if primary in columns:
            return prices[primary].to_numpy(dtype=float)
        if secondary in columns:
            return prices[secondary].to_numpy(dtype=float)
        return np.full(n, np.nan, dtype=float)

    best_bid = column("bid", "best_bid")
    best_ask = column("ask", "best_ask")
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(best_bid) & np.isfinite(best_ask)
        mid = np.where(usable, (best_bid + best_ask) / 2.0, float(fallback))
    nan = np.full(n, np.nan, dtype=float)
    return mid, np.where(usable, best_bid, nan), np.where(usable, best_ask, nan)


def _float_column(frame: pd.DataFrame, name: str, missing: float) -> tuple[np.ndarray, np.ndarray]:
    """One column as float64, plus the mask of cells that are falsy.

    The falsy mask exists because both trade-size call sites read the value as
    ``float(x or default)`` and the two defaults DIFFER -- 0.0 in the queue walk,
    the order amount at the fill -- so the substitution cannot be baked in here.
    Under that idiom 0.0 and None take the default while NaN does not, which is
    why None cannot simply be converted to NaN.
    """
    n = len(frame)
    if name not in frame.columns:
        return np.full(n, missing, dtype=float), np.zeros(n, dtype=bool)
    series = frame[name]
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_object_dtype(series):
        values = series.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            return values, values == 0.0
    raw = series.to_numpy(dtype=object)
    values = np.empty(n, dtype=float)
    falsy = np.empty(n, dtype=bool)
    for i, item in enumerate(raw):
        falsy[i] = not bool(item)
        values[i] = 0.0 if item is None else float(item)
    return values, falsy


# Enough samples for stable percentiles without holding a list the length of
# the fill count on a multi-hundred-hour tape.
HOLDING_TIME_SAMPLE_CAP = 200_000


def trade_event_arrays(trades: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(price, size, size_is_falsy) for a trade frame."""
    price, _ = _float_column(trades, "price", np.nan)
    size, size_falsy = _float_column(trades, "size", 0.0)
    if "size" not in trades.columns:
        # ``row.get("size", default)`` returns the DEFAULT when the column is
        # absent, and both defaults are truthy at their call sites, so an absent
        # column must read as falsy for every row.
        size_falsy = np.ones(len(trades), dtype=bool)
    return price, size, size_falsy


def match_holding_time(
    open_lots: deque, fill_ts_ns: int, signed_size: float, metrics: "ReplayMetrics"
) -> None:
    """FIFO-match a fill against opposing open lots and record how long they were held.

    A bid fill first closes any open short lots, oldest first, and only the
    remainder opens a new long lot (and symmetrically for an ask). That is the
    same accounting a venue applies to a position, so "holding time" here is the
    life of an actual exposure rather than the gap between consecutive fills --
    quoting both sides means those two differ by orders of magnitude.

    Size-weighted on purpose: this feeds a horizon choice, and an unweighted mean
    would let a dust partial count as much as a full lot.
    """
    remaining = float(signed_size)
    while remaining != 0.0 and open_lots:
        lot_ts, lot_size = open_lots[0]
        if (lot_size > 0.0) == (remaining > 0.0):
            break  # same direction: this fill adds to the position, nothing closes
        matched = min(abs(lot_size), abs(remaining))
        if matched <= 0.0:
            break
        held_seconds = max(0.0, total_seconds_from_ns(fill_ts_ns - lot_ts))
        metrics.holding_time_seconds_weighted += held_seconds * matched
        metrics.holding_time_base_matched += matched
        metrics.holding_time_pairs += 1
        if len(metrics.holding_time_samples) < HOLDING_TIME_SAMPLE_CAP:
            metrics.holding_time_samples.append((held_seconds, matched))
        if abs(lot_size) > matched:
            open_lots[0] = (lot_ts, lot_size - math.copysign(matched, lot_size))
        else:
            open_lots.popleft()
        remaining -= math.copysign(matched, remaining)
    if remaining != 0.0:
        open_lots.append((fill_ts_ns, remaining))


def holding_time_percentile(samples: list[tuple[float, float]], quantile: float) -> float | None:
    """Size-weighted percentile of the holding-time samples."""
    if not samples:
        return None
    ordered = sorted(samples)
    total = sum(weight for _, weight in ordered)
    if total <= 0.0:
        return None
    target = float(quantile) * total
    seen = 0.0
    for seconds, weight in ordered:
        seen += weight
        if seen >= target:
            return float(seconds)
    return float(ordered[-1][0])


def trade_flow_arrays(trades: pd.DataFrame) -> tuple[list[bool], list[str]]:
    """(is_buy, trade_id) for the VPIN feed.

    The fill simulator matches on price alone and never needed the aggressor
    side; VPIN is an order-flow imbalance and cannot be computed without it.
    A missing side column reads as sell so an absent column cannot manufacture
    a one-sided imbalance.
    """
    if "side" in trades.columns:
        is_buy = (trades["side"].astype(str).str.lower() == "buy").tolist()
    else:
        is_buy = [False] * len(trades)
    if "trade_id" in trades.columns:
        trade_ids = trades["trade_id"].astype(str).tolist()
    else:
        trade_ids = [""] * len(trades)
    return is_buy, trade_ids


def first_level_size_array(frame: pd.DataFrame, side: str) -> np.ndarray:
    """:func:`first_level_size` for every row of ``frame``, column-wise.

    Walks the same key preference in the same order and falls through on the
    same failures, but resolves a whole column at a time: a numeric column
    converts for every row at once (``float(nan)`` succeeds, so NaN is a
    resolved value and not a fallthrough), and only object or nested-list
    columns need a per-row pass.
    """
    n = len(frame)
    out = np.zeros(n, dtype=float)
    resolved = np.zeros(n, dtype=bool)
    for key in _FIRST_LEVEL_SIZE_KEYS[side]:
        if key not in frame.columns or resolved.all():
            continue
        series = frame[key]
        nested = key in _NESTED_LEVEL_KEYS
        if not nested and pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_object_dtype(series):
            pending = ~resolved
            out[pending] = series.to_numpy(dtype=float)[pending]
            resolved[pending] = True
            continue
        raw = series.to_numpy(dtype=object)
        for i in np.nonzero(~resolved)[0]:
            value = raw[i]
            if value is None:
                continue
            try:
                out[i] = float(value[0][1]) if nested else float(value)
            except Exception:
                continue
            resolved[i] = True
    return out


# A markout is only a markout if the mid is observed near the horizon. The
# search takes the FIRST mid at or after it, so a fill landing just before a
# data outage would otherwise have its "30 s markout" measured against a mid
# minutes later, charging the whole outage's drift to that one fill. Past the
# end of the tape both functions already decline; an outage is the same
# situation and gets the same answer.
#
# Both implementations carry it, because test_replay_performance asserts they
# agree fill-for-fill -- and that guard is what caught the tolerance being added
# to only one of them.
MARKOUT_MAX_STALENESS_MS = 5_000


def future_mid(
    prices: pd.DataFrame,
    start_ts: pd.Timestamp,
    horizon_ms: int,
    fallback: float,
    max_staleness_ms: int | None = MARKOUT_MAX_STALENESS_MS,
) -> float | None:
    target = start_ts + pd.Timedelta(milliseconds=horizon_ms)
    idx = prices["timestamp"].searchsorted(target, side="left")
    if idx >= len(prices):
        return None
    found = prices["timestamp"].iloc[int(idx)]
    if max_staleness_ms is not None:
        if (found - target) > pd.Timedelta(milliseconds=int(max_staleness_ms)):
            return None
    mid, _, _ = mid_from_price_row(prices.iloc[int(idx)], fallback)
    return mid


def future_mid_from_arrays(
    price_ts_ns: np.ndarray,
    price_mid: np.ndarray,
    start_ns: int,
    horizon_ms: int,
    max_staleness_ms: int = MARKOUT_MAX_STALENESS_MS,
) -> float | None:
    """:func:`future_mid` against the hoisted columns.

    The old version searched the frame and then materialised a Series per fill
    per horizon -- four Series for every fill. Same ``side="left"`` search on the
    same ascending timestamps, same mid.

    Returns None when the nearest mid at or after the horizon is more than
    ``max_staleness_ms`` past it, rather than reporting a drift measurement as a
    markout.
    """
    target_ns = start_ns + horizon_ms * _NS_PER_MS
    idx = int(price_ts_ns.searchsorted(target_ns, side="left"))
    if idx >= len(price_ts_ns):
        return None
    if max_staleness_ms is not None:
        if int(price_ts_ns[idx]) - target_ns > int(max_staleness_ms) * _NS_PER_MS:
            return None
    return float(price_mid[idx])


def markout_value(side: str, fill_price: float, future_mid_price: float) -> float:
    if side == "bid":
        return future_mid_price - fill_price
    return fill_price - future_mid_price


def quote_depth_bps(mid: float, price: float) -> float:
    """Distance from mid in bps -- the coordinate the per-side attribution, the
    depth buckets and the fill calibration all have to agree on."""
    return 0.0 if mid <= 0 else abs(float(mid) - float(price)) / float(mid) * 10_000.0


def quote_depth_key(side: str, mid: float, price: float) -> str:
    return f"{side}:{quote_depth_bps(mid, price):.2f}bps"


def quote_depth_bucket_key(side: str, mid: float, price: float, bucket_bps: float) -> str:
    depth_bps = quote_depth_bps(mid, price)
    bucket = max(0.0, float(bucket_bps))
    if bucket <= 0:
        return f"{side}:{depth_bps:.2f}bps"
    lower = np.floor(float(depth_bps) / bucket) * bucket
    upper = lower + bucket
    return f"{side}:{lower:.2f}-{upper:.2f}bps"


def usable_tick_size(tick_size: float) -> float | None:
    """The tick, or None when there is no usable one. Loop-invariant per run."""
    tick = finite_float_or_none(tick_size)
    return None if tick is None or tick <= 0 else tick


def round_price_to_tick(side: str, price: float, tick: float) -> float:
    """Round one price onto a KNOWN-GOOD tick.

    Split out of :func:`round_price_for_side` only so the caller can resolve the
    tick once instead of per quote. The np.floor/np.ceil and the 12-digit round
    are kept exactly as they were: ``round`` on a numpy scalar dispatches to
    numpy's scale-and-rint, which is not the same algorithm as Python's round on
    a float, so swapping in math.floor would silently change quoted prices.
    """
    scaled = float(price) / tick
    if side == "bid":
        rounded = np.floor(scaled) * tick
    else:
        rounded = np.ceil(scaled) * tick
    return float(round(rounded, 12))


def round_price_for_side(side: str, price: float, tick_size: float) -> float:
    tick = usable_tick_size(tick_size)
    if tick is None:
        return float(price)
    return round_price_to_tick(side, price, tick)


def round_amount_down(amount: float, step_size: float) -> float:
    step = finite_float_or_none(step_size)
    if step is None or step <= 0:
        return max(0.0, float(amount))
    rounded = np.floor(float(amount) / step) * step
    return float(round(max(0.0, rounded), 12))


def load_fill_calibration(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "usable": False,
            "applied": False,
            "path": None,
            "reasons": ["not_supplied"],
            "bucket_bps": None,
            "fill_probability_by_depth": {},
            "fill_probability_by_side": {},
        }
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "provided": True,
            "usable": False,
            "applied": False,
            "path": str(path),
            "reasons": ["missing_calibration_file"],
            "bucket_bps": None,
            "fill_probability_by_depth": {},
            "fill_probability_by_side": {},
        }
    except Exception as exc:
        return {
            "provided": True,
            "usable": False,
            "applied": False,
            "path": str(path),
            "reasons": [f"invalid_calibration_file:{exc}"],
            "bucket_bps": None,
            "fill_probability_by_depth": {},
            "fill_probability_by_side": {},
        }
    if not isinstance(payload, dict):
        return {
            "provided": True,
            "usable": False,
            "applied": False,
            "path": str(path),
            "reasons": ["calibration_not_object"],
            "bucket_bps": None,
            "fill_probability_by_depth": {},
            "fill_probability_by_side": {},
        }

    usable = bool(payload.get("usable_for_calibration"))
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    bucket_bps = finite_float_or_none(inputs.get("bucket_bps"))
    by_depth_raw = payload.get("fill_probability_by_depth") if isinstance(payload.get("fill_probability_by_depth"), dict) else {}
    by_side_raw = payload.get("fill_probability_by_side") if isinstance(payload.get("fill_probability_by_side"), dict) else {}
    by_depth = {
        str(key): float(value)
        for key, value in by_depth_raw.items()
        if finite_float_or_none(value) is not None and 0.0 <= float(value) <= 1.0
    }
    by_side = {
        str(key).lower(): float(value)
        for key, value in by_side_raw.items()
        if finite_float_or_none(value) is not None and 0.0 <= float(value) <= 1.0
    }
    reasons = [str(reason) for reason in payload.get("reasons", []) if str(reason)]
    if usable and bucket_bps is None:
        usable = False
        reasons.append("missing_calibration_bucket_bps")
    if usable and not by_depth and not by_side:
        usable = False
        reasons.append("missing_calibration_probabilities")
    return {
        "provided": True,
        "usable": usable,
        "applied": usable,
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "reasons": reasons,
        "bucket_bps": bucket_bps,
        "accepted_quotes": inputs.get("accepted_quotes"),
        "maker_fills": payload.get("maker_fills"),
        "maker_ratio": payload.get("maker_ratio"),
        "fill_probability_by_depth": by_depth,
        "fill_probability_by_side": by_side,
    }


def calibration_probability_key(calibration: dict[str, Any], side: str, depth_key: str) -> tuple[str, float | None]:
    if not calibration.get("applied"):
        return depth_key, None
    by_depth = calibration.get("fill_probability_by_depth") or {}
    if depth_key in by_depth:
        return depth_key, float(by_depth[depth_key])
    by_side = calibration.get("fill_probability_by_side") or {}
    side_key = str(side).lower()
    if side_key in by_side:
        return f"{side_key}:side", float(by_side[side_key])
    return depth_key, 0.0


def calibrated_fill_allowed(
    calibration: dict[str, Any],
    attempts_by_key: dict[str, int],
    fills_by_key: dict[str, int],
    *,
    side: str,
    depth_key: str,
) -> tuple[bool, str | None]:
    key, probability = calibration_probability_key(calibration, side, depth_key)
    if probability is None:
        return True, None
    attempts = int(attempts_by_key.get(key, 0))
    accepted_fills = int(fills_by_key.get(key, 0))
    max_fills = int(np.floor(float(attempts) * max(0.0, min(1.0, float(probability))) + 1e-12))
    if accepted_fills + 1 > max_fills:
        return False, key
    fills_by_key[key] = accepted_fills + 1
    return True, key


def update_margin_metrics(metrics: ReplayMetrics, config: ReplayConfig, mid: float) -> None:
    notional = abs(float(metrics.inventory_base)) * float(mid)
    leverage = max(float(config.leverage), 1e-12)
    maintenance_rate = max(float(config.maintenance_margin_rate), 0.0)
    equity = float(config.starting_equity_usdc) + float(metrics.cash_usdc) + float(metrics.inventory_base) * float(mid)
    margin_used = notional / leverage
    maintenance_margin = notional * maintenance_rate
    liquidation_buffer = equity - maintenance_margin

    metrics.starting_equity_usdc = float(config.starting_equity_usdc)
    metrics.equity_usdc = equity
    metrics.min_equity_usdc = equity if metrics.min_equity_usdc is None else min(metrics.min_equity_usdc, equity)
    metrics.notional_exposure_usdc = notional
    metrics.margin_used_usdc = margin_used
    metrics.max_margin_used_usdc = max(float(metrics.max_margin_used_usdc), margin_used)
    metrics.maintenance_margin_usdc = maintenance_margin
    metrics.min_liquidation_buffer_usdc = (
        liquidation_buffer
        if metrics.min_liquidation_buffer_usdc is None
        else min(metrics.min_liquidation_buffer_usdc, liquidation_buffer)
    )
    if notional > 0 and liquidation_buffer < 0:
        metrics.liquidation_breach_events += 1


def active_orderbook_row(orderbooks: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    if orderbooks.empty or "timestamp" not in orderbooks:
        return None
    idx = orderbooks["timestamp"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    return orderbooks.iloc[int(idx)]


def first_level_size(row: pd.Series | None, side: str) -> float:
    if row is None:
        return 0.0
    keys = _FIRST_LEVEL_SIZE_KEYS["bid"] if side == "bid" else _FIRST_LEVEL_SIZE_KEYS["ask"]
    for key in keys:
        value = row.get(key, None)
        if value is None:
            continue
        if key in _NESTED_LEVEL_KEYS:
            try:
                return float(value[0][1])
            except Exception:
                continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def book_depth_arrays(frame: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(prices, cumulative_sizes) per snapshot for one side of the book.

    Cumulative along the level axis, so `cum[i, j]` is all the size resting at
    level j or better in snapshot i. That is exactly the volume an incoming
    sweep must consume before it can reach a quote sitting at level j.

    Returns None when the frame carries no level columns, which keeps the
    caller on the touch-only queue model instead of silently pretending the
    book is empty -- an empty book would mean "we are always first in queue",
    the most flattering possible assumption.
    """
    prices, sizes = [], []
    level = 0
    while f"{side}_price_{level}" in frame.columns and f"{side}_size_{level}" in frame.columns:
        prices.append(pd.to_numeric(frame[f"{side}_price_{level}"], errors="coerce").to_numpy(float))
        sizes.append(pd.to_numeric(frame[f"{side}_size_{level}"], errors="coerce").to_numpy(float))
        level += 1
    if not prices:
        return None
    price_matrix = np.column_stack(prices)
    size_matrix = np.nan_to_num(np.column_stack(sizes), nan=0.0)
    return price_matrix, np.cumsum(size_matrix, axis=1)


def queue_ahead_from_book(
    price_matrix: np.ndarray,
    cum_sizes: np.ndarray,
    book_idx: int,
    side: str,
    price: float,
    *,
    cumulative: bool = False,
) -> float:
    """Resting size a sweep must clear before reaching our quote at ``price``.

    Assumes we joined the BACK of the queue at our own level, which is right for
    a quote re-placed every few hundred ms: it has no aged priority to claim.

    ``cumulative`` selects which of two quantities is returned, and the choice
    MATTERS because it has to pair with how the queue is consumed.
    ``scan_for_matching_trade`` decrements only on prints at our price or beyond
    (``candidate_price >= price`` for an ask). The volume that clears the better
    levels arrives as prints BELOW our price and never decrements anything.

    So ``cumulative=False`` -- our own level only -- is the CORRECT pairing: a
    sweep that reaches us has already cleared the better levels by definition,
    and what stands between us and the fill is the size queued at our own price.
    ``cumulative=True`` charges the whole at-or-better ladder against prints that
    can only ever be a fraction of it, over-penalising by ~19x at 60 bps on this
    tape. It is kept because a result that survives it is robust to anything a
    real queue could do.
    """
    levels = price_matrix[book_idx]
    if side == "ask":
        # Ask prices ascend; a buy sweep clears every level at or below ours.
        index = int(np.searchsorted(levels, float(price), side="right")) - 1
    else:
        # Bid prices descend, so search the negated ladder to keep it ascending.
        index = int(np.searchsorted(-levels, -float(price), side="right")) - 1
    if index < 0:
        return 0.0
    index = min(index, cum_sizes.shape[1] - 1)
    if cumulative:
        value = float(cum_sizes[book_idx, index])
    else:
        previous = float(cum_sizes[book_idx, index - 1]) if index > 0 else 0.0
        value = float(cum_sizes[book_idx, index]) - previous
    return value if math.isfinite(value) and value > 0.0 else 0.0


def is_joining_best(side: str, price: float, best_bid: float, best_ask: float) -> bool:
    if side == "bid":
        return abs(float(price) - float(best_bid)) <= max(1e-9, abs(float(best_bid)) * 1e-9)
    return abs(float(price) - float(best_ask)) <= max(1e-9, abs(float(best_ask)) * 1e-9)


def scan_for_matching_trade(
    side: str,
    price: float,
    positions: Iterable[int] | Sequence[int],
    *,
    trade_ts_ns: np.ndarray | None,
    trade_price: np.ndarray,
    trade_size: np.ndarray,
    trade_size_falsy: np.ndarray,
    used: np.ndarray | None,
    queue_ahead: float,
    queue_decay_per_second: float = 0.0,
    active_at_ns: int | None = None,
) -> tuple[int, float]:
    """Walk a window of trades and return (position of the fill, queue decayed).

    The single implementation of the fill rule.
    :func:`matching_trade_with_queue_decay` is the DataFrame adapter over it and
    ``run_replay`` calls it straight, because the frame the adapter would build
    is the frame this rewrite exists to stop building.

    The arithmetic is deliberately the same statements the row-wise version ran,
    including the ``min``/``max`` calls: ``min(remaining, decay)`` and
    ``max(0.0, size)`` propagate NaN differently from the comparison rewrites
    they invite, and a NaN size is reachable from a real feed.

    ``positions`` gives the window IN EVENT ORDER; ``used`` is the consumed-trade
    mask, applied here so the caller never has to build a filtered copy.
    Returns -1 for no fill.
    """
    remaining_queue = max(0.0, float(queue_ahead))
    initial_queue = remaining_queue
    decay_rate = max(0.0, float(queue_decay_per_second))
    queue_decay_base = 0.0
    last_ns = active_at_ns
    has_ts = trade_ts_ns is not None
    # A side that is neither bid nor ask crosses nothing, as before.
    mode = 0 if side == "bid" else (1 if side == "ask" else 2)
    for position in positions:
        if used is not None and used[position]:
            continue
        trade_ns = int(trade_ts_ns[position]) if has_ts else None
        if last_ns is None and trade_ns is not None:
            last_ns = trade_ns
        if decay_rate > 0 and last_ns is not None and trade_ns is not None and remaining_queue > 0:
            elapsed_seconds = max(0.0, total_seconds_from_ns(trade_ns - last_ns))
            decay = min(remaining_queue, initial_queue * decay_rate * elapsed_seconds)
            remaining_queue -= decay
            queue_decay_base += decay
            last_ns = trade_ns

        candidate_price = trade_price[position]
        if not math.isfinite(candidate_price):
            continue
        if mode == 0:
            crosses = candidate_price <= price
        elif mode == 1:
            crosses = candidate_price >= price
        else:
            crosses = False
        if not crosses:
            continue
        candidate_size = 0.0 if trade_size_falsy[position] else float(trade_size[position])
        if remaining_queue > 0:
            remaining_queue -= max(0.0, candidate_size)
            if remaining_queue >= 0:
                continue
        return int(position), queue_decay_base
    return -1, queue_decay_base


def matching_trade_with_queue_decay(
    side: str,
    price: float,
    window: pd.DataFrame,
    queue_ahead: float,
    *,
    queue_decay_per_second: float = 0.0,
    active_at: pd.Timestamp | None = None,
) -> tuple[pd.Series | None, float]:
    """DataFrame adapter over :func:`scan_for_matching_trade`."""
    trade_price, trade_size, trade_size_falsy = trade_event_arrays(window)
    position, queue_decay_base = scan_for_matching_trade(
        side,
        price,
        range(len(window)),
        trade_ts_ns=timestamps_as_ns(window, coerce=True),
        trade_price=trade_price,
        trade_size=trade_size,
        trade_size_falsy=trade_size_falsy,
        used=None,
        queue_ahead=queue_ahead,
        queue_decay_per_second=queue_decay_per_second,
        active_at_ns=timestamp_as_ns(active_at),
    )
    if position < 0:
        return None, queue_decay_base
    return window.iloc[position], queue_decay_base


def matching_trade(
    side: str,
    price: float,
    window: pd.DataFrame,
    queue_ahead: float,
    *,
    queue_decay_per_second: float = 0.0,
    active_at: pd.Timestamp | None = None,
) -> pd.Series | None:
    trade, _queue_decay_base = matching_trade_with_queue_decay(
        side,
        price,
        window,
        queue_ahead,
        queue_decay_per_second=queue_decay_per_second,
        active_at=active_at,
    )
    return trade


def run_replay(
    config: ReplayConfig,
    params: dict[str, float],
    tape: ReplayTape | None = None,
) -> ReplayMetrics:
    """Score one configuration.

    ``tape=None`` loads the data here, which is what a single run wants and is
    byte-identical to what this function always did. A SWEEP must pass a tape
    from :func:`load_tape` instead: the collector keeps writing, so two runs
    minutes apart otherwise score different data and the comparison is worthless
    (see :class:`ReplayTape`).
    """
    prices, trades, orderbooks, input_files = (
        load_symbol_data(config) if tape is None else tape.frames()
    )
    metrics = ReplayMetrics()
    metrics.input_files = input_files
    metrics.starting_equity_usdc = float(config.starting_equity_usdc)
    metrics.equity_usdc = float(config.starting_equity_usdc)
    metrics.min_equity_usdc = float(config.starting_equity_usdc)
    metrics.min_liquidation_buffer_usdc = float(config.starting_equity_usdc)
    metrics.quote_refresh_interval_ms = int(config.quote_refresh_interval_ms)
    metrics.decision_latency_ms = int(config.decision_latency_ms)
    metrics.order_ack_latency_ms = int(config.order_ack_latency_ms)
    metrics.cancel_latency_ms = int(config.cancel_latency_ms)
    metrics.event_clock = str(config.event_clock)
    metrics.event_clock_by_stream = {
        name: str(frame.attrs.get("event_clock", "unknown"))
        for name, frame in (("prices", prices), ("trades", trades), ("orderbooks", orderbooks))
    }
    metrics.price_tick_size = float(config.price_tick_size)
    metrics.amount_step_size = float(config.amount_step_size)
    metrics.fill_calibration = load_fill_calibration(config.fill_calibration_path)

    if prices.empty:
        return metrics

    prices = prices.reset_index(drop=True)
    trades = trades.reset_index(drop=True)
    orderbooks = orderbooks.reset_index(drop=True)
    if config.max_price_events is not None and config.max_price_events > 0 and len(prices) > config.max_price_events:
        start_ts = prices.iloc[-config.max_price_events]["timestamp"]
        prices = prices[prices["timestamp"] >= start_ts].reset_index(drop=True)
        if not trades.empty and "timestamp" in trades:
            trades = trades[trades["timestamp"] >= start_ts].reset_index(drop=True)
        if not orderbooks.empty and "timestamp" in orderbooks:
            orderbooks = orderbooks[orderbooks["timestamp"] >= start_ts].reset_index(drop=True)
    metrics.input_rows = {
        "prices": int(len(prices)),
        "trades": int(len(trades)),
        "orderbooks": int(len(orderbooks)),
    }
    metrics.data_start = prices.iloc[0]["timestamp"].isoformat()
    metrics.data_end = prices.iloc[-1]["timestamp"].isoformat()
    metrics.price_event_count = int(len(prices))
    metrics.data_span_seconds = max(
        0.0,
        float((prices.iloc[-1]["timestamp"] - prices.iloc[0]["timestamp"]).total_seconds()),
    )
    if metrics.data_span_seconds > 0:
        metrics.price_events_per_day = float(metrics.price_event_count) / (metrics.data_span_seconds / 86_400.0)
    gaps = prices["timestamp"].diff().dt.total_seconds().dropna()
    if not gaps.empty:
        metrics.max_price_gap_seconds = float(gaps.max())
        metrics.p95_price_gap_seconds = float(gaps.quantile(0.95))
        # Same distribution in ms, because that is the unit the latency ladder is
        # read in: a rung shorter than p10 cannot be distinguished from a faster
        # one on this tape.
        metrics.price_gap_ms_p10 = float(gaps.quantile(0.10) * 1000.0)
        metrics.price_gap_ms_p50 = float(gaps.quantile(0.50) * 1000.0)
        metrics.price_gap_ms_p90 = float(gaps.quantile(0.90) * 1000.0)
    snapshot_ts = prices.iloc[0]["timestamp"]
    metrics.parameter_series.append(
        replay_parameter_snapshot(
            config,
            params,
            snapshot_ts,
            data_start=metrics.data_start,
            data_end=metrics.data_end,
        )
    )
    metrics.toxicity_series.append(
        replay_toxicity_snapshot(
            config,
            params,
            snapshot_ts,
            data_start=metrics.data_start,
            data_end=metrics.data_end,
        )
    )

    total_quote_latency_ms = config.decision_latency_ms + config.order_ack_latency_ms
    quote_refresh_interval_ms = max(0, int(config.quote_refresh_interval_ms))
    next_quote_decision_ns: int | None = None
    hjb_cache = solve_replay_hjb(config, params)
    # mm_core.select_delta reads the inventory axis through
    # np.asarray(hjb["q_grid"], dtype=float), and the solvers hand back an int64
    # grid -- so that allocated a fresh array on every side of every quote.
    # Converting once makes the asarray a no-op. It cannot move a value: the
    # grid is small integers, exact in float64, and nothing else reads this
    # dict (it never leaves run_replay).
    if isinstance(hjb_cache, dict) and "q_grid" in hjb_cache:
        hjb_cache["q_grid"] = np.asarray(hjb_cache["q_grid"], dtype=float)
    episodic = str(config.hjb_time_mode) == "episodic"
    horizon = float(config.hjb_horizon_seconds)
    episode_start_ns: int | None = None

    # ---- hoisted columns and loop invariants ------------------------------
    # See the "Column views" block above for why. Nothing here changes what is
    # computed; it changes only how many times per event pandas is asked for it.
    price_ts_ns = timestamps_as_ns(prices)
    if price_ts_ns is None:
        raise TypeError("replay needs a datetime 'timestamp' column on the price stream")
    price_mid_arr, price_best_bid_arr, price_best_ask_arr = price_event_arrays(
        prices, config.mid_fallback
    )
    price_count = len(prices)
    # tolist() once, then index Python lists in the loop. ndarray element access
    # boxes a numpy scalar every time, which then has to be unboxed again by the
    # float()/int() the metrics need; the list holds the identical values as
    # native ints and floats. The arrays stay for searchsorted, which needs them.
    price_ts = price_ts_ns.tolist()
    price_mid_list = price_mid_arr.tolist()
    price_best_bid_list = price_best_bid_arr.tolist()
    price_best_ask_list = price_best_ask_arr.tolist()

    trades_empty = bool(trades.empty)
    trade_ts_ns = timestamps_as_ns(trades)
    if not trades_empty and trade_ts_ns is None:
        raise TypeError("replay needs a datetime 'timestamp' column on the trade stream")
    trade_price_arr, trade_size_arr, trade_size_falsy_arr = trade_event_arrays(trades)
    trade_ts = None if trade_ts_ns is None else trade_ts_ns.tolist()
    trade_price = trade_price_arr.tolist()
    trade_size = trade_size_arr.tolist()
    trade_size_falsy = trade_size_falsy_arr.tolist()
    trade_is_buy, trade_ids = trade_flow_arrays(trades)
    trade_ts_series = trades["timestamp"] if "timestamp" in trades.columns else None
    # A boolean mask instead of the set of consumed row labels, and the window
    # instead of a filtered copy of it: the old pair cost an isin() over a fresh
    # DataFrame per quote, which was a quarter of the whole run.
    trade_used = [False] * len(trades)
    trades_sorted = is_monotonic_ns(trade_ts_ns)

    book_ts_ns = timestamps_as_ns(orderbooks)
    if not orderbooks.empty and "timestamp" in orderbooks.columns and book_ts_ns is None:
        raise TypeError("replay needs a datetime 'timestamp' column on the orderbook stream")
    if book_ts_ns is not None and len(book_ts_ns):
        book_bid_size = first_level_size_array(orderbooks, "bid")
        book_ask_size = first_level_size_array(orderbooks, "ask")
        book_bid_depth = book_depth_arrays(orderbooks, "bid")
        book_ask_depth = book_depth_arrays(orderbooks, "ask")
    else:
        book_ts_ns = None
        book_bid_size = book_ask_size = None
        book_bid_depth = book_ask_depth = None

    inventory_config = _inventory_config(config.inventory_unit_base, config.q_max, config.q_min)
    quote_config = half_spread_quote_config(
        spread_multiplier=config.spread_multiplier,
        maker_fee=config.maker_fee,
        min_half_spread_bps=config.min_half_spread_bps,
        max_half_spread_bps=config.max_half_spread_bps,
    )
    # Both arguments are configuration, so the order size never varies by event.
    order_amount = round_amount_down(config.inventory_unit_base, config.amount_step_size)
    order_amount_ok = order_amount > 0
    calibration_bucket_bps = metrics.fill_calibration.get("bucket_bps")
    calibration_applied = bool(metrics.fill_calibration.get("applied"))
    funding_rate_per_hour = config.funding_rate_per_hour
    queue_decay_per_second = config.queue_decay_per_second
    q_min = int(config.q_min)
    q_max = int(config.q_max)
    long_only = q_min >= 0
    short_floor = float(q_min) * float(config.inventory_unit_base)
    maker_fee = config.maker_fee
    taker_fee = config.taker_fee
    price_tick = usable_tick_size(config.price_tick_size)
    latency_ns = int(total_quote_latency_ms) * _NS_PER_MS
    refresh_ns = quote_refresh_interval_ms * _NS_PER_MS
    cancel_ns = int(config.cancel_latency_ms) * _NS_PER_MS
    episode_min_elapsed = horizon * float(config.episode_min_elapsed_fraction)
    consumed_trade_events = 0

    # ---- toxic-flow guard -------------------------------------------------
    # Sized from this tape's own traded volume, exactly as vpin_bucket_units
    # does live. NOTE a fidelity limit worth remembering when reading per-window
    # scores: a 6 h window never completes 30 buckets of ADV/50, so those runs
    # exercise the fast breaker and the warm-up re-entry rule rather than
    # steady-state VPIN.
    guard_config = FlowGuardConfig(
        enabled=bool(config.flow_guard_enabled),
        fast_move_window_ms=float(config.flow_guard_fast_move_window_ms),
        fast_move_threshold_bps=float(config.flow_guard_fast_move_threshold_bps),
        vpin_buckets_per_day=int(config.flow_guard_vpin_buckets_per_day),
        vpin_window_buckets=int(config.flow_guard_vpin_window_buckets),
        vpin_threshold=float(config.flow_guard_vpin_threshold),
        cooldown_ms=float(config.flow_guard_cooldown_ms),
        warmup_reentry_cooldown_ms=float(config.flow_guard_warmup_reentry_cooldown_ms),
        warmup_reentry_calm_fraction=float(config.flow_guard_warmup_reentry_calm_fraction),
    )
    metrics.flow_guard_enabled = guard_config.enabled
    flow_guard = FlowGuard(guard_config)
    mid_window = MidWindow(guard_config.fast_move_window_ms)
    if trades_empty or trade_ts is None:
        guard_bucket_volume = 1.0
    else:
        guard_bucket_volume = bucket_volume_from_totals(
            float(sum(v for v in trade_size if v > 0.0)),
            (trade_ts[-1] - trade_ts[0]) / _NS_PER_MS,
            guard_config.vpin_buckets_per_day,
        )
    metrics.flow_guard_vpin_bucket_volume = guard_bucket_volume
    vpin_tracker = VpinTracker(guard_bucket_volume, guard_config.vpin_window_buckets)
    # Read-only cursor over the SAME trade arrays the fill simulator uses. It
    # never touches `trade_used`, so feeding the guard cannot consume a fill.
    guard_trade_cursor = 0
    guard_vpin: float | None = None

    flatten_slippage_bps = float(config.flatten_slippage_bps)
    queue_model_uses_book = str(config.queue_model) in {"book_level", "book_depth"}
    queue_model_is_cumulative = str(config.queue_model) == "book_depth"
    flatten_after_ns = (
        None if config.flatten_after_ms is None
        else int(float(config.flatten_after_ms) * _NS_PER_MS)
    )
    # Open exposure, FIFO. Entries are (ts_ns, signed_base): positive is long.
    open_lots: deque[tuple[int, float]] = deque()

    for row_idx in range(price_count):
        row_ts_ns = price_ts[row_idx]
        mid = price_mid_list[row_idx]
        metrics.final_mid = mid

        # inventory_to_q IS round(inventory_to_q_exact(...)) on the same config,
        # so the exact value is computed once and rounded rather than twice.
        q_exact = mm_core.inventory_to_q_exact(metrics.inventory_base, inventory_config)
        q = int(round(q_exact))
        metrics.q_residual_abs_sum += abs(q_exact - q)
        metrics.q_residual_samples += 1

        # Episode clock on SIMULATED time. Rolls at T, or early once flat and
        # far enough in that a single round trip cannot pin it at t=0.
        tau: float | None = None
        if episodic:
            if episode_start_ns is None:
                episode_start_ns = row_ts_ns
                metrics.episodes += 1
            elapsed = max(0.0, total_seconds_from_ns(row_ts_ns - episode_start_ns))
            if elapsed >= horizon or (
                config.episode_reset_on_flat and q == 0 and elapsed >= episode_min_elapsed
            ):
                episode_start_ns = row_ts_ns
                metrics.episodes += 1
                elapsed = 0.0
            tau = max(0.0, horizon - elapsed)

        if q == q_min:
            metrics.time_at_q_boundary["q_min"] += 1
        if q == q_max:
            metrics.time_at_q_boundary["q_max"] += 1
        metrics.inventory_histogram[q] = metrics.inventory_histogram.get(q, 0) + 1

        if funding_rate_per_hour and row_idx > 0:
            elapsed_hours = max(
                0.0, total_seconds_from_ns(row_ts_ns - price_ts[row_idx - 1]) / 3600.0
            )
            funding = -metrics.inventory_base * mid * float(funding_rate_per_hour) * elapsed_hours
            metrics.funding_usdc += funding
            metrics.cash_usdc += funding

        # Feed the guard on every price event, before the refresh gate: the
        # breaker window and VPIN are properties of the market, not of our
        # quoting cadence, so they must see the whole stream.
        if guard_config.enabled:
            if not trades_empty and trade_ts is not None:
                while guard_trade_cursor < len(trade_ts) and trade_ts[guard_trade_cursor] <= row_ts_ns:
                    if not trade_size_falsy[guard_trade_cursor]:
                        guard_vpin = vpin_tracker.observe(
                            trade_is_buy[guard_trade_cursor],
                            trade_size[guard_trade_cursor],
                            trade_ids[guard_trade_cursor],
                        )
                    guard_trade_cursor += 1
            guard_move_bps = mid_window.observe(row_ts_ns / _NS_PER_MS, mid)
            guard_tripped = flow_guard.evaluate(row_ts_ns / _NS_PER_MS, guard_move_bps, guard_vpin)
        else:
            guard_tripped = False

        # Aggressive flatten, before the refresh gate: the exit deadline belongs
        # to the position, not to the quoting cadence, so it must be checked on
        # every event rather than once per refresh.
        if flatten_after_ns is not None and open_lots:
            oldest_ts, oldest_size = open_lots[0]
            # The deadline carries the SAME latency the quoting path pays: we
            # notice the position, decide, and the taker order reaches the venue
            # one round trip later. Executing at the deadline price instead
            # would be reading a book we could not have traded on.
            if row_ts_ns - oldest_ts >= flatten_after_ns + latency_ns:
                touch_bid = price_best_bid_list[row_idx]
                touch_ask = price_best_ask_list[row_idx]
                # Cross to go flat: sell a long into the bid, buy a short from
                # the ask. Without a touch there is nothing to cross into, so
                # the lot simply waits -- refusing to invent a price is the
                # difference between measuring this policy and flattering it.
                exit_price = touch_bid if oldest_size > 0.0 else touch_ask
                if exit_price is not None and math.isfinite(exit_price) and exit_price > 0.0:
                    # Walk the book: a sell lands below the bid, a buy above the ask.
                    slip = exit_price * (flatten_slippage_bps / 10_000.0)
                    exit_price = exit_price - slip if oldest_size > 0.0 else exit_price + slip
                    size = abs(oldest_size)
                    notional = size * exit_price
                    fee = notional * taker_fee
                    if oldest_size > 0.0:
                        metrics.inventory_base -= size
                        metrics.cash_usdc += notional - fee
                    else:
                        metrics.inventory_base += size
                        metrics.cash_usdc -= notional + fee
                    metrics.taker_fills += 1
                    metrics.fees_usdc += fee
                    metrics.flatten_events += 1
                    metrics.flatten_base += size
                    # What the truncation cost: the half-spread crossed plus the
                    # taker fee. Recorded so the two legs stay separable.
                    metrics.flatten_cost_usdc += abs(mid - exit_price) * size + fee
                    match_holding_time(
                        open_lots, row_ts_ns, -oldest_size, metrics
                    )
                    # Inventory just moved, and `q` was computed at the top of
                    # this row. Leaving it stale makes compute_quotes price the
                    # UNWINDING branch -- the 1.5 bps floor on the side that
                    # would flatten a position we no longer hold -- and that
                    # quote fills, opening a fresh position 1.5 bps from mid
                    # that is crossed out one deadline later. Measured on a 34 h
                    # slice: 40 of 164 non-flat decisions were stale, a real
                    # P&L drag rather than a source of the result, but wrong.
                    q_exact = mm_core.inventory_to_q_exact(
                        metrics.inventory_base, inventory_config
                    )
                    q = int(round(q_exact))

        if next_quote_decision_ns is not None and row_ts_ns < next_quote_decision_ns:
            update_margin_metrics(metrics, config, mid)
            continue

        # Only a quote decision reads the touch, so it is not unpacked above the
        # early continue -- most events on a fast tape never get here.
        best_bid = price_best_bid_list[row_idx]
        best_ask = price_best_ask_list[row_idx]
        metrics.quote_decision_events += 1
        bid, ask, _ = compute_quotes(
            mid,
            q,
            params,
            q_max,
            hjb_cache,
            maker_fee=maker_fee,
            spread_multiplier=config.spread_multiplier,
            min_half_spread_bps=config.min_half_spread_bps,
            max_half_spread_bps=config.max_half_spread_bps,
            tau_remaining=tau,
            q_exact=q_exact,
            quote_config=quote_config,
        )
        if guard_tripped:
            # The analogue of the live path publishing
            # DesiredQuotes::empty(QuoteReason::ToxicFlow, ..): both sides go
            # dark, which also cancels anything resting.
            bid = ask = None
            metrics.flow_guard_withheld_decisions += 1
        inventory_at_decision = metrics.inventory_base
        active_at_ns = row_ts_ns + latency_ns
        refresh_due_ns = row_ts_ns + refresh_ns
        stale_at_ns = max(active_at_ns, refresh_due_ns) + cancel_ns
        next_quote_decision_ns = refresh_due_ns
        # The window the old code cut with a full-frame boolean mask and a take,
        # once per side. On sorted timestamps -- which is every tape the loader
        # produces -- it is a contiguous range, so two binary searches give the
        # same rows without materialising anything. The mask is kept for the
        # unsorted case a monkeypatched loader can still hand over, where a
        # range would be wrong rather than merely slower.
        window_positions: Any = ()
        if not trades_empty:
            if trades_sorted:
                window_positions = range(
                    int(trade_ts_ns.searchsorted(active_at_ns, side="left")),
                    int(trade_ts_ns.searchsorted(stale_at_ns, side="right")),
                )
            else:
                window_positions = np.nonzero(
                    (trade_ts_ns >= active_at_ns) & (trade_ts_ns <= stale_at_ns)
                )[0]

        for side, raw_price in (("bid", bid), ("ask", ask)):
            if raw_price is None:
                continue
            # A long-only agent cannot sell what it does not hold. A two-sided
            # agent can, down to its short bound -- the HJB already returns an
            # infinite depth at that bound, so no extra gate is needed there.
            if long_only and side == "ask" and inventory_at_decision <= 0:
                continue
            price = (
                float(raw_price)
                if price_tick is None
                else round_price_to_tick(side, raw_price, price_tick)
            )
            if abs(float(price) - float(raw_price)) > 1e-12:
                metrics.price_rounding_adjustments += 1
            if not order_amount_ok:
                metrics.amount_rounding_rejects += 1
                continue
            metrics.quote_attempts += 1
            depth_key = quote_depth_key(side, mid, price)
            calibration_key = (
                quote_depth_bucket_key(side, mid, price, calibration_bucket_bps)
                if calibration_bucket_bps is not None
                else depth_key
            )
            if calibration_applied:
                cal_key, _cal_probability = calibration_probability_key(
                    metrics.fill_calibration, side, calibration_key
                )
                metrics.calibration_attempts_by_key[cal_key] = metrics.calibration_attempts_by_key.get(cal_key, 0) + 1
            metrics.quote_attempts_by_depth[depth_key] = metrics.quote_attempts_by_depth.get(depth_key, 0) + 1
            ok, _reason = post_only_check(side, price, best_bid, best_ask)
            if not ok:
                metrics.post_only_rejects += 1
                continue

            if trades_empty:
                metrics.stale_quote_cancels += 1
                continue

            queue_ahead = 0.0
            if book_ts_ns is not None:
                book_idx = int(book_ts_ns.searchsorted(active_at_ns, side="right")) - 1
                if book_idx >= 0:
                    depth = book_bid_depth if side == "bid" else book_ask_depth
                    if queue_model_uses_book and depth is not None:
                        # Everything resting at our level or better. Uses the
                        # most recent snapshot, which is up to ~5.4 s old: queue
                        # DEPTH is slowly varying, unlike a directional signal,
                        # so a stale snapshot is a far weaker assumption here
                        # than pretending the queue is empty.
                        queue_ahead = queue_ahead_from_book(
                            depth[0], depth[1], book_idx, side, price,
                            cumulative=queue_model_is_cumulative,
                        )
                    elif is_joining_best(side, price, best_bid, best_ask):
                        sizes = book_bid_size if side == "bid" else book_ask_size
                        queue_ahead = float(sizes[book_idx])
            fill_index, queue_decay_base = scan_for_matching_trade(
                side,
                price,
                window_positions,
                trade_ts_ns=trade_ts,
                trade_price=trade_price,
                trade_size=trade_size,
                trade_size_falsy=trade_size_falsy,
                used=trade_used,
                queue_ahead=queue_ahead,
                queue_decay_per_second=queue_decay_per_second,
                active_at_ns=active_at_ns,
            )
            metrics.queue_decay_base += queue_decay_base

            if fill_index < 0:
                metrics.stale_quote_cancels += 1
                continue

            fill_trade_size = (
                order_amount if trade_size_falsy[fill_index] else trade_size[fill_index]
            )
            if side == "ask":
                # Selling is bounded by how much further the short bound allows,
                # which is the whole inventory for a long-only agent (q_min=0)
                # and inventory-minus-the-floor for a two-sided one.
                room_to_sell = max(0.0, metrics.inventory_base - short_floor)
                fill_size = min(fill_trade_size, order_amount, room_to_sell)
            else:
                fill_size = min(fill_trade_size, order_amount)
            if fill_size <= 0:
                continue

            calibrated_allowed, _calibrated_key = calibrated_fill_allowed(
                metrics.fill_calibration,
                metrics.calibration_attempts_by_key,
                metrics.calibration_fills_by_key,
                side=side,
                depth_key=calibration_key,
            )
            trade_used[fill_index] = True
            consumed_trade_events += 1
            metrics.consumed_trade_events = consumed_trade_events
            if not calibrated_allowed:
                metrics.calibration_rejected_fills += 1
                metrics.stale_quote_cancels += 1
                continue
            notional = fill_size * price
            fee = notional * maker_fee
            gross_spread = abs(mid - price) * fill_size
            side_pnl = gross_spread - fee
            metrics.maker_fills += 1
            metrics.fees_usdc += fee
            if side == "bid":
                metrics.inventory_base += fill_size
                metrics.cash_usdc -= notional + fee
                metrics.pnl_by_side["bid"] += side_pnl
            else:
                metrics.inventory_base -= fill_size
                metrics.cash_usdc += notional - fee
                metrics.pnl_by_side["ask"] += side_pnl
            metrics.realized_spread_usdc += gross_spread
            match_holding_time(
                open_lots,
                trade_ts[fill_index],
                fill_size if side == "bid" else -fill_size,
                metrics,
            )
            metrics.fills_by_depth[depth_key] = metrics.fills_by_depth.get(depth_key, 0) + 1
            metrics.fills_by_side[side] = metrics.fills_by_side.get(side, 0) + 1
            metrics.fill_depth_bps_sum_by_side[side] = metrics.fill_depth_bps_sum_by_side.get(
                side, 0.0
            ) + quote_depth_bps(mid, price)
            fill_ts = trade_ts_series.iloc[fill_index]
            fill_ts_ns = trade_ts[fill_index]
            for horizon_ms in MARKOUT_HORIZONS_MS:
                future = future_mid_from_arrays(price_ts_ns, price_mid_arr, fill_ts_ns, horizon_ms)
                if future is None:
                    continue
                markout = markout_value(side, price, future)
                # Aggregated per side as well as sampled, because the sample list
                # is only usable by a reader who re-groups it, and the question a
                # sweep asks -- which side is paying, and does it stay paid at
                # 30 s -- has to be answerable straight from to_dict().
                side_markouts = metrics.markout_usdc_by_side[side]
                side_counts = metrics.markout_fills_by_side[side]
                side_markouts[horizon_ms] = side_markouts.get(horizon_ms, 0.0) + markout * fill_size
                side_counts[horizon_ms] = side_counts.get(horizon_ms, 0) + 1
                metrics.markout_samples.append({
                    "fill_ts": fill_ts.isoformat(),
                    "side": side,
                    "horizon_ms": horizon_ms,
                    "fill_price": price,
                    "future_mid": future,
                    "markout_usdc_per_base": markout,
                    "markout_usdc": markout * fill_size,
                })

        update_margin_metrics(metrics, config, mid)

    metrics.holding_time_base_unmatched = float(sum(abs(size) for _, size in open_lots))
    metrics.flow_guard_trips = flow_guard.trips
    metrics.flow_guard_vpin_buckets_seen = vpin_tracker.buckets_seen()
    if metrics.final_mid is not None:
        metrics.mark_to_market_pnl_usdc = metrics.cash_usdc + metrics.inventory_base * metrics.final_mid

    return metrics


def synthetic_symbol_data(config: ReplayConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    ts0 = pd.Timestamp("2026-05-25T10:00:00Z")
    prices = pd.DataFrame(
        [
            {"timestamp": ts0, "bid": config.mid_fallback - 0.5, "ask": config.mid_fallback + 0.5},
            {
                "timestamp": ts0 + pd.Timedelta(seconds=1),
                "bid": config.mid_fallback - 0.4,
                "ask": config.mid_fallback + 0.6,
            },
            {
                "timestamp": ts0 + pd.Timedelta(seconds=2),
                "bid": config.mid_fallback - 0.3,
                "ask": config.mid_fallback + 0.7,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "timestamp": ts0 + pd.Timedelta(milliseconds=600),
                "price": config.mid_fallback - 3.0,
                "size": config.inventory_unit_base,
            }
        ]
    )
    return prices, trades, pd.DataFrame(), {"prices": 0, "trades": 0, "orderbooks": 0}


def synthetic_tape(config: ReplayConfig) -> ReplayTape:
    """The built-in smoke data as a tape, so --synthetic-smoke goes through the
    same entry point as a real run instead of monkeypatching the module global."""
    prices, trades, orderbooks, input_files = synthetic_symbol_data(config)
    for frame in (prices, trades, orderbooks):
        # Synthetic frames carry no exchange_timestamp, so the metrics should say
        # so rather than claiming a clock the smoke data does not have.
        _, clock = apply_event_clock(frame, config.event_clock)
        frame.attrs["event_clock"] = clock
    return ReplayTape(prices=prices, trades=trades, orderbooks=orderbooks, input_files=input_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay HJB market-making quotes over collected Hyperliquid data.")
    parser.add_argument("--symbol", default="CASHCAT")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "HL_data")
    parser.add_argument("--mid", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run against built-in synthetic BBO/trade data.")
    parser.add_argument("--kappa-plus", type=float, required=True)
    parser.add_argument("--kappa-minus", type=float, required=True)
    parser.add_argument("--lambda-plus", type=float, required=True)
    parser.add_argument("--lambda-minus", type=float, required=True)
    parser.add_argument("--epsilon-plus", type=float, default=0.0)
    parser.add_argument("--epsilon-minus", type=float, default=0.0)
    parser.add_argument("--maker-fee", type=float, default=MAKER_FEE)
    parser.add_argument("--taker-fee", type=float, default=TAKER_FEE)
    parser.add_argument(
        "--spread-multiplier",
        type=float,
        default=1.0,
        help="Scales the HJB model depth only (fee cushion added separately); pass 3.0 to mirror the production config.",
    )
    parser.add_argument("--min-half-spread-bps", type=float, default=MIN_HALF_SPREAD_BPS)
    parser.add_argument("--max-half-spread-bps", type=float, default=MAX_HALF_SPREAD_BPS)
    parser.add_argument("--funding-rate-per-hour", type=float, default=0.0)
    parser.add_argument("--starting-equity-usdc", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-rate", type=float, default=0.05)
    parser.add_argument(
        "--queue-decay-per-second",
        type=float,
        default=0.05,
        help="Conservative queue-ahead cancellation decay as a fraction of initial queue per second.",
    )
    parser.add_argument("--price-tick-size", type=float, default=0.0)
    parser.add_argument("--amount-step-size", type=float, default=0.0)
    parser.add_argument("--newest-per-stream", type=int, default=None)
    parser.add_argument("--max-price-events", type=int, default=None)
    parser.add_argument(
        "--hjb-time-mode",
        choices=sorted(mm_core.HJB_TIME_MODES),
        default="episodic",
        help=(
            "episodic runs the book's delta*(t,q) on a simulated episode clock; "
            "stationary reads only the t=0 slice."
        ),
    )
    parser.add_argument("--hjb-horizon-seconds", type=float, default=60.0)
    parser.add_argument(
        "--hjb-alpha",
        type=float,
        default=0.001,
        help="RAW terminal penalty. Only used when --hjb-alpha-kappa is 0; not kappa-invariant.",
    )
    parser.add_argument(
        "--hjb-phi",
        type=float,
        default=0.0001,
        help="RAW running penalty. Only used when --hjb-phi-kappa-t is 0; not kappa-invariant.",
    )
    parser.add_argument(
        "--hjb-phi-kappa-t",
        type=float,
        default=0.0,
        help=(
            "Dimensionless running penalty phi*kappa*T, the sweepable one (live runs 10.0). "
            "0 disables the derivation and falls back to the raw --hjb-phi."
        ),
    )
    parser.add_argument(
        "--hjb-alpha-kappa",
        type=float,
        default=0.0,
        help=(
            "Dimensionless terminal penalty alpha*kappa (live runs 0.05). "
            "0 disables the derivation and falls back to the raw --hjb-alpha."
        ),
    )
    parser.add_argument(
        "--hjb-phi-kappa-t-max",
        type=float,
        default=QuoteConfig.hjb_phi_kappa_t_max,
        help="Ceiling on phi*kappa*T, as live. Sweeping --hjb-phi-kappa-t past it changes nothing.",
    )
    parser.add_argument(
        "--sigma2-per-sec",
        type=float,
        default=None,
        help="Variance per second for mm_core's volatility channel; omitted leaves it quiet.",
    )
    parser.add_argument("--gamma-inventory-risk", type=float, default=QuoteConfig.gamma_inventory_risk)
    parser.add_argument(
        "--q-max",
        type=int,
        default=3,
        help="Inventory clamp, in inventory units.",
    )
    parser.add_argument(
        "--q-min",
        type=int,
        default=0,
        help=(
            "Lower inventory clamp. 0 (the default) is the long-only agent this harness shipped "
            "with; pass -q_max to simulate the two-sided market maker the live system runs."
        ),
    )
    parser.add_argument(
        "--hjb-q-min",
        type=int,
        default=None,
        help=(
            "Inventory domain the HJB is SOLVED on, separate from the clamp above. "
            "Omit for the symmetric [-q_max, q_max] solve; pass 0 to solve the long-only problem."
        ),
    )
    parser.add_argument("--inventory-unit-base", type=float, default=0.01)
    parser.add_argument(
        "--decision-latency-ms",
        type=int,
        default=None,
        help="Overrides the decision half of --latency-ms (default 250).",
    )
    parser.add_argument(
        "--order-ack-latency-ms",
        type=int,
        default=None,
        help="Overrides the ack half of --latency-ms (default 250).",
    )
    parser.add_argument(
        "--cancel-latency-ms",
        type=int,
        default=250,
        help="How long a quote keeps resting after it should have been pulled.",
    )
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=None,
        help=(
            "TOTAL simulated order latency, split evenly across decision and ack (odd values "
            "give the extra millisecond to decision). Scenario labels such as 50 and 500 ms "
            "are hypotheses, not measurements of the current host. At a fixed 1 s refresh, "
            "the refresh interval sets most of the staleness."
        ),
    )
    parser.add_argument(
        "--event-clock",
        choices=("exchange", "local"),
        default="exchange",
        help=(
            "Which collector clock orders events. 'exchange' is correct; 'local' is the "
            "receive clock this harness used before 2026-08-17 and carries ~200 ms of jitter, "
            "which is more than a latency rung."
        ),
    )
    parser.add_argument(
        "--no-episode-reset-on-flat",
        dest="episode_reset_on_flat",
        action="store_false",
        help="Let every episode run the full horizon instead of restarting once flat.",
    )
    parser.set_defaults(episode_reset_on_flat=True)
    parser.add_argument(
        "--quote-refresh-interval-ms",
        type=int,
        default=1000,
        help="Minimum cadence between simulated quote decisions.",
    )
    parser.add_argument(
        "--fill-calibration",
        type=Path,
        default=None,
        help="Optional replay_log_calibration.json artifact used to conservatively throttle simulated fills.",
    )
    return parser.parse_args()


def resolve_latency_ms(args: argparse.Namespace) -> tuple[int, int]:
    """(decision, ack) from --latency-ms plus any explicit override.

    A sweep varies ONE number, so --latency-ms is the total and the split is an
    implementation detail: half each, with the odd millisecond going to decision.
    The individual flags still win, because the two halves are not physically the
    same thing (think time vs round trip) and a stack can improve one alone.
    """
    if args.latency_ms is None:
        decision, ack = 250, 250
    else:
        total = max(0, int(args.latency_ms))
        ack = total // 2
        decision = total - ack
    if args.decision_latency_ms is not None:
        decision = int(args.decision_latency_ms)
    if args.order_ack_latency_ms is not None:
        ack = int(args.order_ack_latency_ms)
    return decision, ack


def main() -> int:
    args = parse_args()
    params = {
        "kappa+": args.kappa_plus,
        "kappa-": args.kappa_minus,
        "lambda+": args.lambda_plus,
        "lambda-": args.lambda_minus,
        "epsilon+": args.epsilon_plus,
        "epsilon-": args.epsilon_minus,
    }
    decision_latency_ms, order_ack_latency_ms = resolve_latency_ms(args)
    config = ReplayConfig(
        symbol=args.symbol,
        data_dir=args.data_dir,
        mid_fallback=args.mid,
        inventory_unit_base=args.inventory_unit_base,
        q_max=args.q_max,
        q_min=args.q_min,
        hjb_q_min=args.hjb_q_min,
        decision_latency_ms=decision_latency_ms,
        order_ack_latency_ms=order_ack_latency_ms,
        cancel_latency_ms=args.cancel_latency_ms,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        spread_multiplier=args.spread_multiplier,
        min_half_spread_bps=args.min_half_spread_bps,
        max_half_spread_bps=args.max_half_spread_bps,
        funding_rate_per_hour=args.funding_rate_per_hour,
        starting_equity_usdc=args.starting_equity_usdc,
        leverage=args.leverage,
        maintenance_margin_rate=args.maintenance_margin_rate,
        queue_decay_per_second=args.queue_decay_per_second,
        price_tick_size=args.price_tick_size,
        amount_step_size=args.amount_step_size,
        quote_refresh_interval_ms=args.quote_refresh_interval_ms,
        fill_calibration_path=args.fill_calibration,
        newest_per_stream=args.newest_per_stream,
        max_price_events=args.max_price_events,
        hjb_time_mode=args.hjb_time_mode,
        hjb_horizon_seconds=args.hjb_horizon_seconds,
        hjb_alpha=args.hjb_alpha,
        hjb_phi=args.hjb_phi,
        hjb_phi_kappa_t=args.hjb_phi_kappa_t,
        hjb_alpha_kappa=args.hjb_alpha_kappa,
        hjb_phi_kappa_t_max=args.hjb_phi_kappa_t_max,
        gamma_inventory_risk=args.gamma_inventory_risk,
        sigma2_per_sec=args.sigma2_per_sec,
        episode_reset_on_flat=args.episode_reset_on_flat,
        event_clock=args.event_clock,
    )
    metrics = run_replay(config, params, tape=synthetic_tape(config) if args.synthetic_smoke else None)
    payload = metrics.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
