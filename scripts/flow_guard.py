#!/usr/bin/env python3
"""Incremental toxic-flow guard, a line-for-line port of the shipped Rust one.

The live bot and the Rust replay both run this guard; the Python staged replay
did not, which biased every sweep it fed. The 08-22 cascade window dominates the
sweep's train slice, so Stage A/B selection was choosing a high inventory
penalty to avoid a cascade the guard already bounds — tuning one lever to solve
a problem another lever owns.

Reference: ``rust_live/crates/mm-runtime/src/flow_guard.rs``. Defaults mirror
``rust_live/crates/mm-settings/src/lib.rs``. Where the two could drift, this
module follows the Rust exactly and says so in a comment.

``scripts/guard_study/tape.py`` holds the *batch* versions of the same two
statistics (``fast_move_bps``, ``vpin_series``) used for offline analysis. They
are vectorised over a whole dataframe and cannot be fed one event at a time,
which is what a replay loop needs, so the bucket-filling and dedup logic is
ported here rather than imported.

Units: this module is float/base-unit throughout (the replay's coordinate),
whereas Rust works in integer venue units. The arithmetic is otherwise
identical; bucket comparisons carry an epsilon because of it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# Matches RECENT_TRADE_ID_CAPACITY in flow_guard.rs.
RECENT_TRADE_ID_CAPACITY = 65_536

# Trade-id strings that carry no identity. A falsy id must NOT collapse
# unrelated anonymous trades into one, which is the Rust `trade_id != 0` gate.
_UNUSABLE_TRADE_IDS = frozenset({"", "0", "None", "nan", "NaT", "<NA>", "none", "null"})

_EPSILON = 1e-12


@dataclass
class FlowGuardConfig:
    """Mirror of the Rust ``FlowGuardConfig``; defaults are the shipped ones."""

    enabled: bool = True
    fast_move_window_ms: float = 5_000.0
    fast_move_threshold_bps: float = 800.0
    vpin_buckets_per_day: int = 50
    vpin_window_buckets: int = 30
    vpin_threshold: float = 0.40
    cooldown_ms: float = 900_000.0
    # Warm-up re-entry. VPIN reads None until `vpin_window_buckets` buckets
    # exist (~14 h of volume at the defaults), and without this rule a trip
    # inside that window withheld every quote for the rest of it.
    warmup_reentry_cooldown_ms: float = 1_800_000.0
    warmup_reentry_calm_fraction: float = 0.5


class MidWindow:
    """Trailing mid move in bps, against the OLDEST sample still in the window.

    Port of ``MidWindow``. Eviction keeps at least one sample, so the reference
    is always defined once anything has been observed.
    """

    def __init__(self, window_ms: float) -> None:
        self._window_ms = float(window_ms)
        self._samples: deque[tuple[float, float]] = deque()

    def observe(self, ts_ms: float, mid: float) -> float:
        self._samples.append((float(ts_ms), float(mid)))
        return self.advance(ts_ms, mid)

    def advance(self, ts_ms: float, mid: float) -> float:
        cutoff = float(ts_ms) - self._window_ms
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()
        if not self._samples:
            return 0.0
        reference = self._samples[0][1]
        if reference <= 0.0 or mid <= 0.0:
            return 0.0
        return abs(float(mid) - reference) / reference * 10_000.0


class VpinTracker:
    """Volume-bucketed order-flow imbalance. Port of ``VpinTracker``.

    ``value()`` returns None until a full window of buckets exists, which the
    guard treats as neither safe nor toxic.
    """

    def __init__(self, bucket_volume: float, window_buckets: int) -> None:
        self._bucket_volume = max(float(bucket_volume), _EPSILON)
        self._window_buckets = max(int(window_buckets), 1)
        self._buy = 0.0
        self._sell = 0.0
        self._imbalances: deque[float] = deque()
        self._imbalance_sum = 0.0
        self._recent_ids: set[str] = set()
        self._recent_order: deque[str] = deque()

    def resize_bucket(self, bucket_volume: float) -> None:
        """Both partial and completed buckets were measured against the old
        denominator, so retaining either would mix scales. Warm up again."""
        nxt = max(float(bucket_volume), _EPSILON)
        if nxt == self._bucket_volume:
            return
        self._bucket_volume = nxt
        self._buy = self._sell = 0.0
        self._imbalances.clear()
        self._imbalance_sum = 0.0

    def observe(self, side_is_buy: bool, size: float, trade_id: str | None = None) -> float | None:
        if trade_id is not None:
            key = str(trade_id)
            if key not in _UNUSABLE_TRADE_IDS and not self._remember(key):
                return self.value()
        remaining = max(float(size), 0.0)
        if remaining == 0.0:
            return self.value()

        # Finish the current partial bucket first, in strict arrival order.
        current = self._buy + self._sell
        if current > 0.0:
            taken = min(remaining, self._bucket_volume - current)
            self._add(side_is_buy, taken)
            remaining -= taken
            if self._buy + self._sell >= self._bucket_volume - _EPSILON:
                self._finish_bucket()

        # Whole buckets out of the remainder are entirely one-sided.
        if remaining >= self._bucket_volume:
            full = int(remaining // self._bucket_volume)
            self._push_one_sided(full)
            remaining -= full * self._bucket_volume

        self._add(side_is_buy, remaining)
        if self._buy + self._sell >= self._bucket_volume - _EPSILON:
            self._finish_bucket()
        return self.value()

    def value(self) -> float | None:
        if len(self._imbalances) < self._window_buckets:
            return None
        denominator = self._window_buckets * self._bucket_volume
        if denominator <= 0.0:
            return None
        return self._imbalance_sum / denominator

    def buckets_seen(self) -> int:
        return len(self._imbalances)

    # -- internals -----------------------------------------------------------
    def _remember(self, trade_id: str) -> bool:
        if trade_id in self._recent_ids:
            return False
        self._recent_ids.add(trade_id)
        self._recent_order.append(trade_id)
        while len(self._recent_order) > RECENT_TRADE_ID_CAPACITY:
            self._recent_ids.discard(self._recent_order.popleft())
        return True

    def _add(self, side_is_buy: bool, qty: float) -> None:
        if side_is_buy:
            self._buy += qty
        else:
            self._sell += qty

    def _finish_bucket(self) -> None:
        self._push_bucket(abs(self._buy - self._sell))
        self._buy = self._sell = 0.0

    def _push_one_sided(self, count: int) -> None:
        # Collapse a pathological huge print to at most the retained window
        # rather than looping qty / V times.
        if count >= self._window_buckets:
            self._imbalances.clear()
            self._imbalances.extend([self._bucket_volume] * self._window_buckets)
            self._imbalance_sum = self._bucket_volume * self._window_buckets
            return
        for _ in range(count):
            self._push_bucket(self._bucket_volume)

    def _push_bucket(self, imbalance: float) -> None:
        self._imbalances.append(imbalance)
        self._imbalance_sum += imbalance
        while len(self._imbalances) > self._window_buckets:
            self._imbalance_sum -= self._imbalances.popleft()


class FlowGuard:
    """Trip/cooldown state machine. Port of ``FlowGuard::evaluate``."""

    def __init__(self, config: FlowGuardConfig | None = None) -> None:
        self.config = config or FlowGuardConfig()
        self._tripped_at_ms: float | None = None
        self._cause: str | None = None
        self.trips = 0

    def evaluate(self, now_ms: float, move_bps: float, vpin: float | None) -> bool:
        """True means withhold quotes."""
        if not self.config.enabled:
            return False
        if move_bps >= self.config.fast_move_threshold_bps:
            self._trip(now_ms, "fast_move")
        elif vpin is not None and vpin >= self.config.vpin_threshold:
            self._trip(now_ms, "vpin")

        if self._tripped_at_ms is None:
            return False
        elapsed_ms = max(0.0, float(now_ms) - self._tripped_at_ms)
        if vpin is not None:
            reopen = elapsed_ms >= self.config.cooldown_ms and vpin < self.config.vpin_threshold
        else:
            # Not a bare timer: the breaker's own trailing window must be calm
            # at a stricter line than the trip line, over a longer cooldown.
            reopen = (
                elapsed_ms >= self.config.warmup_reentry_cooldown_ms
                and move_bps
                < self.config.fast_move_threshold_bps * self.config.warmup_reentry_calm_fraction
            )
        if reopen:
            self._tripped_at_ms = None
            self._cause = None
            return False
        return True

    @property
    def is_tripped(self) -> bool:
        return self._tripped_at_ms is not None

    @property
    def cause(self) -> str | None:
        return self._cause

    def _trip(self, now_ms: float, cause: str) -> None:
        if self._tripped_at_ms is None:
            self.trips += 1
        # Re-arm on every fresh breach so a cascade cannot expire its own
        # cooldown while still running.
        self._tripped_at_ms = float(now_ms)
        self._cause = cause


def bucket_volume_from_totals(total_size: float, span_ms: float, buckets_per_day: int) -> float:
    """Average daily volume / buckets_per_day.

    Equivalent of ``vpin_bucket_units`` in rust_live/src/main.rs: the bucket is
    sized from the tape's own traded volume so the threshold keeps its meaning
    as the instrument's activity changes.
    """
    span_days = max(float(span_ms) / 86_400_000.0, 1e-9)
    total = max(float(total_size), 0.0)
    if total <= 0.0:
        return 1.0
    return max(total / span_days / max(int(buckets_per_day), 1), _EPSILON)
