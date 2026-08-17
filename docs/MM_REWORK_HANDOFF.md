# Market-making rework — status and handoff

**Date:** 2026-08-17 · **Branch:** `main` · **Tests:** 522 passing · **Tree:** clean

> **Superseded — historical record.** This is a point-in-time snapshot of one
> proposed direction (a standalone quoting engine, phases 1–4). That is *not*
> what was built. The shipped design is two Freqtrade instances on separate
> Hyperliquid sub-accounts sharing net inventory over Redis — see `README.md`
> and `docs/PLAN_IMPLEMENTATION_STATUS.md` for current state. Its diagnosis of
> *why* the rework was needed (ETH's one-tick book, the missing fee term, the
> absent viability gate) is still accurate and is why
> `scripts/verify_market_viability.py` exists. Its prescriptions are not
> current. Test count and gate evidence in this file are stale by design.

---

## Why this happened

A review of the project found the deepest problem was not a bug.

**Hyperliquid ETH perp is a one-tick-wide book.** Its touch sits **0.27 bps from
mid** against a **1.5 bps per-side maker fee**, so a passive round trip earns
~0.53 bps and pays ~3.00 bps — **−2.47 bps, structurally, whatever the model
does**. The evidence had been there all along and nothing was reading it: every
accepted quote in `mm_debug.jsonl` was floor-clamped with
`quote_outside_calibrated_range: true` (quoting ~11× past the depth 95% of market
orders ever reach), and `docs/replay_acceptance_report.json` recorded **no maker
fills in any of six variants**.

The README's profitability rule (`κ × ε < 1`) contains **no fee term**, so it
scored this market as "low toxicity, potentially profitable" for months. That
rule is why the project ran so long on a dead instrument.

κ is also not identifiable there: on a one-tick book a touch-taking market order
mechanically has depth = half the spread, so the "distribution" is one spike at
half a tick. Two runs minutes apart gave κ+ = 5.73 vs κ− = 18.32.

Three real defects were also confirmed against freqtrade 2025.10:

- **The kill switch cannot cancel, and strands inventory.** It probes
  `cancel_all_orders` / `fetch_open_orders` / `get_open_orders` — freqtrade's
  `Exchange` has *none* of them. Every kill-switch test passed only because the
  test fixture invented `cancel_all_orders`. Worse, the kill switch sets
  `trading_enabled = False`, which suppresses the exit signal, so an open
  position cannot be unwound by the bot at all.
- **87-minute parameter stall** (2026-08-16) with the collector healthy, and no
  telemetry to explain it.
- **Torn parquet shards silently dropped**, biasing `n_trades` and λ down.

## What was decided

**Full rework** (user, 2026-08-17): a standalone two-sided engine, retiring
freqtrade as the execution path — its one-order-per-pair model makes simultaneous
two-sided quoting impossible. Then back up in dry-run.

**Phase 0 is a hard gate:** no engine work until an instrument is proven able to
pay. Building a better quoter for a market that cannot pay is the exact waste
this whole episode was.

---

## What is done

| Phase | Status | Result |
|---|---|---|
| **0 — viability gate** | ✅ built, ⏳ verdict pending | `scripts/verify_market_viability.py` |
| **1 — carry-over defects** | ✅ | atomic writes, bounded cycles, stall telemetry |
| **2a — shared core** | ✅ | `scripts/mm_core.py`, replay delegates to it |
| **2b — engine** | ⛔ blocked on Phase 0 | — |
| **2c — retire freqtrade** | ⛔ blocked on 2b | — |
| **3 — replay acceptance** | 🟡 mechanics done, run pending | `--two-sided` implemented |
| **4 — dry-run bring-up** | ⛔ blocked | README/docs partly done |

**Phase 0 — viability gate.** Decides from an *empirical profit curve*, not the
fitted model, so a degenerate κ cannot hide the answer:
`edge(δ) = δ − fee·mid − E[markout | depth ≥ δ]`, weighted by traded volume,
maximised over every observed depth, **summing the losing side too**. Three
false-positive routes were found and fixed while testing it against live data:
it cherry-picked the profitable side, used a *trimmed* markout (trimming deletes
exactly the informed tail that *is* adverse selection), and counted arrivals
rather than volume (implying fills of 11× an instrument's daily volume).

**Phase 1.** Collector writes `.tmp` + `os.replace` (readers never see a partial
shard); shards selected by filename timestamp so cost tracks the *window*, not
the retention (2.0 s → 0.68 s); one estimator process per cycle instead of three
(0.80 s full cycle); cycle duration + `--cycle-timeout-seconds`; secrets moved out
of tracked `config.json`.

**Phase 2a.** `mm_core.py` is now the single implementation of quote arithmetic
that previously existed **two or three times over**. It also carries what
freqtrade structurally prevented: both sides priced at once, signed inventory
across the full `[-q_max, +q_max]`, disabled sides staying `None` end-to-end, and
Decimal tick rounding.

**Storage** (asked for separately). Orderbooks cost **27,343 bytes/row against
664 bytes of data** — the cause was file count, not codec: ~360 shards/hour of an
83-column schema means metadata dwarfs payload. Benchmarked parquet/feather/npz;
kept parquet and fixed the real cause. **124 MB → 46 MB live**, orderbooks 76×
smaller, and a **6-hour window now loads in 0.42 s** — same as a 30-minute one,
which is what makes the gate's 6-hour minimum practical. Estimator output is
bit-identical across the change.

**Also fixed:** the exchange test fixture now models freqtrade's real API (the
fiction that hid the kill-switch bug), with a guard test; the README's fee-blind
rule; the replay can simulate two-sided quoting (`--two-sided`).

---

## Where it was left

**Blocked on data collection.** The gate refuses a verdict on <6 h of data — a
short window describes whichever regime it landed in. CASHCAT proved the point:
measured running **19.3× its own daily average volume** for 15 minutes, over which
the profit curve read +$4,117/h, while the other candidates sat at 0.18–0.68×.

`hl-collector` is **running on purpose** (from `HYPERLIQUID_DATA`), collecting
ETH + ACE, CHIP, CASHCAT, PENGU, NIL at 3-day retention. It was at ~1 h of window
when work stopped; the gate needs ~6 h. Nothing else from this project is up.

Candidates came from a screen of all 60 Hyperliquid perps above $0.5M daily
volume: **28 are pinned at exactly one tick like ETH** (a maker cannot even
improve the quote there), and only 9 clear the 3 bps round-trip fee *and* are
wider than one tick. ETH is retained as the known-negative control.

---

## What is left

1. **Run the gate** once ~6 h of data exists:
   ```bash
   python scripts/verify_market_viability.py --crypto ALL --minutes 4320
   ```
   Non-zero exit = nothing viable. **If no symbol passes, that is the answer** —
   passive making does not pay on Hyperliquid at the 1.5 bps base tier, and the
   project should stop rather than proceed.

2. **Phase 2b — `scripts/mm_engine.py`** (only if a symbol passes). Async, WS
   BBO/trades/`userEvents`, both sides resting via `hyperliquid_alo_executor`,
   ~200 ms requote, full signed inventory, `--dry-run`. **A kill switch that
   works**: cancel tracked oids via the SDK, then reduce-only IOC flatten —
   never disable the side that would unwind.

3. **Phase 2c** — replace the `freqtrade` service with `mm-engine`; keep
   `Market_Making.py` for reference only.

4. **Phase 3** — run the two-sided replay on a ≥3-day dataset. **Gate: maker
   fills present and positive net realised spread after two maker fees.** The
   current design has never passed this.

5. **Phase 4** — dry-run bring-up, re-enable the `folder_list.txt` entry (three
   containers, no collector), refresh the gate evidence (all red, ~2 months old).

---

## Read before continuing

- **Secrets in git history.** The old `jwt_secret_key` and API password are still
  in history. Treat as compromised; pick new values in `.env`
  (template: `.env.example`).
- **Do not run the freqtrade strategy near live.** Its kill switch cannot cancel
  resting orders and strands inventory. Documented in two tests now; the engine
  is designed to fix it.
- **Never add a collector to this project's compose.** Two collectors sharing one
  output directory write every tick twice and silently doubled `n_trades` and λ±
  on 2026-08-16.
- **Two-sided quoting does not rescue a dead instrument.** Measured on ETH: it
  doubles quote attempts (692 → 1384) and still gets **zero** maker fills. The
  instrument choice is the binding constraint, not the engine design.
