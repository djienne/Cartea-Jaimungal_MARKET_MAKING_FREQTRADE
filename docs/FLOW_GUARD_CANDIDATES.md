# Guard-candidate study: four signals, four verdicts

Follow-up to `TOXIC_FLOW_GUARD.md`. Four candidate signals were proposed to
fire earlier than the shipped guard or to catch what it cannot see. All four
were investigated on the maximum available frozen CASHCAT tape with criteria
fixed **before** measurement, and every verdict below is the criteria applied
mechanically — no threshold was moved after seeing a result.

| candidate | verdict | one-line reason |
|---|---|---|
| 1. spread-blowout / depth-withdrawal | **DROP** | fires 46 min early — and that makes the crash **3.8x worse** (frozen inventory) |
| 2. bucket-completion-speed alarm | **DROP** | never earlier than the breaker at any parameterization; earliness is bucket-size-fragile |
| 3. inventory-velocity guard | **DROP** | benign net inventory swings are the same size as the cascade's; no threshold separates |
| 4. oracle-vs-mid dislocation + OI drop | **DEFER** | premise confirmed (cascade was CASHCAT-idiosyncratic) but no assetCtx data exists to test the signal |

## Method

- **One freeze.** `scripts/guard_study_tapes/`: `full_tape` (165.11 h,
  08-16 21:59 → 08-23 19:05), plus the immutable `crash_tape`
  (08-21 20:00 → 08-22 12:00) and `calm_tape` (08-19 08:00 → 08-20 00:00)
  reused from the original A/B. Gitignored; scripts in `scripts/guard_study/`.
- **Pre-registered criteria.** KEEP required all three of: **(E)** trips ≥5 s
  before the breaker's 05:11:15 or covers a failure mode the guard cannot see;
  **(FP)** false-positive cost measured as forgone P&L, not counted;
  **(AB)** incremental replay A/B — crash strictly better or equal, calm equal,
  full-tape not worse.
- **Cascade zone** for FP counting: 08-22 03:57 → 09:57, the losing six-hour
  window the staged sweep localised the entire loss to.
- **Self-check.** Every measurement script recomputes the shipped guard's trip
  times from the same series and must land on 05:11:15 (breaker) and 05:11:31
  (VPIN). All did.

## Phase 0 — re-baseline and threshold re-verification

Six replays with the current binary on the frozen tapes:

| window | guard off | guard on |
|---|---:|---:|
| crash | **−87.95** (inv 2,091, 389 fills, 5s markout −166.48) | **−23.13** (inv 339, 234 fills, markout −30.03) |
| calm | −80.90 | −80.90 (bit-identical) |
| full 165 h | −166.65 | −166.65 |

Guard-off crash reproduces `TOXIC_FLOW_GUARD.md` to the cent, proving the
harness faithful. Guard-on reads −23.13 where the doc recorded −33.88: the
trip anatomy is identical (first trip 05:11:15, quotes withheld to ~06:40),
and the delta is **re-entry-timing sensitivity** — VPIN bucket volume derives
from the loaded window and the original A/B's exact window derivation is not
bit-recoverable. Direction and mechanism are unchanged; these re-baselined
numbers are the authoritative ones. The full-tape pair is identical because a
daily-loss limit latches early on this tape (documented in
`TOXIC_FLOW_GUARD.md`), which is why crash/calm windows are the decisive ones.

**Shipped thresholds re-verified on 165.11 h** (`reverify_thresholds.py`):
breaker 631 fires, VPIN 39 breaches — **all inside the cascade zone, zero
outside**. Max 5s move outside: 427 bps (threshold 800). Max VPIN outside:
0.371 (threshold 0.40; was 0.362 on 6.8 days — headroom narrowing slowly,
keep watching as the tape grows).

## Candidate 1 — spread-blowout / depth-withdrawal: DROP

The thesis ("makers pull hundreds of ms before the move") is **not visible on
this tape**: in the two minutes before 05:11:15 the spread sat at 5–40 bps and
top-of-book sizes were normal. What the measurement found instead is an
hour-scale regime signal: the book was visibly degraded from ~03:57, and
`spread ≥ 300 bps` (absolute — beating every adaptive-median variant, the same
way raw VPIN beat its CDF) first fired at **04:25:19, 46 minutes before the
cascade**, with only 3 benign episodes in 165 h (whole-tape max outside the
zone: 387 bps).

E passed, FP passed. Then the A/B destroyed it:

| window | guard alone | guard + spread gate 300 bps | effect |
|---|---:|---:|---:|
| **crash** | **−23.13** | **−88.67** | **−65.54, 3.8x worse** |
| calm | −80.90 | −76.50 | +4.40 |
| full | −166.65 | −149.63 | +17.02 |

The mechanism, from the fill logs: **both legs held +728 units at 04:25:19.**
The spread gate withdrew quoting there and — because empty quotes cancel the
reducing side too — froze that long position through the −70% crash, selling
it near 0.112 on re-entry. The guard-alone leg kept quoting the degraded hour,
actively de-risked in the 05:05–05:09 chop, and entered the cliff with less
exposure and 14 % less left to fall.

**The finding that outlives the candidate: for a market maker with inventory,
earlier withdrawal is not safer.** A quoting gate that fires before the move
freezes whatever position exists for the whole move; a gate that fires as the
move starts freezes a position the market has already begun forcing down. Any
future "get out early" signal must *flatten*, not merely stop quoting — a
different and riskier design (crossing a collapsing book), out of scope here.

The implementation used for this A/B (a `spread_gate_bps` tier in `FlowGuard`)
was reverted after the verdict, per the study's rule that code lands only for
keepers. The A/B configs remain in `scripts/guard_study/configs/*_spreadon.toml`
as the record; they require a build with that transient tier to parse.

## Candidate 2 — bucket-completion-speed alarm: DROP

`bucket_speed.py`, rule `completion_time < median/k AND |imbalance|/V ≥ m`
over k ∈ {5,10,20,50}, m ∈ {0.5,0.7,0.9}, at bucket volumes x0.5 / x1 / x2:

- At the shipped bucket size every (k, m) first fires at **05:11:17 — 2.9 s
  after the breaker**, with 3–13 fires outside the zone.
- At bucket x0.5 it reaches 05:11:14 (+0.84 s — a tie at the tape's ~1 s
  resolution) while outside fires triple (10–42).
- At bucket x2 it is 8.5–18.3 s late.

Never ≥5 s early anywhere in the grid, and the sign of its earliness flips
with bucket size — the fragility the sensitivity check was designed to catch.
The volume clock accelerates *with* the price move on this cascade, not before
it. E failed. Re-examine only if a slow cascade (one the 8%/5s breaker misses)
ever appears on tape; none exists in 165 h.

## Candidate 3 — inventory-velocity guard: DROP

`inventory_velocity.py`, max |Δinventory| over trailing 10/30/60/120 s windows,
from fill streams of the baseline replays plus ten live grid variants:

| source | worst benign 10 s swing | cascade 10 s swing |
|---|---:|---:|
| calm replay | 5,297 | — |
| full replay (outside zone) | 6,339 | — |
| crash replay | 4,214 (before zone) | **6,640** |

Cascade-to-benign ratio is 1.0–1.6x at every window. The pre-registered 3x
margin threshold (~19,000 units) **never crosses even inside the cascade**.
The epistemics argument ("fires only when we are run over") fails empirically:
normal two-sided churn produces net inventory swings as large as the cascade's,
because `q_max` already bounds accumulation — the cascade's damage is the
*price* of the units, not their *rate*. Inventory-level limits (which exist)
dominate any inventory-rate limit on this strategy. E failed; no A/B.

## Candidate 4 — oracle-vs-mid dislocation + OI drop: DEFER

Nothing records `activeAssetCtx` (the bot's wire decoder handles only
bbo/trades/l2Book), so the signal cannot be backtested. Two things were
established now:

1. **The premise holds.** `cross_symbol.py`, 08-22 05:00–05:30 across the
   sibling tapes (frozen to `guard_study_tapes/siblings/` before their 3-day
   retention destroyed the evidence):

   | symbol | worst 5 s move | 05:11 minute |
   |---|---:|---:|
   | **CASHCAT** | **4,716 bps** | **−65.8 %** |
   | NIL | 711 bps | −15.4 % |
   | ACE | 1,016 bps | +6.5 % |
   | CHIP | 452 bps | −3.1 % |
   | PENGU | 375 bps | −0.7 % |
   | ETH | 114 bps | +1.8 % |

   The cascade was CASHCAT-idiosyncratic: the perp dislocated alone and mean
   reverted — exactly the situation oracle-vs-mid divergence measures directly.

2. **Candidate 1's lesson caps the upside.** Even a perfect dislocation signal
   arriving early runs into the frozen-inventory problem; its realistic role is
   *re-entry* quality (distinguishing "dislocated, will revert" from "repriced,
   stay out"), not earlier withdrawal.

Verdict: DEFER — **and collection is now running** (2026-08-23). Rather than a
new container, the existing collectors record it: `activeAssetCtx` is one more
multiplexed subscription on the WebSocket each collector already holds (zero
cost against the 10/IP budget), written as an `asset_ctx/` stream beside
`prices/trades/orderbooks` for every collected symbol. Verified live: both
containers healthy, 4 feeds on the CASHCAT collector, oracle/mark/mid, open
interest, funding and premium landing in parquet. Details in
`DATA_COLLECTION.md`. The pre-registered future test stands: when the recorded
tape contains a cascade, measure oracle-vs-mid divergence and OI-drop timing
against the mid breaker under the same criteria as candidates 1–3 —
remembering that candidate 1's frozen-inventory lesson points its realistic
role at *re-entry* quality, not earlier withdrawal. No bot code.

## Honest limits

- Still n = 1: every earliness number is measured on the one cascade the
  thresholds are then judged against. FP checks use all 165 h, but E cannot
  escape the single event.
- Tape cadence caps timing resolution at ~1 s (`prices` ≈ 1.2 events/s) and
  5 s (`orderbooks`); the shipped breaker's 05:11:15 was measured on the same
  series, so comparisons are like-for-like.
- The candidate-1 A/B is one path through one cascade. The frozen-inventory
  mechanism is structural, but the +17 full-tape improvement it also showed
  means a spread gate *paired with flattening* remains untested, not refuted.

## Files

- `scripts/guard_study/` — `tape.py` (loader + shipped-guard replication),
  `reverify_thresholds.py`, `spread_depth.py`, `bucket_speed.py`,
  `inventory_velocity.py`, `cross_symbol.py`, `make_configs.py`, `configs/`,
  and the per-script `*.json` results.
- `scripts/guard_study_tapes/` (gitignored) — the frozen tapes, sibling
  freeze, and all twelve replay runs with reports and JSONL event logs.
