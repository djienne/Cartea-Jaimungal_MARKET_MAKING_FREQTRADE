#!/usr/bin/env python3
"""Score the whole phi axis on the held-out slice at ONE fixed calibration.

``sweep_replay.py`` stage C only carries its finalists to the held-out slice --
three rows, and in the 2026-09-02 guarded run all three sat at the same
``phi*kappa*T``. That is enough to rank calibrations and not enough to answer
the question a promotion actually asks: is a middle phi better or worse out of
sample than the shipped one? Answering it needs phi varied with everything else
held still, on data the search never saw.

Reuses ``sweep_replay``'s ``build_config``/``score`` rather than re-deriving
them, so a row here is directly comparable to a row there. Reads the frozen
held-out tape the sweep leaves in ``--scratch``, which is the same bytes stage C
scored.

    python scripts/phi_ladder.py                       # defaults below
    python scripts/phi_ladder.py --phi 200 400 1000    # a narrower ladder

Writes ``docs/cashcat_phi_ladder.{json,md}``.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import multiprocessing as mp
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sweep_replay as sw  # noqa: E402
from replay_market_maker import ReplayConfig  # noqa: E402

# The ladder the 2026-09-02 promotion was chosen from. 600 and 800 are not in
# the sweep's own grid; they are here because the interesting question is where
# the curve turns, and at the shipped horizon it turns twice -- 400 and 800 are
# both local dips, so a grid without them reads as monotone when it is not.
DEFAULT_PHI = (30.0, 100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0, 3000.0)

# The horizon this ladder is scored at, and the single most dangerous default in
# this file. `phi_kappa_t` is dimensionless only with T held fixed: the solver
# uses raw phi = phi_kappa_t/(kappa*T), so a rung scored at T=300 describes a
# DIFFERENT control than the same rung shipped at T=150 -- 400@300 is exactly
# 200@150. The first run of this ladder defaulted to 300 while every shipped
# profile runs 150, and the promotion chosen from it was silently twice the
# intended step. This default now tracks `model.horizon_seconds` in
# rust_live/config/cashcat.toml; change them together.
SHIPPED_HORIZON_SECONDS = 150.0

_STATE: dict = {}


def base_config(args: argparse.Namespace) -> ReplayConfig:
    """The sweep's own economics.

    These are ``sweep_replay``'s CLI defaults, and they are not optional: built
    without them the first run of this ladder scored ~0.00 on every row, because
    every lot was the bare ``ReplayConfig`` default size.
    """
    return ReplayConfig(
        symbol=args.symbol,
        data_dir=Path(args.data_dir),
        mid_fallback=args.mid,
        inventory_unit_base=args.inventory_unit_base,
        price_tick_size=args.price_tick_size,
        amount_step_size=args.amount_step_size,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        starting_equity_usdc=args.starting_equity_usdc,
        leverage=args.leverage,
        hjb_time_mode="episodic",
        event_clock="exchange",
    )


def _init(state: dict) -> None:
    _STATE.update(state)


def _run(combo: tuple[float, int, bool]) -> dict:
    phi, q_max, guard = combo
    args = _STATE["args"]
    tape = sw.thaw_tape(Path(args.scratch) / "tape_held_out")
    risk = sw.RiskSetting(phi, args.alpha_kappa, args.horizon_seconds, q_max, guard)
    config = sw.build_config(base_config(args), risk, sw.scenario_by_name(args.scenario))
    row = sw.score(
        config,
        _STATE["params"],
        tape,
        min_fills=args.min_fills,
        min_fills_per_day=args.min_fills_per_day,
    )
    keep = (
        "pnl_usdc",
        "maker_fills",
        "fills_per_day",
        "flow_guard_trips",
        "flow_guard_withheld_decisions",
        "usable",
        "reason",
        "net_realized_spread_usdc",
        "directional_usdc",
        "final_inventory_base",
    )
    return {"phi": phi, "q_max": q_max, "guard": guard, **{name: row.get(name) for name in keep}}


def winning_params(sweep: Path, calibration: str | None) -> tuple[str, dict]:
    """The calibration to hold fixed: a named one, else stage C's best held-out."""
    payload = json.loads(sweep.read_text(encoding="utf-8"))
    rows = payload.get("stage_c") or []
    if not rows:
        raise SystemExit(f"{sweep} carries no stage_c rows to take a calibration from")
    if calibration is not None:
        rows = [row for row in rows if calibration in row["key"]]
        if not rows:
            raise SystemExit(f"no stage_c row matches {calibration!r}")
    best = max(rows, key=lambda row: row["held_out"]["pnl_usdc"])
    return best["key"], best["params"]


def to_markdown(rows: list[dict], key: str, args: argparse.Namespace) -> str:
    lines = [
        "# Held-out phi ladder — " + args.symbol,
        "",
        "Generated by `scripts/phi_ladder.py`. Stage C of `cashcat_sweep.md` carries",
        "only its finalists to the held-out slice, so it cannot say whether a middle",
        "`phi_kappa_t` is better or worse out of sample than the shipped one. This",
        "varies phi alone, on data the search never saw.",
        "",
        f"- calibration held fixed: `{key}`",
        f"- risk held fixed: `alpha_kappa={args.alpha_kappa:g}`, "
        f"**`horizon_seconds={args.horizon_seconds:g}`**, scenario `{args.scenario}`",
        "- **`phi_kappa_t` is only comparable at a FIXED horizon.** The solver uses",
        "  raw `phi = phi_kappa_t / (kappa * T)` and the surface is stationary away",
        "  from the terminal layer, so a rung here matches a shipped config only when",
        "  this horizon equals `model.horizon_seconds` in the profile being changed.",
        f"  Rungs below are directly comparable to a profile with `horizon_seconds ="
        f" {args.horizon_seconds:g}` and to no other.",
        f"- floor: at least {args.min_fills} fills and {args.min_fills_per_day:g}/day",
        "",
        "| phi*kappa*T | q_max | guard | held-out P&L | fills | fills/day | "
        "net spread | directional | trips | usable |",
        "| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in sorted(rows, key=lambda row: (row["phi"], row["q_max"], not row["guard"])):
        lines.append(
            f"| {row['phi']:g} | {row['q_max']} | {'on' if row['guard'] else 'off'} | "
            f"{row['pnl_usdc']:+.2f} | {row['maker_fills']} | {row['fills_per_day']:.1f} | "
            f"{row['net_realized_spread_usdc']:+.2f} | {row['directional_usdc']:+.2f} | "
            f"{row['flow_guard_trips']} | {row['usable']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="held-out phi ladder at a fixed calibration")
    parser.add_argument("--phi", type=float, nargs="+", default=list(DEFAULT_PHI))
    parser.add_argument("--q-max", type=int, nargs="+", default=[6])
    parser.add_argument(
        "--guard",
        type=int,
        nargs="+",
        default=[1, 0],
        help="1 on, 0 off; both by default so every row carries its own A/B",
    )
    parser.add_argument("--alpha-kappa", type=float, default=0.05)
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=SHIPPED_HORIZON_SECONDS,
        help=(
            "MUST equal model.horizon_seconds in the profile you intend to change, "
            "or the rungs describe controls you cannot ship. See "
            "SHIPPED_HORIZON_SECONDS."
        ),
    )
    parser.add_argument("--scenario", default="good")
    parser.add_argument("--sweep", type=Path, default=ROOT / "docs" / "cashcat_sweep.json")
    parser.add_argument(
        "--calibration",
        default=None,
        help="substring of a stage_c key; default is stage C's best held-out row",
    )
    parser.add_argument("--scratch", type=Path, default=Path.home() / ".cache" / "mm_sweep")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-fills", type=int, default=30)
    parser.add_argument("--min-fills-per-day", type=float, default=3.0)
    parser.add_argument("--symbol", default="CASHCAT")
    parser.add_argument("--data-dir", type=Path, default=SCRIPTS / "HL_data")
    parser.add_argument("--mid", type=float, default=0.1015)
    parser.add_argument("--price-tick-size", type=float, default=1e-6)
    parser.add_argument("--amount-step-size", type=float, default=0.0)
    parser.add_argument("--inventory-unit-base", type=float, default=2092.0)
    parser.add_argument("--maker-fee", type=float, default=0.00015)
    parser.add_argument("--taker-fee", type=float, default=0.00045)
    parser.add_argument("--starting-equity-usdc", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "cashcat_phi_ladder.json")
    parser.add_argument(
        "--markdown-output", type=Path, default=ROOT / "docs" / "cashcat_phi_ladder.md"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    held_out = Path(args.scratch) / "tape_held_out"
    if not held_out.exists():
        raise SystemExit(
            f"{held_out} is missing -- run sweep_replay.py first. It freezes the "
            "held-out tape there, and this ladder has to score the same bytes."
        )
    key, params = winning_params(args.sweep, args.calibration)
    print(f"[calibration] {key}", flush=True)

    combos = list(itertools.product(args.phi, args.q_max, [bool(flag) for flag in args.guard]))
    state = {"args": args, "params": params}
    with mp.Pool(max(1, args.workers), initializer=_init, initargs=(state,)) as pool:
        rows = pool.map(_run, combos)

    # Provenance, not just rows. Written because it was not: the first version
    # of this artifact recorded phi/q_max/guard and the metrics and nothing
    # else, so it could not say which tape, which split, which calibration or
    # -- the one that actually bit -- WHICH HORIZON it was scored at. A ladder
    # run at a horizon other than the shipped one is not comparable to the
    # shipped config at all, since phi_kappa_t is only dimensionless with T
    # held fixed, and a reader had no way to notice.
    payload = {
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "calibration_key": key,
        "params": params,
        "held_fixed": {
            "horizon_seconds": float(args.horizon_seconds),
            "alpha_kappa": float(args.alpha_kappa),
            "scenario": dataclasses.asdict(sw.scenario_by_name(args.scenario)),
            "min_fills": int(args.min_fills),
            "min_fills_per_day": float(args.min_fills_per_day),
        },
        "sweep": str(args.sweep),
        "held_out_tape": str(held_out),
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    args.markdown_output.write_text(to_markdown(rows, key, args), encoding="utf-8")
    for row in sorted(rows, key=lambda row: row["phi"]):
        print(
            f"  phi={row['phi']:>7g} guard={str(row['guard']):<5} "
            f"pnl={row['pnl_usdc']:>9.2f} fills={row['maker_fills']:>6} "
            f"({row['fills_per_day']:.1f}/day) trips={row['flow_guard_trips']}",
            flush=True,
        )
    print(f"[done] {args.output}  {args.markdown_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
