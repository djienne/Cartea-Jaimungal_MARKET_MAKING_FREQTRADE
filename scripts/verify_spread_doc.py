#!/usr/bin/env python3
"""Keep ``docs/spread_calculation.tex`` honest.

The document quotes real source lines and real numbers. Both rot silently: a
refactor moves a function and the listing becomes fiction, a default changes and
the worked example describes a spread the code no longer produces. Nothing in a
PDF complains about that, so this script does.

Two checks, both exact:

1. **Snippets.** Every listing in the ``.tex`` is preceded by

       \snip{scripts/mm_core.py}{224}{240}

   which both typesets the attribution for the reader and tells this script what
   to check. It re-reads those lines from the repo and compares them with the
   listing body character for character. A moved or edited function fails here.

2. **Numbers.** Every worked-example figure is tagged

       % CHECK: delta_total = 2.4786e-04

   and this script recomputes it by calling the Python replay/reference pricing
   path, then compares to the printed precision. Rust parity is checked
   separately by ``rust_live/tests/python_parity.rs``. Tagged worked-example
   numbers are therefore executable checks; dated evidence tables elsewhere in
   the document are historical inputs, not recomputed by this script.

``--print`` dumps the worked example instead of checking it, which is how the
numbers got into the document in the first place.

``--fix-snippets`` re-locates the markers when a listing's text still exists in
its file but has moved. Editing anything above a quoted range shifts every
marker below it, and hand-patching line numbers is both tedious and the sort of
job that gets done carelessly. This only ever moves a marker to a range whose
text matches the listing exactly; if the text genuinely changed it refuses and
you get the normal failure.

Book cross-check (done by hand against ``docs/[...]Algorithmic and
High-Frequency Trading[...].pdf``, printed pages 262-264, and recorded here so
the provenance is not lost):

    eq. 10.22  midprice with MO impact          -> Part 1, verbatim
    eq. 10.23  the DPE                          -> Part 1, verbatim
    eq. 10.25  the ansatz H = x + qS + h(t,q)   -> Part 1, verbatim
    eq. 10.26  the h-equation, h(T,q) = -alpha q^2 -> Part 1, verbatim
    eq. 10.27  delta+- = 1/kappa +- eps -+ dh   -> Part 1 and hjb._depths_from_h
    eq. 10.28  the matrix A                     -> Part 3 against hjb.py:235-239
    eq. 10.29  omega(t) = e^{A(T-t)} z          -> Part 3 against hjb.py:249-274

Usage:
    python scripts/verify_spread_doc.py
    python scripts/verify_spread_doc.py --print
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

import mm_core  # noqa: E402

TEX = REPO / "docs" / "spread_calculation.tex"

# ---------------------------------------------------------------------------
# The pinned snapshot the worked example runs on.
#
# One coherent estimator window from the live CASHCAT dry run, 2026-08-17
# 14:20:18Z - 16:20:18Z (120 minutes), re-estimated on 2026-09-02 under
# parameter schema v5 (`python scripts/get_kappa.py` / `get_epsilon.py`
# with --window-start/--window-end). v5 hands the HJB lambda_raw times the
# survival-fit intercept; the kappa/epsilon values differ from the original
# 2026-08-17T16:20:41Z live cycle by under 1% because the collector later
# compacted the window (170 s of outage is now excluded from the denominator).
# Pinned rather than read from the live files so the document's numbers do
# not change every 30 seconds.
# ---------------------------------------------------------------------------
SNAPSHOT = {
    "kappa+": 10055.673508266684,
    "kappa-": 9398.654070051622,
    # lambda_raw 0.12022191331670479 x survival intercept 1.03272...
    "lambda+": 0.12415621832565442,
    # lambda_raw 0.13182855810651778 x survival intercept 1.12928...
    "lambda-": 0.14887163479331725,
    "epsilon+": 2.5224725943970184e-05,
    "epsilon-": 2.7800333704115578e-05,
}
SIGMA2_PER_SEC = 3.352917846799544e-09
DEPTH_P95_PLUS = 0.0003251999999999934
DEPTH_P95_MINUS = 0.0003440499999999971

MID = 0.105685
PRICE_TICK = 1e-5
INVENTORY_UNIT_BASE = 2527.322404371585
TAU_REMAINING = 130.786584

# A deliberately fractional inventory: 6065.6 CASHCAT is 2.4 units, which is
# what a partial fill actually leaves behind and what the interpolation exists
# to price. An integer q would hide the whole mechanism.
SIGNED_BASE = 2.4 * INVENTORY_UNIT_BASE


def build_config() -> mm_core.QuoteConfig:
    """The worked example's pinned Python reference configuration."""
    return mm_core.QuoteConfig(
        maker_fee_rate=0.00015,
        inventory_unit_base=INVENTORY_UNIT_BASE,
        q_max=6,
        allow_short=True,
        hjb_horizon_seconds=150.0,
        hjb_time_mode="episodic",
    )


def worked_example() -> dict[str, float | int | str | None]:
    """Run the real pricing path once and return every intermediate value."""
    config = build_config()
    out: dict[str, float | int | str | None] = {}

    # --- Stage 2: the risk parameters the solver actually receives ----------
    kappa_avg = 0.5 * (SNAPSHOT["kappa+"] + SNAPSHOT["kappa-"])
    phi_vol, phi_vol_source = mm_core.effective_phi(config, SIGMA2_PER_SEC)
    out["kappa_avg"] = kappa_avg
    out["vol_delta"] = phi_vol - config.hjb_phi
    out["phi_kappa_relative"] = config.hjb_phi_kappa_t / (
        kappa_avg * config.hjb_horizon_seconds
    )
    out["n_steps"] = mm_core.hjb_n_steps(config)
    out["dt"] = config.hjb_horizon_seconds / out["n_steps"]

    hjb = mm_core.solve_hjb(SNAPSHOT, config, sigma2_per_sec=SIGMA2_PER_SEC)
    out["phi_effective"] = hjb["phi_effective"]
    out["phi_source"] = hjb["phi_source"]
    out["alpha_effective"] = hjb["alpha_effective"]
    out["phi_kappa_t_dimensionless"] = (
        hjb["phi_effective"] * kappa_avg * config.hjb_horizon_seconds
    )
    out["alpha_kappa_dimensionless"] = hjb["alpha_effective"] * kappa_avg
    out["has_surface"] = "delta_plus_surface" in hjb

    # --- Stage 3: reading delta*(t,q) --------------------------------------
    q_exact = mm_core.inventory_to_q_exact(SIGNED_BASE, config)
    q_int = mm_core.inventory_to_q(SIGNED_BASE, config)
    out["q_exact"] = q_exact
    out["q_int"] = q_int
    out["q_residual"] = q_exact - q_int

    # The two integer nodes the fractional q sits between, so the document can
    # show the blend rather than assert it.
    for q_node in (2, 3):
        out[f"delta_plus_q{q_node}"] = mm_core.select_delta(
            hjb, float(q_node), "ask", tau_remaining=TAU_REMAINING
        )
        out[f"delta_minus_q{q_node}"] = mm_core.select_delta(
            hjb, float(q_node), "bid", tau_remaining=TAU_REMAINING
        )

    # --- Stages 3+4 together, exactly as the strategy calls them -----------
    pair = mm_core.compute_quotes(
        MID,
        q_int,
        hjb,
        config,
        depth_p95_plus=DEPTH_P95_PLUS,
        depth_p95_minus=DEPTH_P95_MINUS,
        price_tick_size=PRICE_TICK,
        tau_remaining=TAU_REMAINING,
        q_exact=q_exact,
    )

    for side in ("bid", "ask"):
        half = getattr(pair, side)
        if half is None:
            out[f"{side}_disabled"] = True
            continue
        out[f"{side}_disabled"] = False
        out[f"{side}_delta_model"] = half.delta_model
        out[f"{side}_fee_cushion"] = half.fee_cushion
        out[f"{side}_delta_pre_clamp"] = half.delta_pre_clamp
        out[f"{side}_delta_total"] = half.delta
        out[f"{side}_bps"] = half.bps
        out[f"{side}_clamped"] = half.clamped
        out[f"{side}_outside_calibrated"] = half.outside_calibrated_range

    out["floor_price"] = config.min_half_spread_bps / 10_000.0 * MID
    out["cap_price"] = config.max_half_spread_bps / 10_000.0 * MID

    # --- The three terms of eq. 10.27, at the integer node q=2 --------------
    # The document decomposes the depth rather than presenting it as solver
    # output, so h has to come out of the surface rather than be back-solved.
    t_grid = np.asarray(hjb["t_grid"], dtype=float)
    row = int(np.searchsorted(t_grid, float(hjb["T_seconds"]) - TAU_REMAINING))
    h_slice = np.asarray(hjb["h_surface"], dtype=float)[row]
    q_grid = [int(q) for q in hjb["q_grid"]]
    idx2 = q_grid.index(2)
    h_diff = float(h_slice[idx2 - 1] - h_slice[idx2])
    kappa_p = SNAPSHOT["kappa+"]
    out["h_diff_q2"] = h_diff
    out["rent_bps"] = (1.0 / kappa_p) / MID * 10_000.0
    out["adverse_bps"] = SNAPSHOT["epsilon+"] / MID * 10_000.0
    out["inventory_bps"] = -h_diff / MID * 10_000.0
    out["delta_plus_q2_bps"] = (1.0 / kappa_p + SNAPSHOT["epsilon+"] - h_diff) / MID * 10_000.0

    # --- The floor IS one maker fee, so the clamp binds at delta_model < 0 ---
    out["floor_equals_fee_cushion"] = (
        config.min_half_spread_bps / 10_000.0 * MID == config.maker_fee_rate * MID
    )
    d1 = mm_core.select_delta(hjb, 1.0, "ask", tau_remaining=TAU_REMAINING)
    d2 = mm_core.select_delta(hjb, 2.0, "ask", tau_remaining=TAU_REMAINING)
    out["ask_zero_crossing_q"] = 1.0 + d1 / (d1 - d2)

    # --- What the finished quotes are worth, back through eq. 10.14 ---------
    ask_delta = config.maker_fee_rate * MID
    out["ask_fill_rate_per_sec"] = SNAPSHOT["lambda+"] * math.exp(-kappa_p * ask_delta)
    out["ask_seconds_per_fill"] = 1.0 / out["ask_fill_rate_per_sec"]
    out["toxicity_plus"] = kappa_p * SNAPSHOT["epsilon+"]
    out["fills_kept_at_live_toxicity_pct"] = math.exp(-out["toxicity_plus"]) * 100.0
    out["fills_kept_at_gate_pct"] = math.exp(-1.5) * 100.0

    # --- The same machine at four inventories ------------------------------
    for q_probe in (0.0, 0.4, 2.4, 5.4):
        probe = mm_core.compute_quotes(
            MID,
            int(round(q_probe)),
            hjb,
            config,
            price_tick_size=PRICE_TICK,
            tau_remaining=TAU_REMAINING,
            q_exact=q_probe,
        )
        tag = f"q{str(q_probe).replace('.', 'p')}"
        out[f"{tag}_bid_bps"] = None if probe.bid is None else probe.bid.bps
        out[f"{tag}_ask_bps"] = None if probe.ask is None else probe.ask.bps
    out["bid_price"] = pair.bid_price
    out["ask_price"] = pair.ask_price
    if pair.bid is not None:
        out["bid_raw_price"] = MID - pair.bid.delta
    if pair.ask is not None:
        out["ask_raw_price"] = MID + pair.ask.delta
    if pair.bid_price is not None and pair.ask_price is not None:
        out["quoted_spread_bps"] = (
            (pair.ask_price - pair.bid_price) / MID * 10_000.0
        )
    return out


# ---------------------------------------------------------------------------
# Check 1: listings match their source
# ---------------------------------------------------------------------------
# The document declares each listing's provenance with a VISIBLE command:
#
#     \snip{scripts/hjb.py}{227}{241}
#
# which typesets the file and line range above the listing. It used to be a
# LaTeX comment, which meant the attribution existed only for this script --
# the reader of the PDF saw twelve unlabelled code blocks and had to take the
# document's word for where they came from. Parsing the same command the reader
# sees keeps the claim and the check from ever drifting apart.
SNIPPET_RE = re.compile(
    r"^\s*\\snip\{(?P<path>[^}]+)\}\{(?P<start>\d+)\}\{(?P<end>\d+)\}\s*$"
)


def _listing_bodies(lines: list[str]) -> list[tuple[int, str, list[str]]]:
    """(marker line no, 'path:start-end', body lines) for each marked listing."""
    found = []
    for idx, line in enumerate(lines):
        match = SNIPPET_RE.match(line.rstrip("\n"))
        if match is None:
            continue
        # The listing must open on one of the next few lines.
        begin = None
        for probe in range(idx + 1, min(idx + 5, len(lines))):
            if lines[probe].lstrip().startswith(r"\begin{lstlisting}"):
                begin = probe
                break
        if begin is None:
            raise SystemExit(
                f"{TEX.name}:{idx + 1}: SNIPPET marker is not followed by a listing"
            )
        end = None
        for probe in range(begin + 1, len(lines)):
            if lines[probe].lstrip().startswith(r"\end{lstlisting}"):
                end = probe
                break
        if end is None:
            raise SystemExit(f"{TEX.name}:{begin + 1}: unterminated listing")
        spec = f"{match.group('path')}:{match.group('start')}-{match.group('end')}"
        found.append((idx + 1, spec, [ln.rstrip("\n") for ln in lines[begin + 1 : end]]))
    return found


def _locate(source_lines: list[str], body: list[str]) -> int | None:
    """1-based start line of the unique occurrence of ``body``, else None.

    Refuses an ambiguous match: if a listing's text appears twice in its file,
    silently picking the first would be a coin flip.
    """
    if not body:
        return None
    wanted = [line.rstrip() for line in body]
    stripped = [line.rstrip() for line in source_lines]
    hits = [
        idx
        for idx in range(len(stripped) - len(wanted) + 1)
        if stripped[idx : idx + len(wanted)] == wanted
    ]
    return hits[0] + 1 if len(hits) == 1 else None


def fix_snippets(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Rewrite markers whose text has moved. Returns (lines, moved, unresolved)."""
    moved: list[str] = []
    unresolved: list[str] = []
    for lineno, spec, body in _listing_bodies(lines):
        path_str, span = spec.rsplit(":", 1)
        start, end = (int(part) for part in span.split("-"))
        source = REPO / path_str
        if not source.is_file():
            unresolved.append(f"{spec}: no such file")
            continue
        source_lines = source.read_text(encoding="utf-8").splitlines()
        current = source_lines[start - 1 : end]
        if len(current) == len(body) and all(
            a.rstrip() == b.rstrip() for a, b in zip(current, body)
        ):
            continue  # still correct
        found = _locate(source_lines, body)
        if found is None:
            unresolved.append(
                f"{spec}: listing text not found (or found more than once) "
                f"-- the code really changed, fix the document"
            )
            continue
        new_spec = f"{path_str}:{found}-{found + len(body) - 1}"
        lines[lineno - 1] = f"\\snip{{{path_str}}}{{{found}}}{{{found + len(body) - 1}}}\n"
        moved.append(f"{spec} -> {new_spec}")
    return lines, moved, unresolved


def check_snippets(lines: list[str]) -> list[str]:
    failures = []
    for lineno, spec, body in _listing_bodies(lines):
        path_str, span = spec.rsplit(":", 1)
        start, end = (int(part) for part in span.split("-"))
        source = REPO / path_str
        if not source.is_file():
            failures.append(f"{TEX.name}:{lineno}: no such file {path_str}")
            continue
        actual = source.read_text(encoding="utf-8").splitlines()[start - 1 : end]
        if len(actual) != len(body):
            failures.append(
                f"{TEX.name}:{lineno}: {spec} spans {len(actual)} lines, "
                f"listing has {len(body)}"
            )
            continue
        for offset, (want, got) in enumerate(zip(actual, body)):
            if want.rstrip() != got.rstrip():
                failures.append(
                    f"{TEX.name}:{lineno}: {path_str}:{start + offset} drifted\n"
                    f"    source:  {want.rstrip()!r}\n"
                    f"    listing: {got.rstrip()!r}"
                )
                break
    return failures


# ---------------------------------------------------------------------------
# Check 2: printed numbers match a live recomputation
# ---------------------------------------------------------------------------
CHECK_RE = re.compile(r"^%\s*CHECK:\s*(?P<name>[A-Za-z0-9_]+)\s*=\s*(?P<value>\S+)\s*$")


def _matches_to_printed_precision(printed: str, actual: float) -> bool:
    """True when ``actual`` rounds to ``printed`` at the precision printed.

    The document rounds for readability, so an exact comparison would fail on
    every line. Reformatting ``actual`` with the same format the author used and
    comparing the strings tests exactly what the reader sees.
    """
    text = printed.strip()
    if text.lower() in {"none", "null"}:
        return actual is None
    if "e" in text.lower():
        mantissa = text.lower().split("e")[0]
        digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return f"{actual:.{digits}e}" == text.lower()
    digits = len(text.split(".")[1]) if "." in text else 0
    return f"{actual:.{digits}f}" == text


def check_numbers(lines: list[str], values: dict) -> list[str]:
    failures = []
    seen = 0
    for idx, line in enumerate(lines):
        match = CHECK_RE.match(line.rstrip("\n"))
        if match is None:
            continue
        seen += 1
        name = match.group("name")
        printed = match.group("value")
        if name not in values:
            failures.append(
                f"{TEX.name}:{idx + 1}: CHECK names {name!r}, which the worked "
                f"example does not produce"
            )
            continue
        actual = values[name]
        if isinstance(actual, str) or actual is None or isinstance(actual, bool):
            if str(actual) != printed:
                failures.append(
                    f"{TEX.name}:{idx + 1}: {name} printed as {printed!r}, "
                    f"computed {actual!r}"
                )
            continue
        if not _matches_to_printed_precision(printed, float(actual)):
            failures.append(
                f"{TEX.name}:{idx + 1}: {name} printed as {printed}, "
                f"computed {actual!r}"
            )
    if seen == 0:
        failures.append(f"{TEX.name}: no CHECK markers found -- numbers unverified")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="dump the worked example instead of checking the document",
    )
    parser.add_argument(
        "--fix-snippets",
        action="store_true",
        help="re-locate markers whose quoted text moved, then check as usual",
    )
    args = parser.parse_args()

    if args.fix_snippets:
        if not TEX.is_file():
            print(f"FAIL: {TEX} does not exist", file=sys.stderr)
            return 1
        lines = TEX.read_text(encoding="utf-8").splitlines(keepends=True)
        lines, moved, unresolved = fix_snippets(lines)
        if moved:
            TEX.write_text("".join(lines), encoding="utf-8")
        for entry in moved:
            print(f"moved: {entry}")
        for entry in unresolved:
            print(f"UNRESOLVED: {entry}", file=sys.stderr)
        if not moved and not unresolved:
            print("nothing to move")

    values = worked_example()

    if args.do_print:
        width = max(len(k) for k in values)
        for key, value in values.items():
            if isinstance(value, float):
                print(f"{key:<{width}} = {value!r:<26} {value:.6e}")
            else:
                print(f"{key:<{width}} = {value!r}")
        return 0

    if not TEX.is_file():
        print(f"FAIL: {TEX} does not exist", file=sys.stderr)
        return 1

    lines = TEX.read_text(encoding="utf-8").splitlines(keepends=True)
    failures = check_snippets(lines) + check_numbers(lines, values)

    if failures:
        print(f"{len(failures)} problem(s) in {TEX.name}:\n", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    snippets = len(_listing_bodies(lines))
    checks = sum(1 for line in lines if CHECK_RE.match(line.rstrip("\n")))
    print(f"OK: {snippets} snippets match source, {checks} numbers recomputed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
