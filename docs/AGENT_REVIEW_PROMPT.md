# Review the method and its implementation

Evaluate this Cartea–Jaimungal market-making research project as a quantitative
researcher and experimental scientist. Focus on whether its mathematics,
estimation, numerical methods, and execution model support trustworthy results.
Do not turn the review into an infrastructure or compliance project.

Read the source, not just the README. Use the included Cartea, Jaimungal and
Penalva (2015) book and the FrenchQuant notebook
`docs/market_making_introduction.ipynb` as references. Identify what the project
implements, changes, or assumes beyond them. Neither the book, the notebook,
prior reports, nor agreement between implementations is automatic ground truth.

Some deviations were introduced in pursuit of profitability. **Are those
deviations scientifically and economically justified?** For each material change,
identify the book section/equation or notebook cell, the implemented departure,
its stated rationale (or say it is undocumented), and the evidence supporting it.
Distinguish necessary venue adaptations and sound model extensions from
outcome-driven tuning, selection bias, or optimistic execution assumptions.
Improved simulated P&L alone is not a justification; deviation alone is not a flaw.

Pay particular attention to spread floors, inventory penalties, calibration
choices, quote lifetimes, queue assumptions and rapid taker exits. Explain which
changes preserve the original model's assumptions and which create a different
strategy. Assess sensitivity and feasible execution costs; propose matched
ablations where evidence is missing. Conclude which deviations to retain,
revise, reject, or leave as unproven hypotheses, and why.

Trace the actual path: collected data → calibration → HJB solve → quotes and
inventory sizing → simulated/live execution → P&L. Inspect especially:

- Equation signs, units, boundary/terminal conditions, inventory penalties,
  discretization, solver residuals and convergence over the parameter regime used.
- Estimator definitions, market-order aggregation, timestamp alignment, missing
  data, clipping, fit support and whether fitted intensities match the model.
- Causality and look-ahead: what data are known when orders activate, cancel or
  fill; queue priority, partial fills, post-only behavior and latency tails.
- Adverse selection, inventory valuation, fees, funding, taker exits, margin
  and risk stops. Distinguish realized profit from marked inventory and ask
  whether positive paper results could survive actual execution constraints.
- Differences between Python, Rust replay, the live-feed paper grid, and real
  execution. Diagnose disagreements rather than forcing outputs to match.
- Selection bias, reused train/test periods, dependent windows, sensitivity to
  assumptions, and evidence needed to distinguish an edge from simulation bias.

Start with `README.md`, `docs/CAUSAL_EXECUTION_REVIEW.md`, `docs/DRY_RUN_GRID.md`,
then trace `rust_live/src/paper.rs`, the Rust crates, and the Python estimators.
The included paper checkpoint, leaderboard and equity history are read-only
snapshots taken while the run continued; their timestamps can differ. History
spans configuration/code changes. Raw market tape and full event logs are not
included: state which conclusions cannot be tested without them.

Give a short scientific assessment, followed by findings ranked by their effect
on results. For each substantiated finding, cite file/line or equation, explain
the mechanism and a concrete consequence, and propose the smallest correction
or discriminating experiment. Separate confirmed defects from assumptions,
model limitations, and untested hypotheses. Include a few highest-value next
experiments; avoid generic checklists, speculative abstractions, extra manifests,
and test suites that merely restate the code.

This is a review-only task. Do not start trading services or send real orders.
Preserving the long dry-run history is mandatory: never propose or perform a
fresh grid, reset counters/accounts, revive stopped variants, or discard history
for a small fix. Changes must resume the same experiment and mark their time.
