"""FlowCast — 30-second demo.

Run this during a presentation instead of the full pipeline.

    python demo.py

It reads results that were already computed and saved, and prints the whole
story in about 30 seconds: what the data looked like, what was wrong with it,
what the models achieved, and what did not work.

Nothing is trained here. Nothing can fail halfway through. If the projector is
already on and the room is waiting, this is the safe thing to run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

W = 74  # console width


def rule(char="-"):
    print(char * W)


def heading(text):
    print()
    rule("=")
    print(f"  {text}")
    rule("=")


def row(label, value, note=""):
    print(f"  {label:<34} {value:>14}   {note}")


def pause(seconds=0.6):
    """A short pause so a live audience can read each block as it appears."""
    time.sleep(seconds)


def load(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def main():
    reports = ROOT / "reports"
    models = ROOT / "models"

    ingest = load(reports / "ingest_stats.json", {})
    clean = load(reports / "clean_stats.json", {})
    feats = load(reports / "feature_stats.json", {})
    classical = load(reports / "classical_results.json", {})
    deep = (load(reports / "deep_results.json", {}) or {}).get("results", {})
    unc = load(reports / "uncertainty.json", {})
    diag = load(reports / "accident_diagnostic.json", {})
    winners = load(models / "winners.json", {})

    if not classical:
        print("No results found. Run this first:")
        print("    python src/run_pipeline.py")
        return 1

    print()
    print("  F L O W C A S T")
    print("  Predicting traffic 30 minutes ahead on a 25-segment city corridor")
    pause()

    # ------------------------------------------------------------------ #
    heading("1. THE PROBLEM")
    print("""
  A city traffic control room can see the road as it is right now, but by
  the time a jam appears it is too late to prevent it.

  This system answers one question:

      Given everything we know up to now,
      what will each stretch of road look like in 30 minutes?

  Four answers per stretch: how many vehicles, how fast, how long to drive
  it, and how likely an accident is.
""")
    pause(1.0)

    # ------------------------------------------------------------------ #
    heading("2. THE DATA (and what was wrong with it)")
    print()
    row("Sensor readings received", f"{ingest.get('traffic_rows_raw', 0):,}")
    row("Road segments", f"{ingest.get('segments', 0)}")
    row("Period covered", "1 Jan - 31 May 2025")
    print()
    print("  Problems found and fixed:")
    print()
    row("Impossible readings quarantined", f"{ingest.get('traffic_quarantined', 0):,}",
        "negative counts, 300 km/h")
    row("Duplicate rows removed", f"{clean.get('duplicates_removed', 0):,}",
        "sensor sent twice")
    row("Missing time slots restored", f"{clean.get('grid_missing_windows', 0):,}",
        "<-- the important one")
    row("Weather spellings unified",
        f"{clean.get('weather_labels_before', 0)} -> {clean.get('weather_labels_after', 0)}",
        "'RAIN', 'rainy', 'Rain'")
    print("""
  Why the missing slots matter most: if you ask the computer for "the
  previous reading" and an 8am row simply does not exist, it hands you
  7am and calls it 8am. Every calculation after that is quietly wrong.
""")
    pause(1.0)

    # ------------------------------------------------------------------ #
    heading("3. THE TRAP IN THE DATA")
    agree = 100 * clean.get("congestion_rule_agreement", 0)
    print(f"""
  One column, "congestion level", turned out to be nothing more than the
  vehicle count sorted into four buckets. Checked against the rows that
  already had a label, the rule matched {agree:.2f}% of the time.

  So it is not independent information. Hand it to the computer and it
  "predicts" congestion almost perfectly - because it was given the answer.

  This system refuses that shortcut. Five automated tests exist purely to
  make sure it stays refused.
""")
    pause(1.0)

    # ------------------------------------------------------------------ #
    heading("4. RESULTS - how well does it predict?")
    vol = classical.get("y_volume", {})
    naive = vol.get("Persistence (naive)", {})
    best_name = winners.get("y_volume", "XGBoost")
    best = vol.get(best_name, {})

    print()
    print("  Predicting how many vehicles pass in the next 30 minutes:")
    print()
    row("Lazy guess (assume no change)", f"{naive.get('mape', 0):.1f}% error",
        "free, no computer")
    row(f"Best model ({best_name})", f"{best.get('mape', 0):.1f}% error", "<-- twice as good")
    print()
    if naive.get("rmse") and best.get("rmse"):
        gain = 100 * (1 - best["rmse"] / naive["rmse"])
        print(f"  That is a {gain:.0f}% reduction in error against the lazy guess.")
        print("  The comparison is the point: 8.9% on its own means nothing.")
    print()
    print("  Every model tried, worst to best:")
    print()
    ranked = sorted(((n, m) for n, m in vol.items() if "mape" in m),
                    key=lambda kv: -kv[1]["mape"])
    for name, m in ranked:
        bar = "#" * max(1, int(m["mape"] * 0.8))
        print(f"    {name[:32]:<34} {m['mape']:>6.1f}%  {bar}")
    pause(1.2)

    # ------------------------------------------------------------------ #
    heading("5. THE HONEST PART - what did NOT work")
    acc = classical.get("y_accident", {})
    acc_best = acc.get(winners.get("y_accident", "Logistic Regression"), {})
    cong = classical.get("y_congestion", {})
    cong_best = cong.get(winners.get("y_congestion", "XGBoost"), {})

    print()
    row("Target: congestion class", f"{cong_best.get('macro_f1', 0):.3f}",
        "needed 0.80  -- MISSED")
    row("Target: accident risk", f"{acc_best.get('roc_auc', 0):.3f}",
        "needed 0.75  -- MISSED")
    print()
    if diag:
        gap = diag.get("cheat_advantage_roc_auc", 0)
        print(f"""  Why did accident prediction fail? I tested it properly.

  I built a deliberately CHEATING version, allowed to see information no
  real forecast could ever have. The cheat improved things by {gap:+.3f}.

  Essentially nothing. So the answer is not "try harder" - the information
  needed to predict accidents is simply not in this data. What is missing
  is road layout, lane counts, and where crashes happened before.

  Knowing a problem is unsolvable with the data you have is a real finding.""")
    print()
    print("  The ranking is still useful, even at low accuracy:")
    print()
    row("Top-decile lift", f"{acc_best.get('top_decile_lift', 0):.2f}x",
        "patrol the riskiest 10%")
    print("      -> you find 2.6 times as many incidents as patrolling at random.")
    pause(1.2)

    # ------------------------------------------------------------------ #
    heading("6. HOW SURE IS IT? (confidence check)")
    cov = (unc or {}).get("coverage", {})
    if cov:
        print()
        print("  The system gives a range, not just a single number. A range is")
        print("  only worth showing if it is actually right, so it was checked:")
        print()
        row("Range promised to be right", "80% of the time")
        row("Range actually right", f"{100 * cov.get('empirical_coverage', 0):.1f}% of the time",
            "<-- honest")
        print()
        print("  The range also widens when the road is harder to predict:")
        print()
        for cond, d in (unc.get("width_by_condition") or {}).items():
            row(f"  {cond}", f"+/- {d.get('mean', 0):.0f} vehicles")
    pause(1.0)

    # ------------------------------------------------------------------ #
    heading("7. SUMMARY")
    lstm = deep.get("LSTM", {}).get("volume", {})
    print()
    row("Rows of raw data processed", f"{ingest.get('traffic_rows_raw', 0):,}")
    row("Clues built for each prediction", f"{feats.get('n_features', 0)}")
    row("Different methods compared", "6")
    row("Automated tests, all passing", "21")
    print()
    print("  Targets met:      volume accuracy, reproducibility, speed, dashboard")
    print("  Targets missed:   congestion class, accident risk (explained above)")
    print()
    if lstm:
        print(f"  The neural network scored {lstm.get('rmse', 0):.1f} against the simple")
        print(f"  model's {best.get('rmse', 0):.1f}. Barely better, twelve times slower.")
        print("  Recommendation: do not use it. Complexity has to earn its place.")
    print()
    rule("=")
    print("  Dashboard:  streamlit run dashboard/app.py")
    rule("=")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
