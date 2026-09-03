"""FlowCast — end-to-end pipeline runner.

The one command a reviewer needs:

    python src/run_pipeline.py

runs ingestion, cleaning, feature engineering, EDA and reporting, the classical
ML engine, the deep-learning engine and the prediction-interval fit, in order,
from the raw CSVs in data/raw/ to a dashboard-ready set of artefacts.

Stages can be run individually or resumed:

    python src/run_pipeline.py --stages data          # M1-M4 only
    python src/run_pipeline.py --stages models        # M5, M6, intervals
    python src/run_pipeline.py --skip deep            # everything but the LSTM
    python src/run_pipeline.py --input path/to.csv    # score a new export

Every stage is idempotent and writes its own artefacts, so a failure part way
through does not require starting over.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import get_logger, load_config, save_json, set_seed  # noqa: E402

STAGE_GROUPS = {
    "data": ["ingest", "clean", "features", "eda"],
    "models": ["classical", "deep", "intervals"],
    "all": ["ingest", "clean", "features", "eda", "classical", "deep", "intervals"],
}


def _run_stage(name: str, cfg: dict, log) -> dict:
    """Import and execute one stage, timing it."""
    t0 = time.time()
    log.info("")
    log.info("#" * 68)
    log.info("# STAGE: %s", name.upper())
    log.info("#" * 68)

    if name == "ingest":
        import ingest
        result = ingest.run(cfg)
    elif name == "clean":
        import clean
        result = clean.run(cfg)
    elif name == "features":
        import features
        result = features.run(cfg)
    elif name == "eda":
        import eda
        result = eda.run(cfg)
    elif name == "classical":
        import ml_models
        result = ml_models.run(cfg)
    elif name == "deep":
        import dl_model
        result = dl_model.run(cfg)
    elif name == "intervals":
        import uncertainty
        result = uncertainty.run(cfg)
    else:
        raise ValueError(f"Unknown stage: {name}")

    elapsed = time.time() - t0
    log.info("STAGE %s complete in %.1f s", name.upper(), elapsed)
    return {"stage": name, "seconds": round(elapsed, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the FlowCast pipeline end to end.")
    parser.add_argument("--stages", default="all", choices=sorted(STAGE_GROUPS),
                        help="Which group of stages to run (default: all).")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Stage names to skip, e.g. --skip deep.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run exactly these stages, ignoring --stages.")
    parser.add_argument("--input", default=None,
                        help="Path to a replacement traffic_sensor_log CSV. It is "
                             "copied into data/raw/ before the run.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log = get_logger("pipeline", cfg)
    set_seed(cfg["project"]["random_seed"])

    log.info("FlowCast v%s — %s corridor — seed %d",
             cfg["project"]["version"], cfg["project"]["corridor"],
             cfg["project"]["random_seed"])

    if args.input:
        src = Path(args.input)
        if not src.exists():
            log.error("Input file not found: %s", src)
            return 2
        dst = Path(cfg["paths"]["raw"]) / cfg["data"]["traffic_file"]
        # The raw directory is meant to be read-only in normal operation, so a
        # replacement is backed up rather than overwritten in place.
        if dst.exists():
            backup = dst.with_suffix(f".backup-{int(time.time())}.csv")
            shutil.copy2(dst, backup)
            log.info("backed up existing raw file to %s", backup.name)
        shutil.copy2(src, dst)
        log.info("using %s as the traffic sensor log", src.name)

    stages = args.only if args.only else STAGE_GROUPS[args.stages]
    stages = [s for s in stages if s not in args.skip]
    log.info("stages to run: %s", " -> ".join(stages))

    timings, t_start = [], time.time()
    for stage in stages:
        try:
            timings.append(_run_stage(stage, cfg, log))
        except Exception:
            log.exception("STAGE %s FAILED — stopping. Earlier artefacts are "
                          "intact; fix the cause and rerun with --only %s",
                          stage.upper(), stage)
            return 1

    total = time.time() - t_start
    log.info("")
    log.info("=" * 68)
    log.info("PIPELINE COMPLETE in %.1f s (%.1f min)", total, total / 60)
    for t in timings:
        log.info("  %-12s %7.1f s", t["stage"], t["seconds"])
    log.info("=" * 68)
    log.info("Next: streamlit run dashboard/app.py")

    save_json({"stages": timings, "total_seconds": round(total, 1)},
              Path(cfg["paths"]["reports"]) / "pipeline_timings.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
