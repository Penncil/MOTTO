"""
Step 3 of the MOTTO demo: aggregate the site results at the coordinating center.

Reads the per-site tensor-train cores written by run_site_analysis.py, combines
them into pooled-equivalent estimates, and writes a results CSV.

Note on file discovery: the central script's automatic directory search expects
filenames like "*_modifier*_results_*.pkl", but the site step writes
"*_OOI*_results_*.pkl". The aggregation itself buckets results by
(outcome, modifier) read from the pickle CONTENTS, not the filename, so we simply
collect the site files ourselves and pass them via pkl_file_paths.

Run AFTER run_site_analysis.py:

    python run_central_analysis.py
"""
import os
import glob
import importlib.util

# --- edit these to match your setup ---------------------------------------- #
CENTRAL_SCRIPT = "central_analysis.py"
RESULTS_DIR = "results"
OUTPUT_CSV = "motto_federated_results.csv"
# --------------------------------------------------------------------------- #

COMPARISON = "glp1_vs_sglt2"   # must match run_site_analysis.py
GRID_POINTS = 20               # MUST match GRID_POINTS in run_site_analysis.py
MIN_SITES_REQUIRED = 2         # a combination is aggregated if >= this many sites have it


def _load(module_path):
    name = os.path.splitext(os.path.basename(module_path))[0].replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    central_mod = _load(CENTRAL_SCRIPT)

    pkl_files = sorted(glob.glob(
        os.path.join(RESULTS_DIR, "**", f"*_{COMPARISON}_OOI*_results_*.pkl"),
        recursive=True,
    ))
    if not pkl_files:
        raise SystemExit(f"No site result files found under ./{RESULTS_DIR}/ "
                         f"(expected *_{COMPARISON}_OOI*_results_*.pkl). "
                         f"Run run_site_analysis.py first.")
    print(f"Found {len(pkl_files)} site result files")

    central = central_mod.CheckpointCentralTensorTrainAnalysis(
        n_strata=5, grid_points=GRID_POINTS,
    )
    central.run_checkpoint_analysis(
        pkl_file_paths=pkl_files,
        comparison_type=COMPARISON,
        min_sites_required=MIN_SITES_REQUIRED,
        analyze_all=True,
        output_file=OUTPUT_CSV,
    )
    print(f"\nDone. Aggregated results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
