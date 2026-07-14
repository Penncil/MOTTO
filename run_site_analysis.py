"""
Step 2 of the MOTTO demo: run the site-level analysis.

Runs ONE site (or all sites) for a single primary outcome and a single effect
modifier. The scope is set by configuring each analyzer instance, so the
analysis code (federated_analysis.py) stays untouched.

Usage:
    python run_site_analysis.py            # run every site in SITES
    python run_site_analysis.py Yale       # run just one site
Or use the per-site wrappers, e.g.:
    python run_site_analysis_yale.py

Each site writes one .pkl per analysed outcome into RESULTS_DIR. No
patient-level data leaves the site step.
"""
import os
import sys
import importlib.util

# --- edit to match your setup ---------------------------------------------- #
SITE_SCRIPT = "federated_analysis.py"
RESULTS_DIR = "results"
# --------------------------------------------------------------------------- #

SITES = ["Penn", "Yale", "Mayo", "UTSW", "UF"]
COMPARISON = "glp1_vs_sglt2"    # matches the synthetic data (GLP-1RA vs SGLT-2i)
GRID_POINTS = 20                # MUST match GRID_POINTS in run_central_analysis.py
WEIGHTING = "overlap"           # overlap weighting, as in the manuscript

# Demonstration scope: one outcome, one effect modifier.
OUTCOME = "visits_depression"   # one primary outcome
MODIFIER = "age_binary"         # one effect modifier


def _load(module_path):
    name = os.path.splitext(os.path.basename(module_path))[0].replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _restrict_scope(analyzer, outcome, modifier):
    """Limit an analyzer instance to one outcome and one modifier (demo scope)."""
    base = outcome.replace("visits_", "")
    analyzer.t2e_outcomes = [outcome]
    analyzer.binary_outcomes = [f"visits_binary_{base}"]
    analyzer.baseline_outcomes = [f"pre_visits_{base}"]
    analyzer.interaction_vars = [modifier]


def run_site(site_name):
    """Run the site-level analysis for a single site."""
    if site_name not in SITES:
        raise SystemExit(f"Unknown site '{site_name}'. Choose from: {', '.join(SITES)}")

    data_file = f"site_{site_name}_data.csv"
    if not os.path.exists(data_file):
        raise SystemExit(f"{data_file} not found. Run generate_simulation_data.py first.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    site_mod = _load(SITE_SCRIPT)

    analyzer = site_mod.SiteTensorTrainAnalysis(
        site_name=site_name,
        comparison_type=COMPARISON,
        grid_points=GRID_POINTS,
        weighting_method=WEIGHTING,
        output_directory=RESULTS_DIR,
    )
    _restrict_scope(analyzer, OUTCOME, MODIFIER)
    analyzer.run_site_complete_analysis(
        data_file=data_file,
        output_dir=RESULTS_DIR,
        site_id=site_name,
    )
    print(f"\nDone: {site_name} -> ./{RESULTS_DIR}/")


def main():
    targets = sys.argv[1:] or SITES
    for site in targets:
        run_site(site)


if __name__ == "__main__":
    main()
