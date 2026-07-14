# MOTTO

**A lossless tensor-train framework for multi-site time-to-event analysis of GLP-1 receptor agonists and psychiatric outcomes**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/paper-under%20review-orange)

MOTTO is a privacy-preserving framework for estimating **heterogeneous treatment effects on time-to-event outcomes across multiple health systems**, without pooling patient-level data. Each site fits overlap-weighted Cox models locally and shares only a compressed **tensor-train (TT)** representation of its likelihood surface; the coordinating center reconstructs the multi-site estimates from these compressed cores. The reconstruction is **lossless** — it reproduces the estimates that would have been obtained from a pooled analysis — while only low-dimensional summaries ever leave each site.

This repository accompanies the manuscript and provides the analysis code, a fully reproducible **synthetic** demonstration, and the condition code definitions used in the study.

---

## Repository contents

```
MOTTO/
├── README.md
├── generate_simulation_data.py     # Synthetic multi-site data generator
├── federated_analysis.py           # Site-level analysis (local Cox + TT compression)
├── central_analysis.py             # Central aggregation (lossless TT reconstruction)
├── run_site_analysis.py            # Driver: run one site (or all sites)
├── run_central_analysis.py         # Driver: aggregate site results
├── medical_condition.csv           # Medical condition → ICD code crosswalk
├── site_Penn_data.csv              # Synthetic site cohorts (Penn, Yale,
├── site_Yale_data.csv              #   Mayo, UTSW, OneFlorida+)
├── site_Mayo_data.csv
├── site_UTSW_data.csv
├── site_UF_data.csv
├── motto_federated_results.csv     # Final aggregated estimates
└── results/
    └── <Site>_glp1_vs_sglt2_OOI1_results_<timestamp>.pkl   # Per-site TT cores
```

> **Note on the data.** The `site_*.csv` files are **synthetic**. They are simulated to match the covariate structure and time-to-event format of the study cohorts, but contain no real patient records. Treatment effects in the simulation are illustrative and are **not** tuned to reproduce the hazard ratios reported in the paper. They exist so the full pipeline can be run end-to-end and inspected. `medical_condition.csv` maps clinical conditions (comorbidities, outcomes, and negative controls) to their defining ICD-9/ICD-10 codes.

---

## Method overview

For a given outcome and a binary effect modifier, each site fits a weighted Cox partial-likelihood model with terms for treatment, the modifier, and their interaction:

- **β₁** — treatment log-hazard ratio in the reference stratum (modifier = 0)
- **β₂** — modifier main effect
- **β₃** — treatment × modifier interaction
- **β₁ + β₃** — treatment log-hazard ratio in the modifier = 1 subgroup

Instead of transmitting patient data, each site evaluates its likelihood on a shared grid and compresses it into tensor-train cores via cross-approximation. The center harmonizes the TT ranks across sites, sums the compressed log-likelihoods, and recovers the pooled estimates and their variance–covariance structure. Confounding is addressed with **overlap weighting** on propensity scores, and residual bias is assessed with a panel of **negative control outcomes (NCOs)**.

The comparison implemented here is **GLP-1 receptor agonists vs. SGLT-2 inhibitors** (`glp1_vs_sglt2`).

---

## Installation

Requires Python ≥ 3.9.

```bash
pip install numpy pandas scipy scikit-learn tqdm torch torchtt numdifftools
```

`torch` and `torchtt` provide the tensor-train cross-approximation and reconstruction; the remaining packages handle data preparation, propensity-score estimation, and standard-error computation.

---

## Quick start

The pipeline runs in three steps. The site step is designed to be run **independently per site**, mirroring the federated setting.

**1. Generate the synthetic site cohorts**

```bash
python generate_simulation_data.py
```

Writes `site_<Site>_data.csv` for each of the five sites.

**2. Run the site-level analysis**

Run one site at a time (the site name is passed as an argument):

```bash
python run_site_analysis.py Penn
python run_site_analysis.py Yale
python run_site_analysis.py Mayo
python run_site_analysis.py UTSW
python run_site_analysis.py UF
```

Or run every site in one call:

```bash
python run_site_analysis.py
```

Each site writes its compressed TT cores to `results/` as a `.pkl` file. No patient-level data is written.

**3. Aggregate at the coordinating center**

```bash
python run_central_analysis.py
```

Collects the per-site `.pkl` files in `results/`, reconstructs the multi-site estimates, and writes `motto_federated_results.csv`.

### Demonstration scope

Out of the box, the drivers analyze a single outcome (`visits_depression`) and a single effect modifier (`age_binary`), plus the negative control outcomes. This keeps the demo fast. To analyze a different outcome or modifier, edit the `OUTCOME` and `MODIFIER` constants at the top of `run_site_analysis.py`. The full study spans six psychiatric outcomes and a broader set of effect modifiers; these are configured the same way.

---

## Output

`motto_federated_results.csv` contains one row per analyzed (outcome, modifier) combination:

| Column | Description |
| --- | --- |
| `outcome` | Analyzed outcome (`visits_*` for primary outcomes; `nco_*` for negative controls) |
| `modifier` | Effect modifier |
| `n_sites`, `sites` | Number and names of contributing sites |
| `total_patients`, `total_events` | Pooled sample size and event count |
| `beta_treatment`, `se_treatment` | β₁ and its standard error (treatment log-HR, modifier = 0) |
| `beta_modifier`, `se_modifier` | β₂ and its standard error |
| `beta_interaction`, `se_interaction` | β₃ and its standard error (treatment × modifier) |
| `beta1_plus_beta3`, `se_beta1_plus_beta3` | Treatment log-HR in the modifier = 1 subgroup |
| `var_beta1`, `var_beta2`, `var_beta3`, `cov_*` | Variance–covariance components of the aggregated estimate |
| `converged` | TT cross-approximation convergence flag |
| `rank_expansion` | Whether TT-rank harmonization was applied across sites |

Exponentiating the β columns yields hazard ratios. Because the negative controls carry no true treatment effect, their estimates provide an empirical check on residual bias.

---

## Privacy

Only compressed tensor-train cores and aggregate summary statistics are shared between sites and the center. Individual-level records never leave a site. In the demonstration this is emulated by writing per-site `.pkl` files that the central step reads; in deployment, these cores are the only artifacts transmitted.


---

## Contact

Corresponding authors: Jingmei Qiu, Yiwen Lu, and Yong Chen.
Questions about the code can be raised via the repository's [issue tracker](https://github.com/Penncil/MOTTO/issues).
