"""
Synthetic data generator for the MOTTO study
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(2025)

# --------------------------------------------------------------------------- #
# Study-level constants                                                        #
# --------------------------------------------------------------------------- #
ENROLL_START = pd.Timestamp("2019-01-01")   # first eligible index date
ENROLL_END = pd.Timestamp("2024-09-30")     # last eligible index date
STUDY_END = pd.Timestamp("2024-12-31")      # administrative censoring date

# Comparison: GLP-1RA (treatment=1) vs SGLT-2i (comparator=0).
# The site code's default comparison_type is "glp1_vs_sglt2".

# Six psychiatric outcomes. Names MUST match the analysis code exactly
# (note the intentional "tobaco" spelling used throughout the pipeline).
PSYCH_OUTCOMES = [
    "anxiety_disorders",
    "depression",
    "bipolar_disorder",
    "schizophrenia_and_other_psychotic_disorders",
    "alcohol_use_disorder",
    "tobaco_use_disorder",
]

# Negative control outcomes. The main text references Supplementary Table S4
N_NCO = 6

# Per-outcome baseline daily hazard (chosen only to yield sensible event
# counts so every outcome x modifier analysis has events to fit).
PSYCH_BASE_HAZARD = {
    "anxiety_disorders": 5.0e-4,
    "depression": 4.5e-4,
    "bipolar_disorder": 1.2e-4,
    "schizophrenia_and_other_psychotic_disorders": 8.0e-5,
    "alcohol_use_disorder": 2.0e-4,
    "tobaco_use_disorder": 3.0e-4,
}
NCO_BASE_HAZARD = 2.5e-4  # shared baseline hazard for negative controls

# Nominal (log-hazard) treatment effect and a couple of effect-modifier
# interactions. Purely illustrative; results are not interpreted.
LOGHR_TREATMENT = 0.10
LOGHR_TRT_X_AGE65 = -0.12   # slightly lower in >=65
LOGHR_TRT_X_INSULIN = -0.15  # slightly lower with baseline insulin

# --------------------------------------------------------------------------- #
# Site configuration                                                           #
# --------------------------------------------------------------------------- #
# Fraction of the manuscript cohort sizes to generate. Set to 1.0 for the full
# sizes; 0.1 gives a ~10x-smaller, faster-to-analyse demonstration cohort.
COHORT_SCALE = 0.10

# Per-site cohort sizes (GLP-1RA and SGLT-2i arms), at full scale.
SITE_CONFIG = {
    "Penn": {"n_glp1": 10292, "n_sglt2": 11479},
    "Yale": {"n_glp1": 14898, "n_sglt2": 11154},
    "Mayo": {"n_glp1": 18999, "n_sglt2": 11439},
    "UTSW": {"n_glp1": 5488, "n_sglt2": 3961},
    "UF": {"n_glp1": 16000, "n_sglt2": 12000},  # OneFlorida+ (synthetic sizes)
}

# Site- and arm-specific demographic distributions.
# race_pct order: [White, Black or African American, Asian, Other].
SITE_PARAMS = {
    "Penn": {
        "age_mean": {"glp1": 57.48, "sglt2": 64.09}, "age_sd": {"glp1": 12.74, "sglt2": 12.04},
        "male_pct": {"glp1": 0.574, "sglt2": 0.400},
        "race_pct": {"glp1": [0.473, 0.412, 0.033, 0.082], "sglt2": [0.546, 0.295, 0.056, 0.103]},
        "hispanic_pct": {"glp1": 0.051, "sglt2": 0.053},
        "insulin_pct": {"glp1": 0.304, "sglt2": 0.296},
        "metformin_pct": {"glp1": 0.602, "sglt2": 0.586},
    },
    "Yale": {
        "age_mean": {"glp1": 56.72, "sglt2": 64.14}, "age_sd": {"glp1": 13.64, "sglt2": 13.18},
        "male_pct": {"glp1": 0.623, "sglt2": 0.413},
        "race_pct": {"glp1": [0.604, 0.221, 0.021, 0.154], "sglt2": [0.620, 0.199, 0.033, 0.148]},
        "hispanic_pct": {"glp1": 0.213, "sglt2": 0.198},
        "insulin_pct": {"glp1": 0.292, "sglt2": 0.346},
        "metformin_pct": {"glp1": 0.510, "sglt2": 0.565},
    },
    "Mayo": {
        "age_mean": {"glp1": 59.72, "sglt2": 66.37}, "age_sd": {"glp1": 12.75, "sglt2": 12.44},
        "male_pct": {"glp1": 0.520, "sglt2": 0.355},
        "race_pct": {"glp1": [0.910, 0.041, 0.025, 0.024], "sglt2": [0.906, 0.038, 0.035, 0.021]},
        "hispanic_pct": {"glp1": 0.057, "sglt2": 0.046},
        "insulin_pct": {"glp1": 0.25, "sglt2": 0.30},
        "metformin_pct": {"glp1": 0.50, "sglt2": 0.55},
    },
    "UTSW": {
        "age_mean": {"glp1": 57.97, "sglt2": 63.63}, "age_sd": {"glp1": 13.27, "sglt2": 12.90},
        "male_pct": {"glp1": 0.615, "sglt2": 0.441},
        "race_pct": {"glp1": [0.595, 0.260, 0.046, 0.099], "sglt2": [0.573, 0.248, 0.072, 0.107]},
        "hispanic_pct": {"glp1": 0.196, "sglt2": 0.214},
        "insulin_pct": {"glp1": 0.228, "sglt2": 0.299},
        "metformin_pct": {"glp1": 0.372, "sglt2": 0.381},
    },
    "UF": {  # OneFlorida+: larger, more ethnically diverse (synthetic)
        "age_mean": {"glp1": 58.10, "sglt2": 64.50}, "age_sd": {"glp1": 13.10, "sglt2": 12.60},
        "male_pct": {"glp1": 0.560, "sglt2": 0.420},
        "race_pct": {"glp1": [0.520, 0.300, 0.040, 0.140], "sglt2": [0.540, 0.280, 0.050, 0.130]},
        "hispanic_pct": {"glp1": 0.270, "sglt2": 0.250},
        "insulin_pct": {"glp1": 0.270, "sglt2": 0.310},
        "metformin_pct": {"glp1": 0.540, "sglt2": 0.560},
    },
}

# Baseline comorbidity / medication prevalences (shared across sites; kept
# non-trivial so every binary effect modifier has both levels present).
COMORB_PREV = {
    "medical_chronic_kidney_disease": 0.18,
    "medical_coronary_atherosclerosis_and_other_heart_disease": 0.22,
    "medical_cerebrovascular_disease": 0.10,
    "medical_hypertension": 0.55,
    "medical_obesity": 0.38,
}
MED_PREV = {
    "med_lipid": 0.52,
    "med_antidepressants": 0.20,
    "med_antipsychotics": 0.06,
    "med_opioids": 0.15,
    "med_arb": 0.24,
    "med_beta_blockers": 0.20,
    "med_diuretics": 0.22,
}


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #
def _arm_concat(fn_glp1, fn_sglt2):
    """Concatenate GLP-1 arm draws followed by SGLT-2 arm draws."""
    return np.concatenate([fn_glp1(), fn_sglt2()])


def _draw_demographics(params, n_glp1, n_sglt2):
    """Age, gender, race, ethnicity for both arms (GLP-1 first)."""
    age = _arm_concat(
        lambda: np.clip(RNG.normal(params["age_mean"]["glp1"], params["age_sd"]["glp1"], n_glp1), 18, 95),
        lambda: np.clip(RNG.normal(params["age_mean"]["sglt2"], params["age_sd"]["sglt2"], n_sglt2), 18, 95),
    )
    gender = _arm_concat(
        lambda: RNG.choice(["Male", "Female"], n_glp1, p=[params["male_pct"]["glp1"], 1 - params["male_pct"]["glp1"]]),
        lambda: RNG.choice(["Male", "Female"], n_sglt2, p=[params["male_pct"]["sglt2"], 1 - params["male_pct"]["sglt2"]]),
    )
    race_labels = ["White", "Black or African American", "Asian", "Other"]
    race = _arm_concat(
        lambda: RNG.choice(race_labels, n_glp1, p=params["race_pct"]["glp1"]),
        lambda: RNG.choice(race_labels, n_sglt2, p=params["race_pct"]["sglt2"]),
    )
    ethn_labels = ["Hispanic or Latino", "Not Hispanic or Latino"]
    ethnicity = _arm_concat(
        lambda: RNG.choice(ethn_labels, n_glp1, p=[params["hispanic_pct"]["glp1"], 1 - params["hispanic_pct"]["glp1"]]),
        lambda: RNG.choice(ethn_labels, n_sglt2, p=[params["hispanic_pct"]["sglt2"], 1 - params["hispanic_pct"]["sglt2"]]),
    )
    return age, gender, race, ethnicity


def _draw_medications(params, n_glp1, n_sglt2, n_total):
    """Baseline medication indicators (arm-specific insulin/metformin)."""
    meds = {}
    meds["med_insulin"] = _arm_concat(
        lambda: RNG.binomial(1, params["insulin_pct"]["glp1"], n_glp1),
        lambda: RNG.binomial(1, params["insulin_pct"]["sglt2"], n_sglt2),
    )
    meds["med_metformin"] = _arm_concat(
        lambda: RNG.binomial(1, params["metformin_pct"]["glp1"], n_glp1),
        lambda: RNG.binomial(1, params["metformin_pct"]["sglt2"], n_sglt2),
    )
    for name, prev in MED_PREV.items():
        meds[name] = RNG.binomial(1, prev, n_total)
    return meds


def _draw_comorbidities(n_total):
    return {name: RNG.binomial(1, prev, n_total) for name, prev in COMORB_PREV.items()}


def _draw_labs(n_total):
    """Continuous labs; the site code derives binary versions internally."""
    return {
        "lab_bmi": RNG.normal(32, 6, n_total),
        "lab_sbp": RNG.normal(130, 15, n_total),
        "lab_dbp": RNG.normal(80, 10, n_total),
        "lab_hba1c": RNG.normal(8.0, 1.5, n_total),
        "lab_hdl": RNG.normal(45, 12, n_total),
        "lab_ldl": RNG.normal(100, 30, n_total),
        "lab_total_cholesterol": RNG.normal(180, 40, n_total),
        "lab_triglycerides": RNG.normal(150, 60, n_total),
    }


def _draw_utilization(n_total):
    return {
        "ed_visits": RNG.poisson(0.8, n_total),
        "inpatient_visits": RNG.poisson(0.3, n_total),
        "outpatient_visits": RNG.poisson(4.0, n_total),
    }


def _draw_index_dates(n_total):
    """Random index (entry) date within the enrollment window."""
    span_days = (ENROLL_END - ENROLL_START).days
    offsets = RNG.integers(0, span_days + 1, n_total)
    entry_date = ENROLL_START + pd.to_timedelta(offsets, unit="D")
    return pd.Series(entry_date).reset_index(drop=True)


def _draw_last_record(entry_date):
    """
    Observed last-contact date per patient: index date plus a random
    follow-up length, administratively censored at STUDY_END. Always at
    least one day after entry so follow-up time is positive.
    """
    loss_days = RNG.exponential(900, len(entry_date)).astype(int)
    loss_days = np.clip(loss_days, 1, 20000)  # keep well within datetime range
    candidate = entry_date + pd.to_timedelta(loss_days, unit="D")
    last_record = candidate.where(candidate <= STUDY_END, STUDY_END)
    too_short = last_record <= entry_date
    last_record = last_record.where(~too_short, entry_date + pd.Timedelta(days=1))
    return last_record


def _draw_time_to_event(base_hazard, log_hazard_ratio, entry_date, last_record):
    """
    Generate a time-to-event outcome under an exponential hazard model.

    Returns (event_date, event_indicator) where event_date is NaT for
    censored patients. Events are constrained to occur on or before the
    patient's last-contact date.
    """
    n = len(entry_date)
    rate = base_hazard * np.exp(log_hazard_ratio)
    latent_days = RNG.exponential(1.0 / rate, n)
    followup_days = (last_record - entry_date).dt.days.to_numpy()

    event = latent_days <= followup_days
    # Non-event latent times can be astronomically large for rare outcomes;
    # zero their offset before the date addition to avoid int64 datetime
    # overflow. Event offsets are bounded by follow-up (<= a few years).
    event_days = np.where(event, np.ceil(latent_days), 0.0)

    event_date = entry_date + pd.to_timedelta(event_days, unit="D")
    event_date = event_date.where(event, pd.NaT)
    return event_date, event.astype(int)


# --------------------------------------------------------------------------- #
# Site assembly                                                                #
# --------------------------------------------------------------------------- #
def generate_site_data(site_name):
    """Assemble the full synthetic cohort for one site."""
    config, params = SITE_CONFIG[site_name], SITE_PARAMS[site_name]
    n_glp1 = max(1, int(round(config["n_glp1"] * COHORT_SCALE)))
    n_sglt2 = max(1, int(round(config["n_sglt2"] * COHORT_SCALE)))
    n_total = n_glp1 + n_sglt2

    treatment = np.concatenate([np.ones(n_glp1, int), np.zeros(n_sglt2, int)])
    drug_class = np.where(treatment == 1, "GLP1", "SGLT2")

    age, gender, race, ethnicity = _draw_demographics(params, n_glp1, n_sglt2)
    entry_date = _draw_index_dates(n_total)
    last_record = _draw_last_record(entry_date)

    columns = {
        "drug_class": drug_class,
        "age_at_entry": age,
        "gender": gender,
        "race": race,
        "ethnicity": ethnicity,
        "entry_date": entry_date.dt.strftime("%Y-%m-%d").to_numpy(),
        "entry_year": entry_date.dt.year.to_numpy(),
        "last_record": last_record.dt.strftime("%Y-%m-%d").to_numpy(),
        "pre_visits_sleep_disorders": RNG.binomial(1, 0.10, n_total),
    }
    columns.update(_draw_utilization(n_total))
    columns.update(_draw_comorbidities(n_total))
    columns.update(_draw_medications(params, n_glp1, n_sglt2, n_total))
    columns.update(_draw_labs(n_total))

    # Effect-modifier signals baked into the psychiatric hazards.
    age65 = (age >= 65).astype(int)
    insulin = columns["med_insulin"]

    for outcome in PSYCH_OUTCOMES:
        loghr = (
            LOGHR_TREATMENT * treatment
            + LOGHR_TRT_X_AGE65 * treatment * age65
            + LOGHR_TRT_X_INSULIN * treatment * insulin
        )
        event_date, event = _draw_time_to_event(
            PSYCH_BASE_HAZARD[outcome], loghr, entry_date, last_record
        )
        columns[f"visits_{outcome}"] = event_date.dt.strftime("%Y-%m-%d").to_numpy()
        columns[f"visits_binary_{outcome}"] = event

    # Negative control outcomes (true null: no treatment effect).
    for k in range(1, N_NCO + 1):
        event_date, event = _draw_time_to_event(
            NCO_BASE_HAZARD, np.zeros(n_total), entry_date, last_record
        )
        columns[f"nco_{k}"] = event_date.dt.strftime("%Y-%m-%d").to_numpy()
        columns[f"nco_binary_{k}"] = event

    return pd.DataFrame(columns)


def main():
    for site_name in SITE_CONFIG:
        df = generate_site_data(site_name)
        output_file = f"site_{site_name}_data.csv"
        df.to_csv(output_file, index=False)
        n_glp1 = int((df["drug_class"] == "GLP1").sum())
        n_sglt2 = int((df["drug_class"] == "SGLT2").sum())
        print(f"{site_name}: N={len(df)}, GLP1={n_glp1}, SGLT2={n_sglt2}, cols={df.shape[1]}")


if __name__ == "__main__":
    main()
