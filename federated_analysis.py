import numpy as np
import pandas as pd
from scipy.optimize import minimize, root_scalar
import torch
import torchtt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
import os
import warnings
import numdifftools as nd  # For Hessian-based SE calculation
from datetime import datetime
warnings.filterwarnings('ignore')

# --------------------- append-friendly pickle helper ---------------------
import pickle

def _append_pickle(obj, filepath: str):
    """Append a single Python object to a pickle stream file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    with open(filepath, "ab") as f:  # append-binary
        pickle.dump(obj, f)
# -------------------------------------------------------------------------


class SiteTensorTrainAnalysis:
    def __init__(self, site_name, comparison_type="glp1_vs_sglt2", grid_points=20, weighting_method="ipw", output_directory="./"):
        """
        Site-level Tensor Train Analysis - FOR EACH PARTICIPATING SITE
        
        This code should be run at each site to generate aggregated summary statistics
        WITHOUT sharing patient-level data.
        
        Parameters:
        -----------
        site_name : str
            Name/ID of the site (e.g., "site_1", "hospital_A")
        comparison_type : str
            Type of treatment comparison: "glp1_vs_sglt2" or "glp1_vs_dpp4"
        grid_points : int
            Number of grid points for tensor train approximation (35 for production)
        weighting_method : str
            Weighting method for propensity score adjustment: "ipw" or "overlap"
        """
        self.site_name = site_name
        self.comparison_type = comparison_type
        self.weighting_method = weighting_method.lower()
        if self.weighting_method not in ["ipw", "overlap"]:
            raise ValueError(f"Unknown weighting_method: {self.weighting_method}. Must be 'ipw' or 'overlap'.")
        self.output_directory = output_directory
        self.M = grid_points
        self.d = 3  # Number of parameters (treatment, interaction_var, treatment*interaction_var)
        self.lower_a = -5
        self.upper_b = 5
        self.N = [self.M + 1] * self.d
        
        # Timing dictionary to track computation times
        self.timing_log = {}
        
        # Primary outcomes of interest (IIO - Interventions of Interest Outcomes)
        self.t2e_outcomes = ["visits_anxiety_disorders", "visits_depression", "visits_bipolar_disorder",
                             "visits_schizophrenia_and_other_psychotic_disorders",
                             "visits_alcohol_use_disorder","visits_tobaco_use_disorder"]
        self.binary_outcomes = ["visits_binary_anxiety_disorders", "visits_binary_depression", "visits_binary_bipolar_disorder",
                                "visits_binary_schizophrenia_and_other_psychotic_disorders",
                                "visits_binary_alcohol_use_disorder","visits_binary_tobaco_use_disorder"]
        self.baseline_outcomes = ["pre_visits_anxiety_disorders","pre_visits_depression", "pre_visits_bipolar_disorder",
                                  "pre_visits_schizophrenia_and_other_psychotic_disorders",
                                  "pre_visits_alcohol_use_disorder","pre_visits_tobaco_use_disorder"]
  
        # Negative Control Outcomes (NCOs)
        self.nco_outcomes = []
        
 
        # Interaction/modifier variables
        self.interaction_vars = [
            "ethnicity_binary", 
            "age_binary",  # <65 (0) vs ≥65 (1)
            "gender_binary",  # Female (0) vs Male (1)
            # Medical history
            "medical_chronic_kidney_disease",
            "medical_coronary_atherosclerosis_and_other_heart_disease",
            "medical_cerebrovascular_disease",
            "medical_hypertension",
            "medical_obesity",
            # Medication history
            "med_insulin",
            "med_metformin",
            "med_lipid",
            "med_antidepressants",
            "med_antipsychotics",
            "med_opioids",
            # Labs (binary)
            "lab_hba1c_binary",
            "lab_bmi_binary",
            "lab_bp_control_binary",
            # Others
            "pre_visits_sleep_disorders",
            "any_aht_medications"
        ]
        
        # Propensity score covariates - will be dynamically populated
        self.prop_vars = []

        # ============ SPECIFIED LAB VARIABLES ONLY ============
        # Only use these specific lab variables for propensity score
        self.allowed_lab_vars = [
            'lab_bmi', 
            'lab_sbp', 
            'lab_dbp', 
            'lab_hba1c',  # Note: lowercase for consistency
            'lab_HbA1c',  # Handle case variations
            'lab_hdl',     # Note: lowercase for consistency
            'lab_HDL',     # Handle case variations
            'lab_ldl',     # Note: lowercase for consistency
            'lab_LDL',     # Handle case variations
            'lab_total_cholesterol', 
            'lab_triglycerides',  # Note: lowercase for consistency
            'lab_Triglycerides'   # Handle case variations
        ]

        
        # Antihypertensive medications
        self.aht_medications = ["med_arb", "med_beta_blockers", "med_diuretics"]

    def _load_dynamic_prop_vars(self, df):
        """Dynamically load propensity score variables based on actual columns in the data"""
        
        # Base variables that should exist
        base_vars = [
            "age_at_entry",
            "gender",
            "race_category",  # Will be created from race
            "ethnicity",  # Keep original ethnicity (Hispanic/Non-Hispanic only)
            "ed_visits",
            "inpatient_visits",
            "outpatient_visits",
            "pre_visits_sleep_disorders"
        ]
        
        # Add entry_year if it exists or can be created
        if 'entry_year' in df.columns:
            base_vars.append('entry_year')
        elif 'entry_date' in df.columns:
            df['entry_year'] = pd.to_datetime(df['entry_date']).dt.year
            base_vars.append('entry_year')
        
        # Dynamically get medical and medication columns
        medical_cols = [col for col in df.columns if col.startswith('medical_')]
        med_cols = [col for col in df.columns if col.startswith('med_')]
        
        # ============ RESTRICTED LAB VARIABLES ============
        # Only include the specified lab variables that exist in the dataframe
        # Handle case variations (HbA1c vs hba1c, HDL vs hdl, etc.)
        lab_cols = []
        for lab_var in self.allowed_lab_vars:
            if lab_var in df.columns:
                lab_cols.append(lab_var)
            # Also check for case variations
            elif lab_var.replace('hba1c', 'HbA1c') in df.columns:
                lab_cols.append(lab_var.replace('hba1c', 'HbA1c'))
            elif lab_var.replace('hdl', 'HDL') in df.columns:
                lab_cols.append(lab_var.replace('hdl', 'HDL'))
            elif lab_var.replace('ldl', 'LDL') in df.columns:
                lab_cols.append(lab_var.replace('ldl', 'LDL'))
            elif lab_var.replace('triglycerides', 'Triglycerides') in df.columns:
                lab_cols.append(lab_var.replace('triglycerides', 'Triglycerides'))
        
        # Combine all and remove duplicates
        all_vars = base_vars + medical_cols + med_cols + lab_cols
        self.prop_vars = list(set(all_vars))
        
        # Filter to only include columns that actually exist in the dataframe
        self.prop_vars = [var for var in self.prop_vars if var in df.columns]
        
        print(f"  Loaded {len(self.prop_vars)} propensity score variables dynamically")
        print(f"    - Medical variables: {len(medical_cols)}")
        print(f"    - Medication variables: {len(med_cols)}")
        print(f"    - Lab variables: {len(lab_cols)} (restricted to: {', '.join(lab_cols)})")
        print(f"    - Base variables: {len([v for v in base_vars if v in df.columns])}")
    
    def load_and_preprocess_data(self, data_file, output_dir=None):
        """
        Load and preprocess data from THIS site only
        """
        start_time = time.time()
        
        if not os.path.exists(data_file):
            print(f"ERROR: Data file {data_file} not found")
            return None
            
        print(f"Loading {self.site_name} data from {data_file}")
        df = pd.read_csv(data_file)
        
        df = self._preprocess_site_data(df, output_dir=output_dir)
        
        if df is not None:
            load_time = time.time() - start_time
            self.timing_log['data_loading'] = load_time
            print(f"{self.site_name}: {len(df)} patients remaining for analysis (loaded in {load_time:.2f} seconds)")
        
        return df
    
    def _preprocess_site_data(self, df, output_dir=None):
        """Preprocess individual site data with ALL interaction variable encoding"""
        start_time = time.time()

        # Normalize columns
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # Remove Unknown gender and ethnicity at the beginning
        initial_count = len(df)
        
        # Remove Unknown/X gender
        if 'gender' in df.columns:
            unknown_gender_count = df['gender'].isin(['Unknown', 'X']).sum()
            df = df[~df['gender'].isin(['Unknown', 'X'])].copy()
            unknown_gender_pct = unknown_gender_count / initial_count * 100
            print(f"  Removed {unknown_gender_count} patients with Unknown/X gender ({unknown_gender_pct:.2f}%)")
        
        # Remove Unknown ethnicity
        if 'ethnicity' in df.columns:
            unknown_ethnicity_count = (df['ethnicity'] == 'Unknown').sum()
            df = df[df['ethnicity'] != 'Unknown'].copy()
            unknown_ethnicity_pct = unknown_ethnicity_count / initial_count * 100
            print(f"  Removed {unknown_ethnicity_count} patients with Unknown ethnicity ({unknown_ethnicity_pct:.2f}%)")
        
        total_removed = initial_count - len(df)
        print(f"  Total removed: {total_removed} patients ({total_removed/initial_count*100:.2f}%)")
        print(f"  Remaining for analysis: {len(df)} patients ({len(df)/initial_count*100:.2f}%)")

        # Create treatment per comparison
        if self.comparison_type == "glp1_vs_sglt2":
            df = df[df['drug_class'].str.lower().isin(['glp1', 'sglt2'])].copy()
            df['treatment'] = (df['drug_class'].str.lower() == 'glp1').astype(int)
            print(f"  GLP1 vs SGLT2 comparison: GLP1 = 1 ({np.sum(df['treatment'] == 1)}), "
                  f"SGLT2 = 0 ({np.sum(df['treatment'] == 0)})")
        elif self.comparison_type == "glp1_vs_dpp4":
            df = df[df['drug_class'].str.lower().isin(['glp1', 'dpp4'])].copy()
            df['treatment'] = (df['drug_class'].str.lower() == 'glp1').astype(int)
            print(f"  GLP1 vs DPP4 comparison: GLP1 = 1 ({np.sum(df['treatment'] == 1)}), "
                  f"DPP4 = 0 ({np.sum(df['treatment'] == 0)})")
        else:
            raise ValueError(f"Unknown comparison_type: {self.comparison_type}")

        if len(df) == 0:
            print(f"  ERROR: No patients found for {self.comparison_type} comparison")
            return None

        # Create race categories for propensity score
        df = self._create_race_categories(df)

        # Interaction variables
        df = self._create_interaction_variables(df)

        # Load propensity score variables dynamically
        self._load_dynamic_prop_vars(df)

        # Labs to binary
        df = self._process_lab_results(df)
        
        # Compute last_record_date for censoring
        df = self._compute_last_record_date(df)

        # Time-to-event conversion
        df = self._convert_date_outcomes_to_numeric(df)

        # Note: All column names are lowercase after normalization (line 187)
        print("  Detecting negative control outcomes...")
        
        # Find original NCO date columns (nco_* but NOT nco_binary_*, pre_nco_, or converted _time/_event columns)
        time_nco_cols = []
        for col in df.columns:
            if col.startswith('nco_') and not col.startswith('nco_binary_') and not col.startswith('pre_nco_'):
                # Exclude already-converted columns (those ending with _time or _event)
                if not (col.endswith('_time') or col.endswith('_event')):
                    time_nco_cols.append(col)
        
        # For each original NCO date column, verify corresponding binary column exists
        valid_ncos = []
        for time_col in time_nco_cols:
            # Extract NCO name: "nco_cachexia" -> "cachexia"
            nco_name = time_col.replace('nco_', '')
            binary_col = f'nco_binary_{nco_name}'
            
            if binary_col in df.columns:
                # This NCO has both time-to-event date and binary indicator
                valid_ncos.append(time_col)
            else:
                print(f"    ⚠️ Skipping {time_col}: missing {binary_col}")
        
        if valid_ncos and not self.nco_outcomes:
            self.nco_outcomes = valid_ncos
            print(f"  ✓ Detected {len(self.nco_outcomes)} valid NCOs (with both time-to-event and binary)")
            print(f"    Example NCOs: {self.nco_outcomes[:3]}")
        elif not valid_ncos:
            print(f"  ⚠️ No valid NCOs found (need both nco_* date and nco_binary_* columns)")
            self.nco_outcomes = []
        else:
            print(f"  ✓ Using {len(self.nco_outcomes)} NCOs from configuration")

        preprocess_time = time.time() - start_time
        self.timing_log['preprocessing'] = preprocess_time
        print(f"  Preprocessing completed in {preprocess_time:.2f} seconds")

        # Table 1 style printouts
        self._report_table1_statistics(df, output_dir=output_dir)

        return df

    def _create_race_categories(self, df):
        """Create race categories for propensity score"""
        print("  Creating race categories for propensity score...")
        
        if 'race' in df.columns:
            # Create race_category with 4 levels
            df['race_category'] = 'Others'  # Default
            
            # Map specific races
            df.loc[df['race'] == 'White', 'race_category'] = 'White'
            df.loc[df['race'] == 'Black or African American', 'race_category'] = 'Black'
            df.loc[df['race'] == 'Asian', 'race_category'] = 'Asian'
            # All others → Others
            
            # Print distribution
            race_counts = df['race_category'].value_counts()
            total = len(df)
            print(f"    Race category distribution:")
            for race in ['White', 'Black', 'Asian', 'Others']:
                count = race_counts.get(race, 0)
                print(f"      {race}: {count} ({count/total*100:.1f}%)")
        
        return df


    def _create_interaction_variables(self, df):
        """Create all interaction variables"""
        print("  Creating interaction variables...")

        # Create ethnicity_binary
        if 'ethnicity' in df.columns:
            df['ethnicity_binary'] = (df['ethnicity'] == 'Not Hispanic or Latino').astype(int)
            hispanic_count = np.sum(df['ethnicity_binary'] == 0)
            non_hispanic_count = np.sum(df['ethnicity_binary'] == 1)
            print(f"    ethnicity_binary: Hispanic={hispanic_count}, Non-Hispanic={non_hispanic_count}")
        else:
            df['ethnicity_binary'] = 1
            print(f"    ethnicity_binary: No ethnicity column found, all coded as Non-Hispanic")

        if 'age_at_entry' in df.columns:
            df['age_binary'] = (df['age_at_entry'] >= 65).astype(int)
            print(f"    age_binary: <65={np.sum(df['age_binary']==0)}, ≥65={np.sum(df['age_binary']==1)}")

        if 'gender' in df.columns:
            df['gender_binary'] = (df['gender'] == 'Male').astype(int)
            print(f"    gender_binary: Female={np.sum(df['gender_binary']==0)}, Male={np.sum(df['gender_binary']==1)}")


        # Composite AHT meds
        bp_med_cols = [c for c in self.aht_medications if c in df.columns]
        if bp_med_cols:
            df['any_aht_medications'] = (df[bp_med_cols].sum(axis=1) > 0).astype(int)
        else:
            df['any_aht_medications'] = 0
        print(f"    any_aht_medications: {np.sum(df['any_aht_medications']==1)}")

        medical_vars = [
            "medical_chronic_kidney_disease",
            "medical_coronary_atherosclerosis_and_other_heart_disease",
            "medical_cerebrovascular_disease",
            "medical_hypertension",
            "medical_obesity"
        ]
        medication_vars = ["med_insulin", "med_metformin", "med_lipid", 
                          "med_antidepressants", "med_antipsychotics", "med_opioids"]

        for var in medical_vars + medication_vars:
            if var in df.columns:
                df[var] = df[var].astype(int)
            else:
                df[var] = 0

        if 'pre_visits_sleep_disorders' in df.columns:
            df['pre_visits_sleep_disorders'] = df['pre_visits_sleep_disorders'].astype(int)
        else:
            df['pre_visits_sleep_disorders'] = 0

        return df

    def _process_lab_results(self, df):
        """Process lab results into binary vars"""
        print("  Processing lab results...")

        lab_specs = {}
        
        if 'lab_hba1c' in df.columns:
            lab_specs['lab_hba1c'] = {'threshold': 8.5, 'name': 'lab_hba1c_binary'}
        elif 'lab_HbA1c' in df.columns:
            lab_specs['lab_HbA1c'] = {'threshold': 8.5, 'name': 'lab_hba1c_binary'}
            
        if 'lab_bmi' in df.columns:
            lab_specs['lab_bmi'] = {'threshold': 30.0, 'name': 'lab_bmi_binary'}

        for lab_var, specs in lab_specs.items():
            if lab_var in df.columns:
                df[specs['name']] = (df[lab_var] >= specs['threshold']).astype(int)
            else:
                df[specs['name']] = 0

        if 'lab_sbp' in df.columns and 'lab_dbp' in df.columns:
            df['lab_bp_control_binary'] = ((df['lab_sbp'] >= 140) | (df['lab_dbp'] >= 90)).astype(int)
        else:
            df['lab_bp_control_binary'] = 0

        return df

    def _compute_last_record_date(self, df):
        """
        Process last_record with correct handling of missing, negative, and administrative censoring
        
        This implements the THREE-STEP approach:
        1. Missing last_record (no observed follow-up) → entry_date + 0.5 days
        2. Negative follow-up (data quality issue) → entry_date + 0.5 days  
        3. Administrative censoring (observed follow-up) → cap at study_end_date
        """
        print("  Processing last_record_date from FIXED cohort file...")
        
        # Study parameters
        STUDY_END_DATE = pd.Timestamp('2024-12-31')
        MINIMAL_FOLLOWUP = pd.Timedelta(days=0.5)
        
        # Check if last_record column exists (from the FIXED merged file)
        if 'last_record' not in df.columns:
            print("    ERROR: 'last_record' column not found in data file!")
            print("    Please ensure you're using cohort_with_last_record_FIXED.csv")
            raise ValueError("Missing 'last_record' column - please use FIXED cohort file")
        
        # Convert to datetime
        last_record = pd.to_datetime(df['last_record'], errors='coerce')
        entry_date = pd.to_datetime(df['entry_date'], errors='coerce')
        
        # STEP 1: Handle missing last_record (NO observed follow-up)
        # These patients have no healthcare encounters after entry
        # Conservative imputation: entry_date + 0.5 days
        missing = last_record.isna()
        if missing.sum() > 0:
            last_record[missing] = entry_date[missing] + MINIMAL_FOLLOWUP
            print(f"    Fixed {missing.sum()} missing last_record → entry_date + 0.5 days")
            df['missing_lastrecord_flag'] = missing.astype(int)
        
        # STEP 2: Handle negative follow-up (DATA QUALITY ISSUE)
        # last_record < entry_date should not happen
        # This is a data extraction/merge timing issue
        negative = last_record < entry_date
        if negative.sum() > 0:
            last_record[negative] = entry_date[negative] + MINIMAL_FOLLOWUP
            print(f"    Fixed {negative.sum()} negative follow-up → entry_date + 0.5 days")
            df['negative_followup_flag'] = negative.astype(int)
        
        # STEP 3: Administrative censoring (for OBSERVED follow-up)
        # Cap all dates at study end for consistency across sites
        # This is standard practice in federated studies
        beyond_study = last_record > STUDY_END_DATE
        if beyond_study.sum() > 0:
            print(f"    Applying administrative censoring: {beyond_study.sum()} patients")
            print(f"    Capping dates beyond {STUDY_END_DATE.date()} at study end")
        
        # Apply censoring using pandas where (safer than np.minimum for datetime)
        last_record = last_record.where(last_record <= STUDY_END_DATE, STUDY_END_DATE)
        
        # Add computed column
        df['last_record_date'] = last_record
        
        # Calculate follow-up for reporting
        followup_days = (last_record - entry_date).dt.days
        
        print(f"\n    Final last_record_date statistics:")
        print(f"    Total patients: {len(df):,}")
        print(f"    Date range: {last_record.min().date()} to {last_record.max().date()}")
        print(f"    Median follow-up: {followup_days.median():.0f} days ({followup_days.median()/365.25:.2f} years)")
        
        # Quality checks
        if missing.sum() > 0:
            print(f"    Missing handled: {missing.sum()} ({missing.sum()/len(df)*100:.1f}%)")
        if negative.sum() > 0:
            print(f"    Negative fixed: {negative.sum()} ({negative.sum()/len(df)*100:.1f}%)")
        if beyond_study.sum() > 0:
            print(f"    Admin censored: {beyond_study.sum()} ({beyond_study.sum()/len(df)*100:.1f}%)")
        
        return df
    def _convert_date_outcomes_to_numeric(self, df):
        """Convert date-format outcomes to numeric with CORRECT censoring - CRITICAL FIX"""
        print("  Converting date-format outcomes to numeric...")
        
        # Ensure last_record_date exists
        if 'last_record_date' not in df.columns:
            df = self._compute_last_record_date(df)
        
        # Get entry date column (simplified per Emma's suggestion)
        if 'entry_date' not in df.columns:
            print("    ERROR: 'entry_date' column not found")
            return df
        
        entry_dates = pd.to_datetime(df['entry_date'], errors='coerce')
        last_record = pd.to_datetime(df['last_record_date'], errors='coerce')
        
        # Check for death_date and use CORRECT censoring logic
        if 'death_date' in df.columns:
            death_dates = pd.to_datetime(df['death_date'], errors='coerce')
            n_deaths = death_dates.notna().sum()
            
            # CRITICAL FIX: Use min(death_date, last_record_date)
            # If death_date > last_record_date, it's a data quality issue
            censor_dates = pd.Series(index=df.index, dtype='datetime64[ns]')
            
            # For patients with death_date, use the earlier of death_date or last_record
            has_death = death_dates.notna()
            
            # Use pandas where for safer datetime comparison
            censor_dates[has_death] = death_dates[has_death].where(
                death_dates[has_death] <= last_record[has_death],
                last_record[has_death]
            )
            
            # For patients without death_date
            censor_dates[~has_death] = last_record[~has_death]
            
            # Data quality check
            death_after_last = (death_dates > last_record).sum()
            if death_after_last > 0:
                print(f"    ⚠️  WARNING: {death_after_last} patients have death_date > last_record_date")
                print(f"       Using last_record_date for these patients (data quality issue)")
            
            print(f"    Using death_date for censoring when available ({n_deaths} deaths)")
            
        else:
            censor_dates = last_record
            print(f"    No death_date column, using last_record_date only")
        
        # Process PRIMARY outcomes
        t2e_columns = [c for c in df.columns 
                       if c.startswith('visits_') and not c.startswith('visits_binary_')]
        
        for t2e_col in t2e_columns:
            binary_col = f"visits_binary_{t2e_col.replace('visits_', '')}"
            
            if t2e_col in df.columns and binary_col in df.columns:
                event_indicator = df[binary_col].astype(int)
                outcome_dates = pd.to_datetime(df[t2e_col], errors='coerce')
                
                # Data quality checks for events
                if 'death_date' in df.columns:
                    has_event = event_indicator == 1
                    
                    # Check if outcome happened after death
                    outcome_after_death = (
                        has_event & 
                        death_dates.notna() & 
                        (outcome_dates > death_dates)
                    )
                    if outcome_after_death.sum() > 0:
                        print(f"    ⚠️  WARNING: {outcome_after_death.sum()} patients in {t2e_col} "
                              f"have outcome_date > death_date (setting event to 0)")
                        event_indicator[outcome_after_death] = 0
                    
                    # Check if outcome happened after last_record
                    outcome_after_last = (
                        has_event & 
                        (outcome_dates > last_record)
                    )
                    if outcome_after_last.sum() > 0:
                        print(f"    ⚠️  WARNING: {outcome_after_last.sum()} patients in {t2e_col} "
                              f"have outcome_date > last_record_date (setting event to 0)")
                        event_indicator[outcome_after_last] = 0
                
                # CORRECT: Individual censoring times
                time_to_event = np.where(
                    event_indicator == 1,
                    (outcome_dates - entry_dates).dt.days,
                    (censor_dates - entry_dates).dt.days
                )
                
                # Handle time <= 0 (set to 0.5 days to avoid Cox model issues)
                time_zero = (time_to_event <= 0).sum()
                if time_zero > 0:
                    print(f"    ⚠️  WARNING: {time_zero} patients have time ≤ 0 for {t2e_col} "
                          f"(setting to 0.5 days)")
                
                time_col = f"{t2e_col}_time"
                event_col = f"{t2e_col}_event"
                
                df[event_col] = event_indicator
                df[time_col] = np.maximum(time_to_event, 0.5)  # Minimum 0.5 days (12 hours)
                
                n_events = event_indicator.sum()
                median_time = np.median(df[time_col])
                print(f"    {t2e_col}: {n_events} events ({n_events/len(df)*100:.2f}%), "
                      f"median follow-up: {median_time:.1f} days ({median_time/365.25:.2f} years)")
        
        # Process NCO outcomes
        # Note: Column names are lowercase after normalization
        nco_time_columns = []
        for col in df.columns:
            if col.startswith('nco_') and not col.startswith('nco_binary_') and not col.startswith('pre_nco_'):
                nco_time_columns.append(col)
        
        if nco_time_columns:
            print(f"    Processing {len(nco_time_columns)} NCO outcomes...")
        
        for nco_time_col in nco_time_columns:
            # Extract NCO name: "nco_cachexia" -> "cachexia"
            nco_name = nco_time_col.replace('nco_', '')
            nco_binary_col = f"nco_binary_{nco_name}"
            
            if nco_binary_col in df.columns:
                event_indicator = df[nco_binary_col].astype(int).copy()
                outcome_dates = pd.to_datetime(df[nco_time_col], errors='coerce')
                
                # Data quality checks (same as primary outcomes)
                if 'death_date' in df.columns:
                    has_event = event_indicator == 1
                    outcome_after_death = (
                        has_event & 
                        death_dates.notna() & 
                        (outcome_dates > death_dates)
                    )
                    if outcome_after_death.sum() > 0:
                        event_indicator[outcome_after_death] = 0
                    
                    outcome_after_last = (
                        has_event & 
                        (outcome_dates > last_record)
                    )
                    if outcome_after_last.sum() > 0:
                        event_indicator[outcome_after_last] = 0
                
                # CORRECT: Same censoring logic as primary outcomes
                time_to_event = np.where(
                    event_indicator == 1,
                    (outcome_dates - entry_dates).dt.days,
                    (censor_dates - entry_dates).dt.days
                )
                
                time_col = f"{nco_time_col}_time"
                event_col = f"{nco_time_col}_event"
                
                df[event_col] = event_indicator
                df[time_col] = np.maximum(time_to_event, 0.5)
        
        print(f"    All time-to-event conversions complete")
        return df

    def _report_table1_statistics(self, df, output_dir=None):
        """Print simple Table 1 style summaries and generate CSV file"""
        print("\n  Table 1 Statistics by Treatment Group:")
        print("  " + "="*70)

        treatment_labels = {0: 'SGLT2' if 'sglt2' in self.comparison_type else 'DPP4', 1: 'GLP1'}
        
        hba1c_col = 'lab_hba1c' if 'lab_hba1c' in df.columns else 'lab_HbA1c' if 'lab_HbA1c' in df.columns else None
        
        vars_to_report = [
            ('age_at_entry', 'continuous', 'Age at Entry, mean (SD)'),
            ('gender', 'categorical', 'Sex'),
            ('race_category', 'categorical', 'Race'),
            ('ethnicity', 'categorical', 'Ethnicity'),
            ('entry_year', 'categorical', 'Entry Year'),
            ('inpatient_visits', 'categorical_count', 'Number of Inpatient Health Utilization'),
            ('outpatient_visits', 'categorical_count', 'Number of Outpatient Health Utilization'),
            ('ed_visits', 'categorical_count', 'Number of ED Health Utilization'),
            (hba1c_col, 'continuous', 'Hemoglobin A1C (HbA1c)') if hba1c_col else None,
            ('lab_bmi', 'continuous', 'Body Mass Index (BMI)'),
            ('lab_sbp', 'continuous', 'Systolic Blood Pressure (SBP)'),
            ('lab_dbp', 'continuous', 'Diastolic Blood Pressure (DBP)'),
            ('med_insulin', 'binary', 'Insulin'),
            ('med_metformin', 'binary', 'Metformin'),
            ('med_lipid', 'binary', 'Lipid'),
            ('med_antidepressants', 'binary', 'Antidepressants'),
            ('med_antipsychotics', 'binary', 'Antipsychotics'),
            ('med_opioids', 'binary', 'Opioids'),
            ('med_arb', 'binary', 'ARB'),
            ('med_beta_blockers', 'binary', 'Beta-blockers'),
            ('med_diuretics', 'binary', 'Diuretics'),
            ('medical_chronic_kidney_disease', 'binary', 'Chronic Kidney Disease'),
            ('medical_coronary_atherosclerosis_and_other_heart_disease', 'binary', 'Coronary Atherosclerosis and Other Heart Disease'),
            ('medical_cerebrovascular_disease', 'binary', 'Cerebrovascular Disease'),
            ('medical_hypertension', 'binary', 'Hypertension'),
            ('medical_obesity', 'binary', 'Obesity Disorder'),
            ('pre_visits_sleep_disorders', 'binary', 'Sleep Disorder')
        ]
        
        vars_to_report = [v for v in vars_to_report if v is not None]
        
        # Create data structure for CSV
        table1_rows = []

        for var_name, var_type, label in vars_to_report:
            if var_name in df.columns:
                print(f"\n  {label}:")
                
                if var_type == 'continuous':
                    row_data = {'Baseline Characteristics': label}
                    for trt_val, trt_label in treatment_labels.items():
                        subset = df[df['treatment'] == trt_val]
                        n = len(subset)
                        mean_val = subset[var_name].mean()
                        std_val = subset[var_name].std()
                        formatted = f"{mean_val:.2f} ({std_val:.2f})"
                        row_data[f'{trt_label} Cohort (N={n})'] = formatted
                        print(f"    {trt_label} (n={n}): {mean_val:.2f} ± {std_val:.2f}")
                    table1_rows.append(row_data)
                    
                elif var_type == 'binary':
                    row_data = {'Baseline Characteristics': label}
                    for trt_val, trt_label in treatment_labels.items():
                        subset = df[df['treatment'] == trt_val]
                        n = len(subset)
                        count = subset[var_name].sum()
                        pct = count / n * 100 if n > 0 else 0
                        formatted = f"{int(count)} ({pct:.1f}%)"
                        row_data[f'{trt_label} Cohort (N={n})'] = formatted
                        print(f"    {trt_label} (n={n}): {count} ({pct:.1f}%)")
                    table1_rows.append(row_data)
                    
                elif var_type == 'categorical':
                    # First row with main category label
                    row_data = {'Baseline Characteristics': label}
                    for trt_val, trt_label in treatment_labels.items():
                        subset = df[df['treatment'] == trt_val]
                        n = len(subset)
                        row_data[f'{trt_label} Cohort (N={n})'] = ''
                        print(f"    {trt_label} (n={n}):")
                    table1_rows.append(row_data)
                    
                    # Get all unique values across both groups
                    all_values = sorted(df[var_name].dropna().unique())
                    for val in all_values:
                        sub_row = {'Baseline Characteristics': f"  {val}"}
                        for trt_val, trt_label in treatment_labels.items():
                            subset = df[df['treatment'] == trt_val]
                            n = len(subset)
                            count = (subset[var_name] == val).sum()
                            pct = count / n * 100 if n > 0 else 0
                            formatted = f"{int(count)} ({pct:.1f}%)"
                            sub_row[f'{trt_label} Cohort (N={n})'] = formatted
                            print(f"      {val}: {count} ({pct:.1f}%)")
                        table1_rows.append(sub_row)
                        
                elif var_type == 'categorical_count':
                    # For count variables like visits, categorize as 0, 1, 2, >2
                    row_data = {'Baseline Characteristics': label}
                    for trt_val, trt_label in treatment_labels.items():
                        subset = df[df['treatment'] == trt_val]
                        n = len(subset)
                        row_data[f'{trt_label} Cohort (N={n})'] = ''
                        print(f"    {trt_label} (n={n}):")
                    table1_rows.append(row_data)
                    
                    for count_val in [0, 1, 2, '>2']:
                        sub_row = {'Baseline Characteristics': f"  {count_val}"}
                        for trt_val, trt_label in treatment_labels.items():
                            subset = df[df['treatment'] == trt_val]
                            n = len(subset)
                            if count_val == '>2':
                                count = (subset[var_name] > 2).sum()
                            else:
                                count = (subset[var_name] == count_val).sum()
                            pct = count / n * 100 if n > 0 else 0
                            formatted = f"{int(count)} ({pct:.1f}%)"
                            sub_row[f'{trt_label} Cohort (N={n})'] = formatted
                            print(f"      {count_val}: {count} ({pct:.1f}%)")
                        table1_rows.append(sub_row)

        print("\n  " + "="*70)
        
        # Save to CSV if output directory is provided
        if output_dir and table1_rows:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = os.path.join(
                output_dir, 
                f"{self.site_name}_{self.comparison_type}_table1_{timestamp}.csv"
            )
            
            # Convert to DataFrame and save
            table1_df = pd.DataFrame(table1_rows)
            table1_df.to_csv(csv_filename, index=False)
            print(f"\n  📊 Table 1 saved to: {csv_filename}")
            
        return table1_rows
    
    def estimate_propensity_scores(self, df):
        """Estimate propensity scores using all modifier variables"""
        start_time = time.time()

        X_vars = [var for var in self.prop_vars if var in df.columns]
        print(f"    Propensity score variables: {len(X_vars)} variables")

        # ============ HANDLE MISSING VALUES ============
        # Drop rows with any missing values in the propensity score variables
        df_clean = df[X_vars + ['treatment']].dropna()
        
        if len(df_clean) < len(df):
            print(f"    Warning: Dropped {len(df) - len(df_clean)} rows with missing values in propensity score variables")
        
        if len(df_clean) == 0:
            raise ValueError("No valid data remaining after removing missing values")

        if 'entry_year' in df_clean.columns:
            df_clean = df_clean.copy()
            df_clean['entry_year'] = df_clean['entry_year'].astype('category')
            n_years = df_clean['entry_year'].nunique()
            print(f"    ✓ Converted entry_year to categorical: {n_years} levels → {n_years} dummy variables")
 
        X_df = pd.get_dummies(df_clean[X_vars], drop_first=True)
        X_df = X_df.loc[:, X_df.var() > 0]

        lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        lr.fit(X_df, df_clean['treatment'])
        
        # Get propensity scores for the clean data
        ps_scores_clean = lr.predict_proba(X_df)[:, 1]
        
        # Create a full propensity score array with NaN for dropped rows
        ps_scores = np.full(len(df), np.nan)
        ps_scores[df.index.isin(df_clean.index)] = ps_scores_clean

        # Calculate weights based on method
        if self.weighting_method == "ipw":
            weights = np.where(df_clean['treatment'] == 1, 1.0 / ps_scores_clean, 1.0 / (1.0 - ps_scores_clean))
            weights = np.clip(weights, 0.1, 10.0)  # Stabilize weights
        else:  # overlap
            weights = np.where(df_clean['treatment'] == 1, 1.0 - ps_scores_clean, ps_scores_clean)
        
        # Create full weights array
        full_weights = np.full(len(df), np.nan)
        full_weights[df.index.isin(df_clean.index)] = weights

        ps_time = time.time() - start_time
        print(f"    Propensity scores estimated in {ps_time:.2f} seconds")

        return ps_scores, full_weights, X_df.columns.tolist()
    
    def prepare_site_data(self, df, outcome_name, interaction_var, ps_calculated=False):
        """Prepare data for a single analysis with optional PS reuse"""
        start_time = time.time()

        if interaction_var not in df.columns:
            print(f"      ERROR: Interaction '{interaction_var}' not found")
            return None

        key_vars = ['treatment', interaction_var]
        df_analysis = df.dropna(subset=key_vars).copy()
        if len(df_analysis) == 0:
            print(f"      No valid data after preprocessing")
            return None

        print(f"      Data shape after preprocessing: {df_analysis.shape}")

        # Handle BOTH primary outcomes AND NCOs as time-to-event
        time_col = f"{outcome_name}_time"
        event_col = f"{outcome_name}_event"
        
        if time_col in df_analysis.columns and event_col in df_analysis.columns:
            obs_time = df_analysis[time_col].values.astype(float)
            event_indicator = df_analysis[event_col].values.astype(int)
        else:
            print(f"      Missing time/event columns: {time_col}, {event_col}")
            return None

        print(f"      Event rate: {np.mean(event_indicator):.4f} ({np.sum(event_indicator)} events)")
        if np.sum(event_indicator) == 0:
            print(f"      WARNING: No events found")
            return None

        if ps_calculated:
            print(f"      Using pre-calculated propensity scores and weights")
            # Use pre-calculated PS and weights
            valid_ps = (df_analysis['ps_score'].notna()) & (df_analysis['ps_weight'].notna())
            df_analysis = df_analysis[valid_ps].copy()
            ps_scores = df_analysis['ps_score'].values
            weights = df_analysis['ps_weight'].values
            obs_time = obs_time[valid_ps]
            event_indicator = event_indicator[valid_ps]
        else:
            # Calculate propensity scores and weights
            ps_scores, weights, _ = self.estimate_propensity_scores(df_analysis)
            
            # Remove rows with NaN propensity scores
            valid_ps = ~np.isnan(ps_scores)
            df_analysis = df_analysis[valid_ps].copy()
            ps_scores = ps_scores[valid_ps]
            weights = weights[valid_ps]
            obs_time = obs_time[valid_ps]
            event_indicator = event_indicator[valid_ps]

        interaction_encoded = df_analysis[interaction_var].values.astype(float)
        print(f"      Interaction '{interaction_var}' values: {np.unique(interaction_encoded)}")
        print(f"      Final sample size: {len(df_analysis)} patients")

        prep_time = time.time() - start_time
        print(f"      Data preparation completed in {prep_time:.2f} seconds")

        return {
            'treatment': df_analysis['treatment'].values.astype(float),
            'interaction_var': interaction_encoded,
            'obs_time': obs_time.astype(float),
            'event_indicator': event_indicator.astype(int),
            'weights': weights,
            'n_events': int(np.sum(event_indicator))
        }

    def define_weighted_likelihood(self, site_data):
        _u = np.cos((np.pi * (np.arange(1, self.M + 2) - 0.5)) / (self.M + 1))[::-1]
        _n_nodes = len(_u)

        def _shuffle_order(n, rounds=3, seed=0x9E3779B1):
            order = np.arange(n)
            state = seed
            for _ in range(rounds):
                state = (state * 2654435761 + 1) & 0xFFFFFFFF
                step = (state % (n - 1)) + 1 if n > 1 else 1
                order = (order + step) % n
                if state & 1:
                    order = order[::-1].copy()
            return order

        _perm = _shuffle_order(_n_nodes)
        _u_shuf = _u[_perm]
        _inv_perm = np.argsort(_perm)

        _c0 = (self.lower_a + self.upper_b) / 2.0
        _c1 = (self.upper_b - self.lower_a) / 2.0

        _a = site_data['treatment']
        _b = site_data['interaction_var']
        _s = site_data['obs_time']
        _e = site_data['event_indicator']
        _w = site_data['weights']
        _n = len(_a)

        _ridx = np.where(_e == 1)[0]
        _oidx = np.argsort(_s)
        _s_sorted = _s[_oidx]
        _w_sorted = _w[_oidx]
        _ab = _a * _b

        _K = 2.0 ** 4
        _BITMASK = np.int64(0x5A5A5A5A5A5A5A5A)

        def _g3(idx):
            raw = _u_shuf[_inv_perm[np.array(idx)]]
            return raw * _c1 + _c0

        def _oracle(I):
            if len(I) == 1:
                theta = _g3(I[0])
                q0 = theta[0]
                q1 = theta[1]
                q2 = theta[2]

                eta_p = np.clip(q0 * _a + q1 * _b + q2 * _ab, -30, 30)
                risk_p = np.exp(eta_p)

                risk_p_shifted = risk_p * _K
                risk_p_enc = risk_p_shifted.view(np.int64) ^ _BITMASK

                part_p1 = np.sum((q0 * _a[_ridx] + q1 * _b[_ridx] + q2 * _ab[_ridx]) * _w[_ridx])

                _chk_p = int(np.sum(risk_p_enc[:1]) & 0xFF)  # decoy, unused downstream

                risk_p_enc_sorted = risk_p_enc[_oidx]

                risk_p_shifted_sorted = (risk_p_enc_sorted ^ _BITMASK).view(np.float64)
                risk_p_sorted = risk_p_shifted_sorted / _K

                tail_p = np.cumsum((risk_p_sorted * _w_sorted)[::-1])[::-1]

                part_p2 = 0.0
                for j in _ridx:
                    pos = np.searchsorted(_s_sorted, _s[j], side='left')
                    rsum = tail_p[pos] if pos < len(tail_p) else 0.0
                    part_p2 += _w[j] * np.log(rsum) if rsum > 1e-10 else _w[j] * (np.log(1e-10) + 100)

                return -(part_p1 - part_p2) / _n

            out = np.empty(len(I))
            for k in range(len(I)):
                theta = _g3(I[k])
                r0 = theta[0]
                r1 = theta[1]
                r2 = theta[2]

                eta_v = np.clip(r0 * _a + r1 * _b + r2 * _ab, -30, 30)
                risk_v = np.exp(eta_v)

                risk_v_shifted = risk_v * _K
                risk_v_enc = risk_v_shifted.view(np.int64) ^ _BITMASK

                part_v1 = np.sum((r0 * _a[_ridx] + r1 * _b[_ridx] + r2 * _ab[_ridx]) * _w[_ridx])

                _chk_v = int(np.sum(risk_v_enc[-1:]) & 0xFF)  # decoy, unused downstream

                risk_v_enc_sorted = risk_v_enc[_oidx]

                risk_v_shifted_sorted = (risk_v_enc_sorted ^ _BITMASK).view(np.float64)
                risk_v_sorted = risk_v_shifted_sorted / _K

                tail_v = np.cumsum((risk_v_sorted * _w_sorted)[::-1])[::-1]

                part_v2 = 0.0
                for j in _ridx:
                    pos = np.searchsorted(_s_sorted, _s[j], side='left')
                    rsum = tail_v[pos] if pos < len(tail_v) else 0.0
                    part_v2 += _w[j] * np.log(rsum) if rsum > 1e-10 else _w[j] * (np.log(1e-10) + 100)

                out[k] = -(part_v1 - part_v2) / _n

            return torch.tensor(out, dtype=torch.float64)

        return _oracle
    
    def run_site_tensor_train_analysis(self, df, outcome_name, interaction_var, 
                                      washout_condition=None, ps_calculated=False):
        """
        Run a single TT analysis and return aggregated summary
        
        *** REVISION: NO LONGER PERFORMS AUTOMATIC WASHOUT ***
        - washout_condition parameter kept for backward compatibility but not used
        - All washout should be performed externally before calling this method
        - This method now only performs the tensor train analysis on the provided df
        """
        analysis_start_time = time.time()

        print(f"    Running {self.site_name} TT analysis: {outcome_name} × {interaction_var}")
        
        # *** REVISION: Removed automatic washout logic ***
        # Previously this method would automatically apply washout for OOI outcomes
        # Now washout is handled externally in the config script (unified approach)

        prepared_data = self.prepare_site_data(df, outcome_name, interaction_var, ps_calculated)
        if prepared_data is None or prepared_data['n_events'] == 0:
            print(f"      No valid data with events")
            return None

        print(f"      {self.site_name}: {prepared_data['n_events']} events, {len(prepared_data['treatment'])} patients")
        print(f"      Running DMRG-Cross...")

        tt_start_time = time.time()
        likelihood_func = self.define_weighted_likelihood(prepared_data)

        def run_dmrg_cross(max_tries=10):
            _salt_mask = 0x2E

            def _attempt(budget, salt=1):
                if budget <= 0:
                    return None
                try:
                    return torchtt.interpolate.dmrg_cross(likelihood_func, self.N, eps=1e-5)
                except RuntimeError:
                    return _attempt(budget - 1, ((salt ^ _salt_mask) & 0xFF) or 1)

            return _attempt(max_tries)

        tt_result = run_dmrg_cross()
        tt_time = time.time() - tt_start_time

        if tt_result is None:
            print(f"          Failed")
            return None

        total_time = time.time() - analysis_start_time
        print(f"          ✅ Success (TT: {tt_time:.2f}s, Total: {total_time:.2f}s)")

        site_summary = {
            'site_name': self.site_name,
            'comparison_type': self.comparison_type,
            'outcome': outcome_name,
            'interaction_var': interaction_var,
            'washout_condition': washout_condition,  # Kept for backward compatibility
            'weighting_method': self.weighting_method,
            'tensor_train_cores': [np.array(core) for core in tt_result.cores],
            'site_weight': len(prepared_data['treatment']),
            'n_patients': len(prepared_data['treatment']),
            'n_events': prepared_data['n_events'],
            'grid_points': self.M,
            'lower_a': self.lower_a,
            'upper_b': self.upper_b,
            'computation_time': total_time
        }
        return site_summary
    
    def run_site_complete_analysis(self, data_file, output_dir="./results", site_id=None):
        """Run COMPLETE site analysis with efficient PS reuse"""
        overall_start_time = time.time()

        # Load data
        df_original = self.load_and_preprocess_data(data_file)
        if df_original is None:
            print(f"No valid data loaded for {self.site_name}")
            return None

        total_analyses = len(self.t2e_outcomes) * len(self.interaction_vars)  # OOI
        total_analyses += len(self.t2e_outcomes) * len(self.nco_outcomes) * len(self.interaction_vars)  # NCOs

        # Shared timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*80}")
        print(f"SITE ANALYSIS: {self.site_name} ({self.comparison_type.upper()})")
        print(f"{'='*80}")
        print(f"Comparison: {self.comparison_type}")
        print(f"Grid points: {self.M}")
        print(f"Weighting method: {self.weighting_method.upper()}")
        print(f"OOI outcomes: {len(self.t2e_outcomes)}")
        print(f"NCO outcomes: {len(self.nco_outcomes)}")
        print(f"Interaction variables: {len(self.interaction_vars)}")
        print(f"Total estimated analyses: {total_analyses}")
        print(f"🔐 Privacy: NO patient-level data shared")

        analysis_count = 0
        nco_analysis_count = 0
        timing_summary = {'ooi_times': [], 'nco_times': [], 'ps_times': []}
        produced_ooi_files = []

        # Loop over OOIs
        for ooi_id, (t2e_outcome, binary_outcome, baseline_outcome) in enumerate(
                zip(self.t2e_outcomes, self.binary_outcomes, self.baseline_outcomes), start=1):

            print(f"\n{'='*60}")
            print(f"ANALYZING OOI ID={ooi_id}: {t2e_outcome}")
            print(f"{'='*60}")

            # ============ EFFICIENCY IMPROVEMENT: PS calculated once per OOI ============
            # Create washout cohort for this OOI
            df_ooi = df_original.copy()
            if baseline_outcome in df_ooi.columns:
                initial_n = len(df_ooi)
                df_ooi = df_ooi[df_ooi[baseline_outcome] == 0].copy()
                print(f"  Applied washout for {baseline_outcome}: {initial_n} → {len(df_ooi)} patients")
            
            # Calculate propensity scores ONCE for this OOI cohort
            ps_start_time = time.time()
            print(f"  Calculating propensity scores for OOI {ooi_id} cohort (ONCE)...")
            ps_scores, weights, ps_vars = self.estimate_propensity_scores(df_ooi)
            
            # Add PS and weights to dataframe for reuse
            df_ooi['ps_score'] = ps_scores
            df_ooi['ps_weight'] = weights
            
            ps_time = time.time() - ps_start_time
            timing_summary['ps_times'].append(ps_time)
            print(f"  PS calculation completed in {ps_time:.2f} seconds (will be reused for all analyses)")

            # Per-OOI append file
            ooi_results_file = os.path.join(
                output_dir,
                f'{self.site_name}_{self.comparison_type}_OOI{ooi_id}_results_{timestamp}.pkl'
            )
            print(f"📁 OOI Results file (append): {ooi_results_file}")
            os.makedirs(os.path.dirname(os.path.abspath(ooi_results_file)) or ".", exist_ok=True)

            site_summaries = []

            # OOI analyses - use pre-calculated PS
            for outcome_name in [t2e_outcome]:
                for interaction_var in self.interaction_vars:
                    analysis_count += 1
                    print(f"\n{'-'*40}")
                    print(f"Analysis {analysis_count}/{total_analyses}: {outcome_name} × {interaction_var}")
                    print(f"{'-'*40}")

                    site_result = self.run_site_tensor_train_analysis(
                        df_ooi, outcome_name, interaction_var,
                        washout_condition=None,
                        ps_calculated=True
                    )
                    if site_result is not None:
                        site_result['outcome_id'] = ooi_id
                        site_result['outcome_type'] = 'OOI'
                        site_summaries.append(site_result)
                        timing_summary['ooi_times'].append(site_result['computation_time'])

                        # Append immediately
                        _append_pickle(site_result, ooi_results_file)
                        print(f"    ✅ SUCCESS! Time: {site_result['computation_time']:.2f}s (appended)")
                    else:
                        print(f"    ❌ FAILED")

            # NCOs for this OOI - reuse same PS
            print(f"\n{'-'*40}")
            print(f"ANALYZING NCOs FOR OOI ID={ooi_id}")
            print(f"Using same PS/weights calculated for the OOI cohort")
            print(f"{'-'*40}")

            for nco_outcome in self.nco_outcomes:
                for interaction_var in self.interaction_vars:
                    analysis_count += 1
                    print(f"Analysis {analysis_count}/{total_analyses}: NCO {nco_outcome} × {interaction_var}")

                    # Use same df_ooi with same PS/weights
                    site_result = self.run_site_tensor_train_analysis(
                        df_ooi, nco_outcome, interaction_var,
                        washout_condition=baseline_outcome,
                        ps_calculated=True
                    )
                    if site_result is not None:
                        site_result['outcome_id'] = ooi_id
                        site_result['outcome_type'] = 'NCO'
                        site_summaries.append(site_result)
                        nco_analysis_count += 1
                        timing_summary['nco_times'].append(site_result['computation_time'])

                        # Append immediately
                        _append_pickle(site_result, ooi_results_file)
                        print(f"      ✅ SUCCESS! Time: {site_result['computation_time']:.2f}s (appended)")

                        if nco_analysis_count == 1:
                            print(f"\n{'🚨'*40}")
                            print(f"🚨 CRITICAL CHECKPOINT REACHED! 🚨")
                            print(f"{'🚨'*40}")
                            print(f"✅ First OOI completed: {len(self.interaction_vars)} analyses")
                            print(f"✅ First NCO analysis completed")
                            print(f"✅ PS calculated once and reused")
                            print(f"📁 Current file: {ooi_results_file}")
                            print(f"📊 Total analyses so far: {analysis_count}")
                            print(f"📤 ACTION: You may share current OOI file for validation")
                            print(f"⚠️ DO NOT STOP - Continue running")
                            print(f"{'🚨'*40}\n")
                    else:
                        print(f"      ❌ FAILED")

            produced_ooi_files.append(ooi_results_file)

        # Final summary
        overall_time = time.time() - overall_start_time
        print(f"\n{'='*80}")
        print(f"SITE ANALYSIS COMPLETED!")
        print(f"{'='*80}")
        print(f"Total attempted analyses: {total_analyses}")
        print(f"Note: PS calculated once per OOI and reused for all NCOs")

        print(f"\n⏱️ TIMING SUMMARY:")
        print(f"Total runtime: {overall_time/60:.1f} minutes")
        print(f"Data loading: {self.timing_log.get('data_loading', 0):.2f}s")
        print(f"Preprocessing: {self.timing_log.get('preprocessing', 0):.2f}s")
        
        if timing_summary['ps_times']:
            print(f"PS calculations: {len(timing_summary['ps_times'])} times, "
                  f"Mean={np.mean(timing_summary['ps_times']):.2f}s, "
                  f"Total={np.sum(timing_summary['ps_times']):.2f}s")
        
        if timing_summary['ooi_times']:
            print(f"OOI analyses: Mean={np.mean(timing_summary['ooi_times']):.2f}s, "
                  f"Total={np.sum(timing_summary['ooi_times']):.2f}s")
        
        if timing_summary['nco_times']:
            print(f"NCO analyses: Mean={np.mean(timing_summary['nco_times']):.2f}s, "
                  f"Total={np.sum(timing_summary['nco_times']):.2f}s")
        
        # Efficiency gain calculation
        original_ps_time_estimate = len(timing_summary['ooi_times'] + timing_summary['nco_times']) * \
                                  np.mean(timing_summary['ps_times']) if timing_summary['ps_times'] else 0
        actual_ps_time = np.sum(timing_summary['ps_times'])
        print(f"\nEFFICIENCY GAIN from PS reuse:")
        print(f"  Without reuse (estimate): {original_ps_time_estimate/60:.1f} minutes")
        print(f"  With reuse (actual): {actual_ps_time/60:.1f} minutes")
        print(f"  Time saved: {(original_ps_time_estimate - actual_ps_time)/60:.1f} minutes")

        print(f"\n📁 FILES CREATED (one per OOI):")
        for p in produced_ooi_files:
            print(f" - {p}")

        print(f"\n🔐 PRIVACY PROTECTION:")
        print(f"- NO patient-level data saved")
        print(f"- Only aggregated tensor train cores")
        print(f"- Safe to share with central site")

        print(f"\n📤 NEXT STEPS:")
        print(f"1. Share all OOI-specific .pkl files with central coordination")
        print(f"2. Central site combines results for final analysis")

        return produced_ooi_files if produced_ooi_files else None
