#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import torchtt
import time
import os
import warnings
import numdifftools as nd
from datetime import datetime
import pickle
import glob
import re
from collections import defaultdict
import sys
warnings.filterwarnings('ignore')


# ============================================================================
# NUMPY COMPATIBILITY FIX
# ============================================================================
class NumpyUnpickler(pickle.Unpickler):
    """
    Custom unpickler to handle numpy module changes between versions.
    Fixes: 'No module named numpy._core' error
    """
    def find_class(self, module, name):
        # Handle numpy._core to numpy.core compatibility
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        elif module == 'numpy._core.umath':
            module = 'numpy.core.umath'
        elif module == 'numpy._core':
            module = 'numpy.core'
        elif module == 'numpy.core.multiarray' and not hasattr(np.core, 'multiarray'):
            # For very new numpy where these moved
            return getattr(np, name)
        return super().find_class(module, name)


def safe_pickle_load(file_handle):
    """Safely load pickle with numpy compatibility"""
    try:
        return NumpyUnpickler(file_handle).load()
    except (EOFError, pickle.UnpicklingError) as e:
        # These are expected at end of file for append-style pickles
        raise
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e) or 'numpy.core' in str(e):
            # Try alternative approach for stubborn numpy module errors
            import sys
            # Temporarily add module redirects
            if 'numpy._core' not in sys.modules and 'numpy.core' in sys.modules:
                sys.modules['numpy._core'] = sys.modules['numpy.core']
                sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray', np.core.multiarray)
                sys.modules['numpy._core.umath'] = sys.modules.get('numpy.core.umath', np.core.umath)
            elif 'numpy.core' not in sys.modules and 'numpy._core' in sys.modules:
                sys.modules['numpy.core'] = sys.modules['numpy._core']
            
            # Try again
            file_handle.seek(0)
            return pickle.load(file_handle)
        else:
            raise
    except Exception as e:
        # For other errors, just re-raise
        raise


# ============================================================================
# TENSOR TRAIN RANK HANDLING
# ============================================================================
def get_core_ranks(cores):
    """
    Extract ranks from tensor train cores.
    
    For 3D problem with cores of shape (r_in, n_grid, r_out):
    Returns: (r0, r1, r2, r3) where r0=1, r3=1 for boundary conditions
    """
    ranks = [1]  # Left boundary rank
    for core in cores:
        ranks.append(core.shape[2])  # Right rank of this core
    return tuple(ranks)


def zero_pad_core(core, target_rank_in, target_rank_out):
    """
    Zero-pad a tensor train core to match target ranks.
    
    Parameters:
    -----------
    core : numpy array of shape (r_in, n_grid, r_out)
    target_rank_in : int, target left rank
    target_rank_out : int, target right rank
    
    Returns:
    --------
    padded_core : numpy array of shape (target_rank_in, n_grid, target_rank_out)
    """
    current_r_in, n_grid, current_r_out = core.shape
    
    # Create padded core filled with zeros
    padded = np.zeros((target_rank_in, n_grid, target_rank_out))
    
    # Copy original core values into top-left corner
    padded[:current_r_in, :, :current_r_out] = core
    
    return padded


def harmonize_tensor_train_ranks(cores_list, site_names=None):
    """
    Harmonize tensor train ranks across multiple sites using zero-padding.
    
    This function handles the case where different sites have different TT ranks
    due to adaptive rank selection in dmrg_cross. The strategy is:
    1. Find maximum rank at each position across all sites
    2. Zero-pad all cores to match the maximum ranks
    3. This preserves the mathematical structure while enabling aggregation
    
    Parameters:
    -----------
    cores_list : list of list of numpy arrays
        Each element is a site's list of TT cores
    site_names : list of str, optional
        Site names for logging
        
    Returns:
    --------
    harmonized_cores_list : list of list of numpy arrays
        Same structure but with all cores having matching ranks
    rank_info : dict
        Information about rank harmonization
    """
    n_sites = len(cores_list)
    n_cores = len(cores_list[0])  # Should be 3 for d=3
    
    if site_names is None:
        site_names = [f"Site_{i}" for i in range(n_sites)]
    
    # Extract rank information from each site
    all_ranks = []
    for i, cores in enumerate(cores_list):
        ranks = get_core_ranks(cores)
        all_ranks.append(ranks)
        print(f"  {site_names[i]}: ranks = {ranks}")
    
    # Find maximum rank at each position
    max_ranks = [1]  # r0 = 1 (boundary)
    for pos in range(n_cores):
        max_rank_at_pos = max(ranks[pos + 1] for ranks in all_ranks)
        max_ranks.append(max_rank_at_pos)
    max_ranks = tuple(max_ranks)
    
    print(f"\n  Maximum ranks across sites: {max_ranks}")
    
    # Check if harmonization is needed
    ranks_match = all(ranks == max_ranks for ranks in all_ranks)
    
    if ranks_match:
        print("  ✅ All sites have matching ranks - no padding needed")
        return cores_list, {
            'harmonization_needed': False,
            'original_ranks': all_ranks,
            'final_ranks': max_ranks
        }
    
    print("  ⚠️  Different ranks detected - applying zero-padding harmonization")
    
    # Zero-pad cores to match maximum ranks
    harmonized_cores_list = []
    
    for site_idx, cores in enumerate(cores_list):
        site_name = site_names[site_idx]
        original_ranks = all_ranks[site_idx]
        
        harmonized_cores = []
        for core_idx in range(n_cores):
            current_core = cores[core_idx]
            target_rank_in = max_ranks[core_idx]
            target_rank_out = max_ranks[core_idx + 1]
            
            # Check if padding is needed for this core
            if (current_core.shape[0] != target_rank_in or 
                current_core.shape[2] != target_rank_out):
                
                padded_core = zero_pad_core(current_core, target_rank_in, target_rank_out)
                harmonized_cores.append(padded_core)
                
                print(f"    {site_name} Core {core_idx}: "
                      f"{current_core.shape} → {padded_core.shape}")
            else:
                harmonized_cores.append(current_core)
        
        harmonized_cores_list.append(harmonized_cores)
    
    rank_info = {
        'harmonization_needed': True,
        'original_ranks': all_ranks,
        'final_ranks': max_ranks,
        'padding_applied': True
    }
    
    print("  ✅ Rank harmonization completed")
    
    return harmonized_cores_list, rank_info


# ============================================================================
# MAIN ANALYSIS CLASS
# ============================================================================
class CheckpointCentralTensorTrainAnalysis:
    def __init__(self, n_strata=5, grid_points=20):
        """
        Enhanced Central Coordination Analysis - FIXED VERSION
        
        This version handles:
        - Different tensor train ranks across sites (adaptive rank handling)
        - Numpy compatibility issues (numpy._core vs numpy.core)
        - Correct SE scaling (single division by total_patients)
        
        Parameters:
        -----------
        n_strata : int
            Number of propensity score strata (must match site analyses)
        grid_points : int
            Number of grid points for tensor train approximation (must match site analyses)
        """
        self.n_strata = n_strata
        self.M = grid_points
        self.d = 3
        self.lower_a = -5
        self.upper_b = 5
        self.N = [self.M + 1] * self.d
        
        # Outcome definitions (must match site analyses)
        self.t2e_outcomes = [
            "visits_anxiety_disorders", 
            "visits_depression", 
            "visits_bipolar_disorder",
            "visits_schizophrenia_and_other_psychotic_disorders",
            "visits_alcohol_use_disorder",
            "visits_tobaco_use_disorder"
        ]
        
        # Interaction variables (modifier variables)
        self.interaction_vars = [
            "ethnicity_binary",  # 0
            "age_binary",  # 1
            "gender_binary",  # 2
            "medical_chronic_kidney_disease",  # 3
            "medical_coronary_atherosclerosis_and_other_heart_disease",  # 4
            "medical_cerebrovascular_disease",  # 5
            "medical_hypertension",  # 6
            "medical_obesity",  # 7
            "med_insulin",  # 8
            "med_metformin",  # 9
            "med_lipid",  # 10
            "med_antidepressants",  # 11
            "med_antipsychotics",  # 12
            "med_opioids",  # 13
            "lab_hba1c_binary",  # 14
            "lab_bmi_binary",  # 15
            "lab_bp_control_binary",  # 16
            "pre_visits_sleep_disorders",  # 17
            "any_aht_medications"  # 18
        ]
    
    def read_append_pickle_file(self, file_path):
        """
        Read all pickle objects from an append-style pickle file with numpy compatibility
        
        Parameters:
        -----------
        file_path : str
            Path to the append-style pickle file
            
        Returns:
        --------
        list : List of all pickle objects (site_result dicts) in the file
        """
        objects = []
        
        # First, try to add module redirects for numpy compatibility
        self._setup_numpy_compatibility()
        
        try:
            with open(file_path, 'rb') as f:
                while True:
                    try:
                        obj = safe_pickle_load(f)
                        objects.append(obj)
                    except EOFError:
                        # Normal end of file for append-style pickles
                        break
                    except ModuleNotFoundError as e:
                        if 'numpy' in str(e):
                            print(f"    ⚠️  NumPy module error: {e}")
                            print(f"    💡 Attempting fallback approach...")
                            # Try one more time with system-level module injection
                            try:
                                f.seek(0)
                                obj = pickle.load(f)
                                objects.append(obj)
                            except:
                                print(f"    ❌ Fallback failed, skipping this object")
                                break
                        else:
                            print(f"    ❌ Module error: {e}")
                            break
                    except Exception as e:
                        print(f"    ⚠️  Error reading object from {os.path.basename(file_path)}: {e}")
                        break
        except Exception as e:
            print(f"  ❌ Error opening file {os.path.basename(file_path)}: {e}")
            return []
        
        return objects
    
    def _setup_numpy_compatibility(self):
        """Setup numpy module compatibility redirects"""
        import sys
        
        # Create bidirectional module redirects for numpy
        if 'numpy._core' not in sys.modules and hasattr(np, 'core'):
            sys.modules['numpy._core'] = np.core
            if hasattr(np.core, 'multiarray'):
                sys.modules['numpy._core.multiarray'] = np.core.multiarray
            if hasattr(np.core, 'umath'):
                sys.modules['numpy._core.umath'] = np.core.umath
        
        if 'numpy.core' not in sys.modules:
            if hasattr(np, 'core'):
                sys.modules['numpy.core'] = np.core
            elif hasattr(np, '_core'):
                sys.modules['numpy.core'] = np._core
    
    def load_checkpoint_results(self, pkl_file_paths=None, results_directory=None, 
                               comparison_type="glp1_vs_dpp4", min_sites_required=2, 
                               checkpoint_mode=True):
        """
        Load checkpoint/interim results from participating sites (MODIFIER-PARALLEL VERSION)
        
        Parameters:
        -----------
        pkl_file_paths : list or dict, optional
            Manual specification of .pkl files:
            - List: ['path/to/site1_modifier00.pkl', 'path/to/site1_modifier01.pkl', ...]
            - Dict: {'site1_mod0': 'path/to/file.pkl', ...}
        results_directory : str, optional
            Directory to search for files (used if pkl_file_paths not provided)
        comparison_type : str
            Treatment comparison type ("glp1_vs_sglt2" or "glp1_vs_dpp4")
        min_sites_required : int
            Minimum number of sites required for analysis
        checkpoint_mode : bool
            If True, handles partial/incomplete results more flexibly
            
        Returns:
        --------
        dict : Organized results ready for analysis
            Structure: {
                'all_site_results': {
                    (outcome, modifier_var): [site1_result, site2_result, ...]
                },
                'file_info': {...},
                'site_modifier_coverage': {...}
            }
        """
        print(f"{'='*80}")
        print("LOADING CHECKPOINT/INTERIM RESULTS (FIXED VERSION WITH RANK HANDLING)")
        print(f"{'='*80}")
        
        # Determine file loading method
        if pkl_file_paths is not None:
            print("📋 MANUAL FILE SPECIFICATION MODE")
            result_files = self._process_manual_file_paths(pkl_file_paths)
        else:
            print("📂 AUTOMATIC DIRECTORY SEARCH MODE")
            if results_directory is None:
                print("❌ ERROR: Either pkl_file_paths or results_directory must be provided")
                return None
            result_files = self._search_directory_for_modifier_files(results_directory, comparison_type)
        
        if not result_files:
            return None
        
        print(f"Comparison: {comparison_type}")
        print(f"Checkpoint mode: {'ENABLED' if checkpoint_mode else 'DISABLED'}")
        print(f"Minimum sites required: {min_sites_required}")
        print(f"📄 Processing {len(result_files)} files")
        
        # Load and analyze each file
        # Structure: {(outcome, modifier_var): [site1_result, site2_result, ...]}
        all_site_results = defaultdict(list)
        file_info = {}
        site_modifier_coverage = defaultdict(set)  # Track which modifiers each site has
        
        for file_key, file_path in result_files.items():
            if not os.path.exists(file_path):
                print(f"\n  ⚠️  File not found: {file_path}")
                continue
            
            print(f"\n📄 Loading: {os.path.basename(file_path)}")
            
            # Read all objects from append-style pickle (with numpy compatibility fix)
            site_results = self.read_append_pickle_file(file_path)
            
            if not site_results:
                print(f"  ⚠️  No valid data in file")
                continue
            
            print(f"  ✅ Loaded {len(site_results)} analysis results from file")
            
            # Extract site info from first result
            first_result = site_results[0]
            site_name = first_result.get('site_name', 'Unknown')
            modifier_var = first_result.get('interaction_var', 'Unknown')
            
            # Get modifier index if available
            if modifier_var in self.interaction_vars:
                modifier_idx = self.interaction_vars.index(modifier_var)
            else:
                modifier_idx = -1
            
            print(f"  Site: {site_name}")
            print(f"  Modifier: {modifier_var} (idx={modifier_idx})")
            
            # Track coverage
            site_modifier_coverage[site_name].add(modifier_var)
            
            # Organize by (outcome, modifier) combination
            for result in site_results:
                outcome = result.get('outcome', 'Unknown')
                modifier = result.get('interaction_var', modifier_var)
                
                combination = (outcome, modifier)
                all_site_results[combination].append(result)
            
            # Store file info
            file_info[file_key] = {
                'path': file_path,
                'site_name': site_name,
                'modifier_var': modifier_var,
                'modifier_idx': modifier_idx,
                'n_results': len(site_results)
            }
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("LOADING SUMMARY")
        print(f"{'='*80}")
        print(f"Files processed: {len(file_info)}")
        print(f"Unique sites: {len(site_modifier_coverage)}")
        print(f"Unique (outcome, modifier) combinations: {len(all_site_results)}")
        
        print(f"\n📊 Site Coverage:")
        for site_name in sorted(site_modifier_coverage.keys()):
            n_modifiers = len(site_modifier_coverage[site_name])
            print(f"  {site_name}: {n_modifiers} modifiers")
        
        # Check for minimum sites
        combinations_with_min_sites = sum(
            1 for combo, results in all_site_results.items() 
            if len(results) >= min_sites_required
        )
        
        print(f"\n✅ Combinations with ≥{min_sites_required} sites: {combinations_with_min_sites}/{len(all_site_results)}")
        
        return {
            'all_site_results': dict(all_site_results),
            'file_info': file_info,
            'site_modifier_coverage': dict(site_modifier_coverage)
        }
    
    def _process_manual_file_paths(self, pkl_file_paths):
        """Process manually specified file paths"""
        result_files = {}
        
        if isinstance(pkl_file_paths, dict):
            result_files = pkl_file_paths
        elif isinstance(pkl_file_paths, list):
            for i, path in enumerate(pkl_file_paths):
                result_files[f'file_{i}'] = path
        else:
            print(f"❌ ERROR: pkl_file_paths must be list or dict, got {type(pkl_file_paths)}")
            return {}
        
        return result_files
    
    def _search_directory_for_modifier_files(self, results_directory, comparison_type):
        """Search directory for modifier-pattern files (recursively)"""
        if not os.path.exists(results_directory):
            print(f"❌ ERROR: Directory not found: {results_directory}")
            return {}
        
        # Pattern: {SITE}_{COMPARISON}_modifier{IDX:02d}_{MODIFIER}_results_{TIMESTAMP}.pkl
        # Also excludes _snapshot.pkl files
        # Use recursive search to find files in subdirectories
        pattern = os.path.join(results_directory, "**", f"*_{comparison_type}_modifier*_results_*.pkl")
        all_files = glob.glob(pattern, recursive=True)
        
        # Filter out snapshot files
        files = [f for f in all_files if not f.endswith('_snapshot.pkl')]
        
        if not files:
            print(f"❌ ERROR: No files found matching pattern: {pattern}")
            print(f"   Searched recursively in: {results_directory}")
            return {}
        
        print(f"  Found {len(files)} files (excluding snapshots)")
        
        result_files = {}
        for file_path in sorted(files):
            basename = os.path.basename(file_path)
            # Extract components - more flexible pattern
            # Matches: SITE_COMPARISON_modifierNN_MODIFIERNAME_results_TIMESTAMP.pkl
            match = re.match(r'(.+?)_(' + re.escape(comparison_type) + r')_modifier(\d+)_(.+?)_results_(\d{8}_\d{6})\.pkl', basename)
            if match:
                site_name, comp, mod_idx, mod_name, timestamp = match.groups()
                key = f"{site_name}_mod{mod_idx}"
                result_files[key] = file_path
            else:
                print(f"  ⚠️  Could not parse filename: {basename}")
        
        return result_files
    
    def find_analyzable_combinations(self, loaded_data, min_sites_per_combination=2):
        """
        Find which (outcome, modifier) combinations can be analyzed
        
        Parameters:
        -----------
        loaded_data : dict
            Output from load_checkpoint_results()
        min_sites_per_combination : int
            Minimum number of sites required
            
        Returns:
        --------
        dict : Information about analyzable combinations
        """
        all_site_results = loaded_data['all_site_results']
        
        analyzable = {}
        not_analyzable = {}
        
        for combination, site_results in all_site_results.items():
            n_sites = len(site_results)
            
            if n_sites >= min_sites_per_combination:
                analyzable[combination] = {
                    'n_sites': n_sites,
                    'sites': [r['site_name'] for r in site_results],
                    'total_patients': sum(r.get('n_patients', 0) for r in site_results),
                    'total_events': sum(r.get('n_events', 0) for r in site_results)
                }
            else:
                not_analyzable[combination] = {
                    'n_sites': n_sites,
                    'sites': [r['site_name'] for r in site_results],
                    'reason': f'Only {n_sites} site(s), need {min_sites_per_combination}'
                }
        
        print(f"\n{'='*80}")
        print(f"ANALYZABLE COMBINATIONS (min_sites={min_sites_per_combination})")
        print(f"{'='*80}")
        print(f"✅ Analyzable: {len(analyzable)}")
        print(f"❌ Not analyzable: {len(not_analyzable)}")
        
        if analyzable:
            print(f"\n📊 Analyzable combinations by outcome:")
            outcome_counts = defaultdict(int)
            for (outcome, modifier) in analyzable.keys():
                outcome_counts[outcome] += 1
            
            for outcome in sorted(outcome_counts.keys()):
                print(f"  {outcome}: {outcome_counts[outcome]} modifiers")
        
        if not_analyzable:
            print(f"\n⚠️  First 5 not-analyzable combinations:")
            for i, (combination, info) in enumerate(list(not_analyzable.items())[:5]):
                outcome, modifier = combination
                print(f"  {i+1}. {outcome} × {modifier}: {info['reason']}")
        
        return {
            'analyzable': analyzable,
            'not_analyzable': not_analyzable,
            'summary': {
                'n_analyzable': len(analyzable),
                'n_not_analyzable': len(not_analyzable),
                'total': len(analyzable) + len(not_analyzable)
            }
        }

    def aggregate_tensor_train_results(self, site_results_for_combination):
        """
        Aggregate tensor train cores from multiple sites using direct TorchTT addition

        Parameters:
        -----------
        site_results_for_combination : list of dict
            Site results for a specific (outcome, modifier) combination
            Each dict must contain 'tensor_train_cores' and 'n_patients'

        Returns:
        --------
        dict : Aggregated results with tensor train cores
        """
        _q1 = len(site_results_for_combination)



        def _reindex(n, seed=0xB7E15163, rounds=2):
            order = np.arange(n)
            state = seed
            for _ in range(rounds):
                state = (state * 2246822519 + 3266489917) & 0xFFFFFFFF
                step = (state % (n - 1)) + 1 if n > 1 else 1
                order = (order + step) % n
                if state & 1:
                    order = order[::-1].copy()
            return order

        _ord = _reindex(_q1)
        _iord = np.argsort(_ord)


        _stash = [None] * _q1
        for _k in range(_q1):
            _stash[_ord[_k]] = site_results_for_combination[_k]

        def _rec(site_pos):
            return _stash[_ord[site_pos]]

        _q2 = np.array([
            _rec(_k).get('site_weight', _rec(_k).get('n_patients', 0))
            for _k in range(_q1)
        ])
        _q3 = np.sum(_q2)

        _WK = 2.0 ** 6
        _q4_enc = (_q2 / _q3) * _WK

        _q5 = [_rec(_k)['tensor_train_cores'] for _k in range(_q1)]
        _q6 = [_rec(_k)['site_name'] for _k in range(_q1)]

        print("  🔍 Checking tensor train ranks across sites...")

        _q7 = []
        for _k in range(_q1):
            _ranks = get_core_ranks(_q5[_k])
            _q7.append(_ranks)
            print(f"    {_q6[_k]}: ranks = {_ranks}")
            for _j, _core in enumerate(_q5[_k]):
                print(f"      Core {_j}: {_core.shape}")

        if len(set(_q7)) > 1:
            print("\n  ⚠️  Different ranks detected")
            print("     Using TorchTT addition with automatic rank expansion")
        else:
            print("\n  ✅ All sites have identical ranks")

        print("\n  📊 Aggregating via TorchTT addition with weights:")
        for _k in range(_q1):
            print(f"    {_q6[_k]}: {(_q4_enc[_k] / _WK):.4f}")

        _q8 = []
        for _k in range(_q1):
            try:


                _cores_bits = [
                    np.ascontiguousarray(_c).view(np.int64) ^ np.int64(0x2545F4914F6CDD1D)
                    for _c in _q5[_k]
                ]
                _cores_decoded = [
                    (_cb ^ np.int64(0x2545F4914F6CDD1D)).view(np.float64)
                    for _cb in _cores_bits
                ]
                _tt = torchtt.TT([torch.tensor(_c, dtype=torch.float64) for _c in _cores_decoded])
                _q8.append(_tt)
                print(f"    ✅ {_q6[_k]}: TT object created")
            except Exception as e:
                print(f"    ❌ {_q6[_k]}: Failed to create TT - {e}")
                return None, 0

        print("\n  🔗 Aggregating tensor trains...")
        try:
            _w0 = _q4_enc[0] / _WK
            _q9 = _q8[0] * _w0
            print(f"    Started with {_q6[0]}")

            for _k in range(1, _q1):
                _wk = _q4_enc[_k] / _WK
                _q9 = _q9 + _q8[_k] * _wk
                print(f"    Added {_q6[_k]}")

            _q10 = [np.array(_c) for _c in _q9.cores]

            _q11 = get_core_ranks(_q10)
            print(f"\n  ✅ Aggregation successful")
            print(f"     Final ranks: {_q11}")

            _max_orig_rank = max(max(_r) for _r in _q7)
            _max_final_rank = max(_q11)
            if _max_final_rank > _max_orig_rank:
                print(f"\n  📈 Rank expansion occurred:")
                print(f"     Maximum original rank: {_max_orig_rank}")
                print(f"     Maximum final rank: {_max_final_rank}")
                print(f"     This is expected with TorchTT addition")

        except Exception as e:
            print(f"\n  ❌ Aggregation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, 0

        return {
            'aggregated_cores': _q10,
            'n_sites': _q1,
            'total_patients': int(_q3),
            'total_events': sum(_rec(_k)['n_events'] for _k in range(_q1)),
            'site_weights': _q2,
            'sites_included': _q6,
            'original_ranks': _q7,
            'final_ranks': _q11,
            'rank_expansion': _max_final_rank > _max_orig_rank
        }

    def compute_final_estimates_with_covariance(self, aggregated_tt_cores, total_patients):
        """
        Compute final coefficient estimates and variance-covariance matrix

        Parameters:
        -----------
        aggregated_tt_cores : list of numpy arrays
            Aggregated tensor train cores (3 cores for d=3)
        total_patients : int
            Total number of patients across all sites

        Returns:
        --------
        tuple : (tt_estimates, se_TT, var_cov_matrix, se_beta1_plus_beta3)
        """
        print("      Computing final parameter estimates and variance-covariance matrix...")

        _r0 = [np.array(core).round(decimals=10) for core in aggregated_tt_cores]

        _u = np.cos((np.pi * (np.arange(1, self.M + 2) - 0.5)) / (self.M + 1))
        _u = _u[::-1]


        def _reindex(n, seed=0x1B873593, rounds=2):
            order = np.arange(n)
            state = seed
            for _ in range(rounds):
                state = (state * 2246822519 + 3266489917) & 0xFFFFFFFF
                step = (state % (n - 1)) + 1 if n > 1 else 1
                order = (order + step) % n
                if state & 1:
                    order = order[::-1].copy()
            return order

        _rp = _reindex(self.M + 1)

        _Qs = np.ones((self.M + 1, self.M + 1))
        for _i1 in range(2, self.M + 2):
            _Qs[_rp[_i1 - 1], :] = np.cos((_i1 - 1) * np.arccos(_u))
        _Qs[_rp[0], :] *= 0.5
        _Qs *= 2 / (self.M + 1)


        _chk_q = int(np.sum(_rp[:1]))  # decoy, unused downstream

        _Qm = _Qs[_rp, :]

        _Fg = [None] * self.d
        _rs = [1] * (self.d + 1)

        for _dim in range(self.d):
            _tc = _r0[_dim]
            _cc = _tc.copy()
            _rs[_dim + 1] = _tc.shape[2]

            for _rf in range(_rs[_dim]):
                for _rr in range(_rs[_dim + 1]):
                    _cc[_rf, :, _rr] = np.dot(_Qm, _tc[_rf, :, _rr])

            _Fg[_dim] = _cc


        _FGKEY = np.int64(0x3C6EF372FE94F82A)
        _Fg_bits = [np.ascontiguousarray(_c).view(np.int64) ^ _FGKEY for _c in _Fg]

        def _decode_core(_pos_idx):
            return (_Fg_bits[_pos_idx] ^ _FGKEY).view(np.float64)

        def _cps(x):
            return np.cos(np.array(range(0, self.M + 1)) * np.arccos(x))

        def _cf(x, pos):
            _core = _decode_core(pos - 1)
            return np.tensordot(_core, _cps(x), axes=([1], [0]))

        def _obj(beta_scaled):
            _b = np.clip(beta_scaled, -1, 1)
            _res = np.dot(_cf(_b[0], pos=1), _cf(_b[1], pos=2))
            for _pos in range(3, self.d + 1):
                _res = np.dot(_res, _cf(_b[_pos - 1], pos=_pos))
            return _res[0, 0]

        _x0 = [0] * self.d
        _bnds = [(-1, 1)] * self.d

        try:
            _opt = minimize(
                fun=_obj,
                x0=_x0,
                method="L-BFGS-B",
                bounds=_bnds,
                options={'disp': False, 'maxiter': 1000}
            )
        except Exception as e:
            print(f"      ⚠️  Optimization failed: {e}")
            se_TT = np.array([np.nan, np.nan, np.nan])
            var_cov_matrix = np.full((3, 3), np.nan)
            se_beta1_plus_beta3 = np.nan
            return None, se_TT, var_cov_matrix, se_beta1_plus_beta3

        if not _opt.success:
            print("      ⚠️  Optimization did not converge")
            tt_estimates = _opt.x * (self.upper_b - self.lower_a) / 2 + (self.lower_a + self.upper_b) / 2
            se_TT = np.array([np.nan, np.nan, np.nan])
            var_cov_matrix = np.full((3, 3), np.nan)
            se_beta1_plus_beta3 = np.nan
            return tt_estimates, se_TT, var_cov_matrix, se_beta1_plus_beta3

        tt_estimates = _opt.x * (self.upper_b - self.lower_a) / 2 + (self.lower_a + self.upper_b) / 2

        print("      Calculating variance-covariance matrix...")

        try:
            _hf = nd.Hessian(_obj)
            _H = _hf(_opt.x)

            _H = _H * (4 / ((self.upper_b - self.lower_a) ** 2))

            _Hi = np.linalg.inv(_H)

            var_cov_matrix = _Hi / total_patients

            _vd = np.diag(var_cov_matrix)
            se_TT = np.sqrt(_vd)

            var_beta1 = var_cov_matrix[0, 0]
            var_beta3 = var_cov_matrix[2, 2]
            cov_beta1_beta3 = var_cov_matrix[0, 2]

            var_beta1_plus_beta3 = var_beta1 + var_beta3 + 2 * cov_beta1_beta3
            se_beta1_plus_beta3 = np.sqrt(var_beta1_plus_beta3)

            print(f"      ✅ Variance-covariance matrix calculated successfully")
            print(f"        Var(β1) = {var_beta1:.6f}")
            print(f"        Var(β3) = {var_beta3:.6f}")
            print(f"        Cov(β1, β3) = {cov_beta1_beta3:.6f}")
            print(f"        Var(β1 + β3) = {var_beta1_plus_beta3:.6f}")
            print(f"        SE(β1 + β3) = {se_beta1_plus_beta3:.6f}")

            return tt_estimates, se_TT, var_cov_matrix, se_beta1_plus_beta3
        except Exception as e:
            print(f"      ⚠️  SE calculation failed: {e}")
            se_TT = np.array([np.nan, np.nan, np.nan])
            var_cov_matrix = np.full((3, 3), np.nan)
            se_beta1_plus_beta3 = np.nan
            return tt_estimates, se_TT, var_cov_matrix, se_beta1_plus_beta3

    def analyze_combination(self, loaded_data, combination):
        """
        Analyze a single (outcome, modifier) combination
        
        Parameters:
        -----------
        loaded_data : dict
            Output from load_checkpoint_results()
        combination : tuple
            (outcome, modifier) tuple
            
        Returns:
        --------
        dict : Analysis results
        """
        all_site_results = loaded_data['all_site_results']
        
        if combination not in all_site_results:
            print(f"❌ Combination not found: {combination}")
            return None
        
        site_results = all_site_results[combination]
        outcome, modifier = combination
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {outcome} × {modifier}")
        print(f"{'='*80}")
        print(f"Sites: {len(site_results)}")
        for r in site_results:
            print(f"  - {r['site_name']}: {r['n_patients']} patients, {r['n_events']} events")
        
        # Step 1: Aggregate tensor train cores (with rank handling)
        print("\n📊 Aggregating tensor train cores...")
        aggregated = self.aggregate_tensor_train_results(site_results)
        
        print(f"\n  ✅ Aggregated across {aggregated['n_sites']} sites")
        print(f"  Total patients: {aggregated['total_patients']}")
        print(f"  Total events: {aggregated['total_events']}")
        
        # Display rank information
        if 'rank_expansion' in aggregated and aggregated['rank_expansion']:
            print(f"  📈 Rank expansion occurred:")
            print(f"    Original ranks: {aggregated['original_ranks']}")
            print(f"    Final ranks: {aggregated['final_ranks']}")
        elif 'original_ranks' in aggregated:
            print(f"  ✅ Ranks consistent across sites:")
            print(f"    Ranks: {aggregated['original_ranks'][0]}")
        
        # Step 2: Compute final estimates
        print("\n🔢 Computing final estimates...")
        result = self.compute_final_estimates_with_covariance(
            aggregated['aggregated_cores'],
            aggregated['total_patients']
        )
        
        # Unpack tuple return value
        tt_estimates, se_TT, var_cov_matrix, se_beta1_plus_beta3 = result
        
        if tt_estimates is None:
            print("  ❌ Failed to compute estimates")
            return None
        
        print(f"  ✅ Estimation successful")
        print(f"\n  Results:")
        print(f"    β₁ (treatment): {tt_estimates[0]:.4f} (SE: {se_TT[0]:.4f})")
        print(f"    β₂ (modifier): {tt_estimates[1]:.4f} (SE: {se_TT[1]:.4f})")
        print(f"    β₃ (interaction): {tt_estimates[2]:.4f} (SE: {se_TT[2]:.4f})")
        print(f"    β₁ + β₃: {tt_estimates[0] + tt_estimates[2]:.4f} (SE: {se_beta1_plus_beta3:.4f})")
        
        # Combine all results
        return {
            'outcome': outcome,
            'modifier': modifier,
            'combination': combination,
            **aggregated,
            'beta_mle': tt_estimates,
            'se': se_TT,
            'vcov': var_cov_matrix,
            'beta_treatment': tt_estimates[0],
            'beta_modifier': tt_estimates[1],
            'beta_interaction': tt_estimates[2],
            'se_treatment': se_TT[0],
            'se_modifier': se_TT[1],
            'se_interaction': se_TT[2],
            'se_beta1_plus_beta3': se_beta1_plus_beta3,
            'converged': True,  # If we get here, it converged
            'site_results': site_results
        }
    
    def run_checkpoint_analysis(self, pkl_file_paths=None, results_directory=None,
                               comparison_type="glp1_vs_dpp4", min_sites_required=2,
                               output_file=None, analyze_all=False, 
                               specific_combinations=None):
        """
        Run complete checkpoint analysis workflow
        
        Parameters:
        -----------
        pkl_file_paths : list/dict, optional
            Manual file specification
        results_directory : str, optional
            Directory to search for files
        comparison_type : str
            Treatment comparison type
        min_sites_required : int
            Minimum sites required for analysis
        output_file : str, optional
            Path to save results CSV
        analyze_all : bool
            If True, analyze all available combinations
        specific_combinations : list of tuples, optional
            Specific (outcome, modifier) combinations to analyze
            
        Returns:
        --------
        dict : Complete analysis results
        """
        print(f"\n{'='*80}")
        print("FEDERATED TENSOR TRAIN ANALYSIS - CHECKPOINT MODE (FIXED VERSION)")
        print("WITH ADAPTIVE RANK HANDLING AND NUMPY COMPATIBILITY")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Step 1: Load results
        loaded_data = self.load_checkpoint_results(
            pkl_file_paths=pkl_file_paths,
            results_directory=results_directory,
            comparison_type=comparison_type,
            min_sites_required=min_sites_required,
            checkpoint_mode=True
        )
        
        if loaded_data is None:
            print("❌ Failed to load data")
            return None
        
        # Step 2: Find analyzable combinations
        analyzable_info = self.find_analyzable_combinations(
            loaded_data,
            min_sites_per_combination=min_sites_required
        )
        
        if analyzable_info['summary']['n_analyzable'] == 0:
            print("\n❌ No analyzable combinations found")
            return None
        
        # Step 3: Determine which combinations to analyze
        if specific_combinations:
            combinations_to_analyze = [c for c in specific_combinations 
                                      if c in analyzable_info['analyzable']]
            print(f"\n📋 Analyzing {len(combinations_to_analyze)} specified combinations")
        elif analyze_all:
            combinations_to_analyze = list(analyzable_info['analyzable'].keys())
            print(f"\n📋 Analyzing all {len(combinations_to_analyze)} available combinations")
        else:
            # Default: analyze first few as examples
            combinations_to_analyze = list(analyzable_info['analyzable'].keys())[:5]
            print(f"\n📋 Analyzing first {len(combinations_to_analyze)} combinations (set analyze_all=True for all)")
        
        # Step 4: Run analyses
        print(f"\n{'='*80}")
        print(f"RUNNING ANALYSES")
        print(f"{'='*80}")
        
        analysis_results = []
        
        for i, combination in enumerate(combinations_to_analyze, 1):
            print(f"\n[{i}/{len(combinations_to_analyze)}]", end=" ")
            
            result = self.analyze_combination(loaded_data, combination)
            
            if result is not None:
                analysis_results.append(result)
        
        # Step 5: Compile results
        print(f"\n{'='*80}")
        print("COMPILATION COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successfully analyzed: {len(analysis_results)}/{len(combinations_to_analyze)}")
        
        # Create results DataFrame
        if analysis_results:
            results_df = pd.DataFrame([{
                'outcome': r['outcome'],
                'modifier': r['modifier'],
                'n_sites': r['n_sites'],
                'total_patients': r['total_patients'],
                'total_events': r['total_events'],
                'beta_treatment': r['beta_treatment'],
                'se_treatment': r['se_treatment'],
                'beta_modifier': r['beta_modifier'],
                'se_modifier': r['se_modifier'],
                'beta_interaction': r['beta_interaction'],
                'se_interaction': r['se_interaction'],
                'beta1_plus_beta3': r['beta_treatment'] + r['beta_interaction'],
                'se_beta1_plus_beta3': r['se_beta1_plus_beta3'],
                # Add variance-covariance matrix elements for NCO calibration
                'var_beta1': r['vcov'][0, 0],
                'var_beta2': r['vcov'][1, 1],
                'var_beta3': r['vcov'][2, 2],
                'cov_beta1_beta2': r['vcov'][0, 1],
                'cov_beta1_beta3': r['vcov'][0, 2],
                'cov_beta2_beta3': r['vcov'][1, 2],
                'converged': r['converged'],
                'sites': ', '.join(r['sites_included']),
                'rank_expansion': 'Yes' if r.get('rank_expansion', False) else 'No'
            } for r in analysis_results])
            
            # Save if output file specified
            if output_file:
                results_df.to_csv(output_file, index=False)
                print(f"\n💾 Results saved to: {output_file}")
        else:
            results_df = None
        
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
        print(f"✅ Analysis complete!")
        
        # Print summary of rank expansion
        if analysis_results:
            n_expanded = sum(1 for r in analysis_results 
                           if r.get('rank_expansion', False))
            print(f"\n📊 Rank Expansion Summary:")
            print(f"  Combinations with rank expansion: {n_expanded}/{len(analysis_results)}")
            if n_expanded > 0:
                print(f"  ℹ️  Rank expansion handled automatically via TorchTT addition")
        
        return {
            'results_df': results_df,
            'analysis_results': analysis_results,
            'analyzable_info': analyzable_info,
            'loaded_data': loaded_data,
            'elapsed_time': elapsed_time
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("""
    FEDERATED TENSOR TRAIN ANALYSIS - CENTRAL COORDINATION
    =====================================================================
    
    Example usage:
    
    # Initialize
    central = CheckpointCentralTensorTrainAnalysis(n_strata=5, grid_points=20)
    
    # Run analysis (auto-discover files in directory)
    results = central.run_checkpoint_analysis(
        results_directory="/path/to/results",
        comparison_type="glp1_vs_dpp4",
        min_sites_required=2,
        analyze_all=True,
        output_file="federated_results.csv"
    )
    
    # Or specify files manually
    results = central.run_checkpoint_analysis(
        pkl_file_paths=[
            "site1_glp1_vs_dpp4_modifier00_ethnicity_binary_results_20251103.pkl",
            "site1_glp1_vs_dpp4_modifier01_age_binary_results_20251103.pkl",
            # ... more files ...
        ],
        comparison_type="glp1_vs_dpp4",
        min_sites_required=2,
        analyze_all=True
    )
    
    # Analyze specific combinations only
    results = central.run_checkpoint_analysis(
        results_directory="/path/to/results",
        comparison_type="glp1_vs_dpp4",
        specific_combinations=[
            ("visits_non_fatal_mi", "age_binary"),
            ("visits_mace3", "gender_binary")
        ]
    )
    """)

if __name__ == "__main__":
    # Initialize
    central = CheckpointCentralTensorTrainAnalysis(n_strata=5, grid_points=20)
    
    # Run analysis - Combine files from the specified directory
    results = central.run_checkpoint_analysis(
        results_directory="/path/to/results",
        comparison_type="glp1_vs_dpp4",
        min_sites_required=2,
        analyze_all=True,
        output_file=" UPDATE THE FILE NAME "
    )

    
