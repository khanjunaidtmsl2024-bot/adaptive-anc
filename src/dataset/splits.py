"""
Dataset Splitting & Generalization Protocol Engine.
PS 26052 — Adaptive Defence ANC.

Enforces zero data leakage and partitions audio into:
- TRAIN: Known speakers (001-008) + Known noise categories
- TEST_A: UNSEEN SPEAKERS (009-010) + Known noise categories (Tests voice generalization)
- TEST_B: UNSEEN NOISE RECORDINGS (Held-out recordings of known classes) (Tests environment generalization)
- TEST_C: UNSEEN NOISE CATEGORIES (shelling, gunfire, drone) (Tests out-of-distribution robustness)
"""

from typing import Dict, List, Tuple
from pathlib import Path


# Explicit, transparent noise taxonomy mapping
TAXONOMY_MAP = {
    # Stationary
    "tank_engine": "STATIONARY",
    "diesel_idle": "STATIONARY",
    "generator_hum": "STATIONARY",
    "hvac": "STATIONARY",
    
    # Periodic / Rotor
    "helicopter_rotor": "PERIODIC_ROTOR",
    "propeller": "PERIODIC_ROTOR",
    "rotor_harmonics": "PERIODIC_ROTOR",
    
    # Non-stationary / Dynamic
    "vehicle_acceleration": "NON_STATIONARY",
    "track_squeal": "NON_STATIONARY",
    "wind_turbulence": "NON_STATIONARY",
    "tactical_siren": "NON_STATIONARY",
    
    # Impulsive
    "gunfire_transient": "IMPULSIVE_DEFENCE",
    "artillery_blast": "IMPULSIVE_DEFENCE",
    "shockwave": "IMPULSIVE_DEFENCE",
    "synthetic_impulse": "SYNTHETIC_IMPULSIVE",
    
    # Synthetic baselines
    "white_noise": "SYNTHETIC_STATIONARY",
    "pink_noise": "SYNTHETIC_STATIONARY",
}


def classify_noise_provenance(noise_id: str, filepath: str = "") -> Tuple[str, str]:
    """
    Deterministically determines noise regime and provenance class.
    NEVER uses random assignment.
    
    Returns:
        (category, provenance_type) where provenance_type is 'REAL_DEFENCE', 'SYNTHETIC_DEFENCE', etc.
    """
    nid = noise_id.lower()
    path_str = str(filepath).lower()
    
    # Check taxonomy keys
    matched_cat = "SYNTHETIC_STATIONARY"
    for key, cat in TAXONOMY_MAP.items():
        if key in nid or key in path_str:
            matched_cat = cat
            break
    
    # Determine provenance type
    if "real" in path_str or "mad" in path_str:
        provenance = "REAL_DEFENCE"
    elif "synth" in nid or "synth" in path_str:
        if "impulse" in matched_cat.lower():
            provenance = "SYNTHETIC_IMPULSIVE"
        else:
            provenance = "SYNTHETIC_DEFENCE"
    else:
        provenance = "STANDARDIZED_SIMULATION"
        
    return matched_cat, provenance


def assign_split_protocol(
    speaker_id: str,
    noise_id: str,
    noise_category: str,
    is_held_out_noise_rec: bool = False
) -> str:
    """
    Assigns sample to TRAIN, TEST_A, TEST_B, or TEST_C based on strict disjoint criteria.
    
    Rules:
    1. If noise_category in ['IMPULSIVE_DEFENCE', 'DRONE_UAV'] -> TEST_C (Unseen Category)
    2. Else if speaker_id in ['SPK_009', 'SPK_010'] -> TEST_A (Unseen Speaker)
    3. Else if is_held_out_noise_rec -> TEST_B (Unseen Noise Recording)
    4. Else -> TRAIN
    """
    # Test C: Unseen Noise Category (OOD test)
    if noise_category in ["IMPULSIVE_DEFENCE", "DRONE_UAV"]:
        return "TEST_C_UNSEEN_NOISE_CATEGORY"
        
    # Test A: Unseen Speakers
    if speaker_id in ["SPK_009", "SPK_010"]:
        return "TEST_A_UNSEEN_SPEAKER"
        
    # Test B: Unseen Noise Recording
    if is_held_out_noise_rec:
        return "TEST_B_UNSEEN_NOISE_REC"
        
    return "TRAIN"
