"""
Path B v15: σ_h symmetry classification on ALL planar molecules
================================================================
Test the sign rule under independent group-theoretic classification
across the planar/linear subset of the 102-compound benchmark.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from v14_symmetry_projection import analyze_symmetry_decomp, detect_mirror_plane
import warnings; warnings.filterwarnings('ignore')

# Planar molecules from v6/v7/v8 (selected for valid σ_h plane)
PLANAR_TESTS = [
    # Group A: Homonuclear σ+π double bonds (planar, π perpendicular)
    ("Ethylene C=C",
     "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241",
     [0,0,-0.667],[0,0,0.667]),
    
    ("trans-N2H2 N=N",
     "N 0 0.625 0; N 0 -0.625 0; H 0.870 1.083 0; H -0.870 -1.083 0",
     [0,0.625,0],[0,-0.625,0]),
    
    ("trans-Diazene N=N (alt)",
     "N 0.625 0 0; N -0.625 0 0; H 1.083 0.870 0; H -1.083 -0.870 0",
     [0.625,0,0],[-0.625,0,0]),
    
    # Group B: Heteronuclear σ+π (C=O, C=N)
    ("H2CO C=O",
     "C 0 0 -0.605; O 0 0 0.599; H 0.937 0 -1.198; H -0.937 0 -1.198",
     [0,0,-0.605],[0,0,0.599]),
    
    # Group D: Aromatic (benzene)
    ("Benzene C-C",
     "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0",
     [1.395,0,0],[0.698,1.208,0]),
    
    # Group D: Pyridine, Furan
    ("Pyridine C2-C3",
     "N 0 1.134 0; C 0.977 0.569 0; C -0.977 0.569 0; C 1.197 -0.820 0; C -1.197 -0.820 0; C 0 -1.599 0; H 1.737 1.222 0; H -1.737 1.222 0; H 2.196 -1.175 0; H -2.196 -1.175 0; H 0 -2.681 0",
     [0.977,0.569,0],[1.197,-0.820,0]),
    
    ("Furan C2-C3",
     "O 0 1.209 0; C 1.157 0.374 0; C -1.157 0.374 0; C 0.710 -0.994 0; C -0.710 -0.994 0; H 2.069 0.953 0; H -2.069 0.953 0; H 1.256 -1.989 0; H -1.256 -1.989 0",
     [1.157,0.374,0],[0.710,-0.994,0]),
    
    # Group H: Halogenated planar
    ("trans-CHF=CHF",
     "C 0 -0.670 0; C 0 0.670 0; F 1.140 -1.290 0; F -1.140 1.290 0; H -0.940 -1.260 0; H 0.940 1.260 0",
     [0,-0.670,0],[0,0.670,0]),
    
    # Group F: LP-induced (N2F4 is planar in trans configuration)
    ("N2F4 N-N (planar trans)",
     "N 0 0.706 0; N 0 -0.706 0; F 1.059 1.227 0; F -1.059 1.227 0; F 1.059 -1.227 0; F -1.059 -1.227 0",
     [0,0.706,0],[0,-0.706,0]),
    
    # Group F: BH3 (planar, D_3h)
    ("BH3 B-H",
     "B 0 0 0; H 1.193 0 0; H -0.596 1.032 0; H -0.596 -1.032 0",
     [0,0,0],[1.193,0,0]),
    
    # Sulfur dioxide
    ("SO2 S=O",
     "S 0 0 0; O 1.252 0 0.863; O -1.252 0 0.863",
     [0,0,0],[1.252,0,0.863]),
    
    # Naphthalene 0-1
    ("Naphthalene 0-1",
     "C 1.243 0.715 0; C 1.243 -0.715 0; C 0 -1.413 0; C -1.243 -0.715 0; C -1.243 0.715 0; C 0 1.413 0; C 2.441 -1.410 0; C 2.441 1.410 0; C -2.441 -1.410 0; C -2.441 1.410 0; H 0 -2.490 0; H 0 2.490 0; H 3.380 -0.870 0; H 3.380 0.870 0; H -3.380 -0.870 0; H -3.380 0.870 0; H 2.441 -2.490 0; H 2.441 2.490 0; H -2.441 -2.490 0; H -2.441 2.490 0",
     [1.243,0.715,0],[1.243,-0.715,0]),
    
    # Pyrrole, Thiophene
    ("Pyrrole C2-C3",
     "N 0 1.136 0; C 1.117 0.393 0; C -1.117 0.393 0; C 0.708 -0.923 0; C -0.708 -0.923 0; H 0 2.132 0; H 2.108 0.816 0; H -2.108 0.816 0; H 1.327 -1.784 0; H -1.327 -1.784 0",
     [1.117,0.393,0],[0.708,-0.923,0]),
    
    ("Thiophene C2-C3",
     "S 0 1.190 0; C 1.260 0.395 0; C -1.260 0.395 0; C 0.715 -0.948 0; C -0.715 -0.948 0; H 2.260 0.875 0; H -2.260 0.875 0; H 1.270 -1.840 0; H -1.840 -1.270 0",
     [1.260,0.395,0],[0.715,-0.948,0]),
]

print("="*120)
print("v15 Path B: σ_h-symmetry σ/π classification on planar benchmark")
print("="*120)
print(f"{'Compound':30} {'mirror':4} {'ε':>8} {'δ_tot':>8} {'δ_σ_sym':>10} {'δ_π_sym':>10} "
      f"{'n_σ':>4} {'n_π':>4} {'sign rule?':>12}")
print("-"*120)

n_sigma_pos = 0
n_sigma_neg = 0
n_pi_pos = 0
n_pi_neg = 0
n_total = 0
n_sigma_only = 0
n_pi_dominant = 0  # δ_π > 0 (textbook expectation)
n_sigma_dominant = 0  # δ_σ > 0 (v1 paper claim)

results = []
for label, atoms, A, B in PLANAR_TESTS:
    try:
        r = analyze_symmetry_decomp(label, atoms, A, B)
        if 'error' in r:
            print(f"{label:30} {'?':4} ERROR: {r['error']}")
            continue
        results.append(r)
        n_total += 1
        
        # Determine pattern
        if r['n_pi'] == 0:
            pattern = "σ-only"
            n_sigma_only += 1
        elif r['delta_pi'] > 0 and r['delta_sigma'] < 0:
            pattern = "🔴 PI-DOM"  # textbook expectation
            n_pi_dominant += 1
        elif r['delta_sigma'] > 0 and r['delta_pi'] < 0:
            pattern = "🔵 SIG-DOM"  # v1 paper claim
            n_sigma_dominant += 1
        elif r['delta_pi'] > 0 and r['delta_sigma'] > 0:
            pattern = "BOTH+"
        else:
            pattern = "OTHER"
        
        if r['delta_sigma'] > 0: n_sigma_pos += 1
        else: n_sigma_neg += 1
        if r['delta_pi'] > 0: n_pi_pos += 1
        else: n_pi_neg += 1
        
        print(f"{label[:30]:30} {r['mirror']:4} {r['eps']:+8.4f} {r['delta_bader']:+8.4f} "
              f"{r['delta_sigma']:+10.4f} {r['delta_pi']:+10.4f} "
              f"{r['n_sigma']:4d} {r['n_pi']:4d} {pattern:>12}")
    except Exception as e:
        print(f"{label[:30]:30} ERROR: {str(e)[:60]}")

print("\n" + "="*120)
print("PATTERN ANALYSIS UNDER GROUP-THEORETIC CLASSIFICATION")
print("="*120)
print(f"  Total non-degenerate: {n_total}")
print(f"  σ-only systems (no π MO): {n_sigma_only}")
print(f"  🔴 PI-dominant (δ_π > 0, δ_σ < 0)  [TEXTBOOK]: {n_pi_dominant}")
print(f"  🔵 σ-dominant (δ_σ > 0, δ_π < 0)  [v1 PAPER]: {n_sigma_dominant}")
print()
print(f"  δ_σ > 0 in {n_sigma_pos}/{n_total} = {100*n_sigma_pos/max(n_total,1):.1f}%")
print(f"  δ_π > 0 in {n_pi_pos}/{n_total} = {100*n_pi_pos/max(n_total,1):.1f}%")
print()
print("="*120)
if n_pi_dominant > n_sigma_dominant:
    print("🔴 CONCLUSION: TEXTBOOK INTERPRETATION CONFIRMED — π MOs PRODUCE ε")
    print("   v1 paper claim INVERTED under proper symmetry classification.")
elif n_sigma_dominant > n_pi_dominant:
    print("🔵 CONCLUSION: v1 PAPER CLAIM SURVIVES — σ MOs produce ε")
else:
    print("⚪ MIXED: no dominant pattern")
print("="*120)
