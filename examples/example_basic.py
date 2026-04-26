"""
Basic example: σ/π decomposition for paradigm molecules
========================================================

Reproduces the original 8-compound benchmark demonstrated in the methodology.
Run with: python example_basic.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from sigma_pi_decomposition import analyze_decomp

results = []

# 1. Ethylene
results.append(analyze_decomp("Ethylene C=C",
    "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241",
    [0,0,-0.667], [0,0,0.667]))

# 2. Propene C=C terminal
results.append(analyze_decomp("Propene C=C",
    "C 0 0.000 -0.766; C 0 0.000 0.570; C 0 1.254 1.305; H 0 -0.930 -1.316; H 0.920 0.540 -1.316; H 0 -0.930 1.110; H 0 2.120 0.640; H 0.870 1.320 1.960; H -0.870 1.320 1.960",
    [0,0,-0.766], [0,0,0.570]))

# 3. Butadiene 1-2
buta_atoms = "C 1.853 0.682 0; C 0.611 0 0; C -0.611 0 0; C -1.853 0.682 0; H 1.853 1.770 0; H 2.810 0.220 0; H 0.611 -1.089 0; H -0.611 -1.089 0; H -1.853 1.770 0; H -2.810 0.220 0"
results.append(analyze_decomp("Butadiene 1-2", buta_atoms, [1.853, 0.682, 0], [0.611, 0, 0]))

# 4. Butadiene 2-3
results.append(analyze_decomp("Butadiene 2-3", buta_atoms, [0.611, 0, 0], [-0.611, 0, 0]))

# 5. Benzene
benz_atoms = "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0"
results.append(analyze_decomp("Benzene C-C", benz_atoms, [1.395, 0, 0], [0.698, 1.208, 0]))

# 6-8. Naphthalene
napth_atoms = """
C  1.243 0.715 0
C  0.000 1.413 0
C -1.243 0.715 0
C -1.243 -0.715 0
C  0.000 -1.413 0
C  1.243 -0.715 0
C  2.484 1.420 0
C  2.484 2.840 0
C  1.243 3.553 0
C  0.000 2.840 0
H  3.444 0.876 0
H  3.444 3.380 0
H  1.243 4.638 0
H -1.000 3.380 0
H -2.205 1.255 0
H -2.205 -1.255 0
H -1.000 -2.483 0
H  1.000 -2.483 0
"""
results.append(analyze_decomp("Naphthalene 0-1", napth_atoms, [1.243, 0.715, 0], [0.0, 1.413, 0]))
results.append(analyze_decomp("Naphthalene 0-5", napth_atoms, [1.243, 0.715, 0], [1.243, -0.715, 0]))
results.append(analyze_decomp("Naphthalene 0-6 (peri)", napth_atoms, [1.243, 0.715, 0], [2.484, 1.420, 0]))

# Summary
print("\n" + "="*100)
print("SIGMA / PI DECOMPOSITION SUMMARY")
print("="*100)
print(f"{'System':28} {'delta_Bader':>12} {'delta_pi':>10} {'delta_sigma':>12} {'recon%':>8} {'sigma/Bader':>12}")
for r in results:
    rcn = 100*(r['delta_total_recon']-r['delta_bader'])/r['delta_bader'] if r['delta_bader'] != 0 else 0
    sigma_ratio = r['delta_sigma_num']/r['delta_bader'] if r['delta_bader'] != 0 else float('nan')
    print(f"{r['name']:28} {r['delta_bader']:12.4f} {r['delta_pi_num']:10.4f} "
          f"{r['delta_sigma_num']:12.4f} {rcn:+8.1f} {sigma_ratio:+12.4f}")
