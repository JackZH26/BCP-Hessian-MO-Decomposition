"""
v18: Complete σ_h symmetry benchmark — final comprehensive run
================================================================
~ 35 planar/quasi-planar molecules across all 4 physical sub-classes:
- Class 1 (π-DOM): traditional π bonds, conjugated, aromatic
- Class 2 (σ-DOM): σ-aromaticity (small rings, Walsh orbitals)
- Class 3 (σ-DOM): substituent-dominant single bonds
- Class 4 (σ-only): no occupied π MOs (BH3, etc.)
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))
from v17_improved_detection import analyze_with_robust_mirror
import warnings; warnings.filterwarnings('ignore')

ALL_TESTS = [
    # ===== Class 1: π-bonded, conjugated, aromatic (expected π-DOM) =====
    ("Ethylene C=C",
     "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241",
     [0,0,-0.667],[0,0,0.667], "Class1"),
    ("Propene C=C",
     "C 0 0.000 -0.766; C 0 0.000 0.570; C 0 1.254 1.305; H 0 -0.930 -1.316; H 0.920 0.540 -1.316; H 0 -0.930 1.110; H 0 2.120 0.640; H 0.870 1.320 1.960; H -0.870 1.320 1.960",
     [0,0,-0.766],[0,0,0.570], "Class1"),
    ("trans-N2H2",
     "N 0 0.625 0; N 0 -0.625 0; H 0.870 1.083 0; H -0.870 -1.083 0",
     [0,0.625,0],[0,-0.625,0], "Class1"),
    ("H2CO",
     "C 0 0 -0.605; O 0 0 0.599; H 0.937 0 -1.198; H -0.937 0 -1.198",
     [0,0,-0.605],[0,0,0.599], "Class1"),
    ("Acetone C=O",
     "C 0 0 0; O 0 0 1.221; C 1.538 0 -0.500; C -1.538 0 -0.500; H 1.916 1.018 -0.152; H 1.916 -1.018 -0.152; H 1.538 0 -1.590; H -1.916 1.018 -0.152; H -1.916 -1.018 -0.152; H -1.538 0 -1.590",
     [0,0,0],[0,0,1.221], "Class1"),
    ("Benzene C-C",
     "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0",
     [1.395,0,0],[0.698,1.208,0], "Class1"),
    ("Pyridine C2-C3",
     "N 0 1.134 0; C 0.977 0.569 0; C -0.977 0.569 0; C 1.197 -0.820 0; C -1.197 -0.820 0; C 0 -1.599 0; H 1.737 1.222 0; H -1.737 1.222 0; H 2.196 -1.175 0; H -2.196 -1.175 0; H 0 -2.681 0",
     [0.977,0.569,0],[1.197,-0.820,0], "Class1"),
    ("Furan C2-C3",
     "O 0 1.209 0; C 1.157 0.374 0; C -1.157 0.374 0; C 0.710 -0.994 0; C -0.710 -0.994 0; H 2.069 0.953 0; H -2.069 0.953 0; H 1.256 -1.989 0; H -1.256 -1.989 0",
     [1.157,0.374,0],[0.710,-0.994,0], "Class1"),
    ("Pyrrole C2-C3",
     "N 0 1.136 0; C 1.117 0.393 0; C -1.117 0.393 0; C 0.708 -0.923 0; C -0.708 -0.923 0; H 0 2.132 0; H 2.108 0.816 0; H -2.108 0.816 0; H 1.327 -1.784 0; H -1.327 -1.784 0",
     [1.117,0.393,0],[0.708,-0.923,0], "Class1"),
    ("Thiophene C2-C3",
     "S 0 1.190 0; C 1.260 0.395 0; C -1.260 0.395 0; C 0.715 -0.948 0; C -0.715 -0.948 0; H 2.260 0.875 0; H -2.260 0.875 0; H 1.270 -1.840 0; H -1.270 -1.840 0",
     [1.260,0.395,0],[0.715,-0.948,0], "Class1"),
    ("Naphthalene C1-C2",
     "C 1.243 0.715 0; C 1.243 -0.715 0; C 0 -1.413 0; C -1.243 -0.715 0; C -1.243 0.715 0; C 0 1.413 0; C 2.441 -1.410 0; C 2.441 1.410 0; C -2.441 -1.410 0; C -2.441 1.410 0; H 0 -2.490 0; H 0 2.490 0; H 3.380 -0.870 0; H 3.380 0.870 0; H -3.380 -0.870 0; H -3.380 0.870 0; H 2.441 -2.490 0; H 2.441 2.490 0; H -2.441 -2.490 0; H -2.441 2.490 0",
     [1.243,0.715,0],[1.243,-0.715,0], "Class1"),
    ("Pyridazine C-C",
     "N 0 1.245 0.687; N 0 1.245 -0.687; C 0 0 1.405; C 0 0 -1.405; C 0 -1.245 0.687; C 0 -1.245 -0.687; H 0 0 2.490; H 0 0 -2.490; H 0 -2.190 1.235; H 0 -2.190 -1.235",
     [0,-1.245,0.687],[0,-1.245,-0.687], "Class1"),
    ("Pyrimidine C-C",
     "N 0 1.245 0.687; C 0 0 1.405; N 0 -1.245 0.687; C 0 -1.245 -0.687; C 0 0 -1.405; C 0 1.245 -0.687; H 0 0 2.490; H 0 -2.190 -1.235; H 0 0 -2.490; H 0 2.190 -1.235",
     [0,-1.245,-0.687],[0,0,-1.405], "Class1"),
    ("Imidazole C2-N3",
     "N 0 0 1.058; C 0 1.092 0.396; N 0 0.738 -0.881; C 0 -0.738 -0.881; C 0 -1.092 0.396; H 0 0 2.060; H 0 2.060 0.870; H 0 -2.060 0.870; H 0 -1.260 -1.640",
     [0,1.092,0.396],[0,0.738,-0.881], "Class1"),
    ("Acrolein C=C",
     "C 0 0 0; C 0 0 1.347; C -1.305 0 2.091; O -2.485 0 1.755; H 0.926 0 -0.530; H -0.926 0 -0.530; H 0.620 0 1.890; H -1.117 0 3.180",
     [0,0,0],[0,0,1.347], "Class1"),
    ("trans-CH3N=NCH3",
     "N 0 0.625 0; N 0 -0.625 0; C 1.300 1.255 0; C -1.300 -1.255 0; H 1.300 1.880 0.890; H 1.300 1.880 -0.890; H 2.180 0.620 0; H -1.300 -1.880 -0.890; H -1.300 -1.880 0.890; H -2.180 -0.620 0",
     [0,0.625,0],[0,-0.625,0], "Class1"),
    ("trans-CHF=CHF",
     "C 0 -0.670 0; C 0 0.670 0; F 1.140 -1.290 0; F -1.140 1.290 0; H -0.940 -1.260 0; H 0.940 1.260 0",
     [0,-0.670,0],[0,0.670,0], "Class1"),
    ("CF2=CF2",
     "C 0 0 -0.660; C 0 0 0.660; F 1.080 0 -1.380; F -1.080 0 -1.380; F 1.080 0 1.380; F -1.080 0 1.380",
     [0,0,-0.660],[0,0,0.660], "Class1"),
    ("trans-N2F2",
     "N 0 0.587 0; N 0 -0.587 0; F 1.106 1.127 0; F -1.106 -1.127 0",
     [0,0.587,0],[0,-0.587,0], "Class1"),
    ("Butadiene 1-2",
     "C 0 0 -1.837; C 0 0 -0.481; C 0 0 0.481; C 0 0 1.837; H 1.026 0 -2.389; H -1.026 0 -2.389; H 1.033 0 -0.033; H -1.033 0 -0.033; H 1.026 0 2.389; H -1.026 0 2.389",
     [0,0,-1.837],[0,0,-0.481], "Class1"),
    ("HNO3 N=O",
     "N 0 0 0; O 1.062 0.876 0; O -1.062 0.876 0; O 0 -1.250 0; H 1.062 1.852 0",
     [0,0,0],[1.062,0.876,0], "Class1"),
    ("SO2",
     "S 0 0 0; O 1.252 0 0.863; O -1.252 0 0.863",
     [0,0,0],[1.252,0,0.863], "Class1"),
    ("SO3",
     "S 0 0 0; O 1.430 0 0; O -0.715 1.238 0; O -0.715 -1.238 0",
     [0,0,0],[1.430,0,0], "Class1"),
    ("Anthracene C1-C2",
     "C 0 0.722 1.230; C 0 -0.722 1.230; C 0 1.413 0; C 0 -1.413 0; C 0 0.722 -1.230; C 0 -0.722 -1.230; C 0 0.722 2.450; C 0 -0.722 2.450; C 0 0.722 -2.450; C 0 -0.722 -2.450; C 0 2.495 -2.460; C 0 -2.495 -2.460; C 0 2.495 2.460; C 0 -2.495 2.460; H 0 1.262 -3.395; H 0 -1.262 -3.395; H 0 1.262 3.395; H 0 -1.262 3.395; H 0 2.495 -1.380; H 0 -2.495 -1.380; H 0 2.495 1.380; H 0 -2.495 1.380; H 0 2.495 0; H 0 -2.495 0",
     [0,0.722,1.230],[0,-0.722,1.230], "Class1"),
    
    # ===== Class 2: σ-aromaticity rings (expected σ-DOM) =====
    ("Cyclopropane C-C",
     "C 0 0.866 0; C -0.750 -0.433 0; C 0.750 -0.433 0; H 0 1.480 0.890; H 0 1.480 -0.890; H -1.282 -0.740 0.890; H -1.282 -0.740 -0.890; H 1.282 -0.740 0.890; H 1.282 -0.740 -0.890",
     [0,0.866,0],[-0.750,-0.433,0], "Class2"),
    ("Oxirane C-O",
     "O 0 0 0.520; C 0.730 0 -0.310; C -0.730 0 -0.310; H 1.260 0.910 -0.585; H 1.260 -0.910 -0.585; H -1.260 0.910 -0.585; H -1.260 -0.910 -0.585",
     [0.730,0,-0.310],[0,0,0.520], "Class2"),
    ("Aziridine C-N",
     "N 0 0 0.580; C 0.740 0 -0.280; C -0.740 0 -0.280; H 0 0.840 1.080; H 1.260 0.910 -0.555; H 1.260 -0.910 -0.555; H -1.260 0.910 -0.555; H -1.260 -0.910 -0.555",
     [0.740,0,-0.280],[0,0,0.580], "Class2"),
    ("Thiirane C-S",
     "S 0 0 0.890; C 0.745 0 -0.500; C -0.745 0 -0.500; H 1.260 0.910 -0.785; H 1.260 -0.910 -0.785; H -1.260 0.910 -0.785; H -1.260 -0.910 -0.785",
     [0.745,0,-0.500],[0,0,0.890], "Class2"),
    
    # ===== Class 3: Substituent-dominant single bonds (expected σ-DOM) =====
    ("Fluorobenzene C-F",
     "F 0 0 0; C 0 0 1.349; C 1.215 0 2.054; C -1.215 0 2.054; C 1.215 0 3.451; C -1.215 0 3.451; C 0 0 4.155; H 2.150 0 1.522; H -2.150 0 1.522; H 2.150 0 3.985; H -2.150 0 3.985; H 0 0 5.230",
     [0,0,0],[0,0,1.349], "Class3"),
    ("N2F4 (planar trans) N-N",
     "N 0 0.706 0; N 0 -0.706 0; F 1.059 1.227 0; F -1.059 1.227 0; F 1.059 -1.227 0; F -1.059 -1.227 0",
     [0,0.706,0],[0,-0.706,0], "Class3"),
    
    # ===== Class 4: σ-only systems (no π MOs) =====
    ("BH3 B-H",
     "B 0 0 0; H 1.193 0 0; H -0.596 1.032 0; H -0.596 -1.032 0",
     [0,0,0],[1.193,0,0], "Class4"),
    ("BF3 B-F",
     "B 0 0 0; F 1.307 0 0; F -0.654 1.133 0; F -0.654 -1.133 0",
     [0,0,0],[1.307,0,0], "Class4"),
]

print("="*135)
print(f"v18 Complete Benchmark: σ_h symmetry classification across 4 physical sub-classes")
print("="*135)
print(f"{'Compound':30} {'Class':6} {'plane':4} {'mode':16} {'ε':>8} {'δ_σ':>10} {'δ_π':>10} {'pattern':>12}")
print("-"*135)

results = []
for label, atoms, A, B, cls in ALL_TESTS:
    try:
        r = analyze_with_robust_mirror(label, atoms, A, B)
        if 'error' in r:
            print(f"{label[:30]:30} {cls:6} ERROR: {r['error'][:40]}")
            continue
        r['expected_class'] = cls
        results.append(r)
        
        if r['n_pi'] == 0:
            pattern = "σ-only"
        elif r['delta_pi'] > 0 and r['delta_sigma'] < 0:
            pattern = "🔴 PI-DOM"
        elif r['delta_sigma'] > 0 and r['delta_pi'] < 0:
            pattern = "🔵 SIG-DOM"
        else:
            pattern = "MIXED"
        
        print(f"{label[:30]:30} {cls:6} {r['plane']:4} {r['plane_mode']:16} {r['eps']:+8.4f} "
              f"{r['delta_sigma']:+10.4f} {r['delta_pi']:+10.4f} {pattern:>12}")
    except Exception as e:
        print(f"{label[:30]:30} {cls:6} ERROR: {str(e)[:40]}")

print("\n" + "="*135)
print("CLASSIFICATION RESULTS")
print("="*135)
class_stats = {'Class1': {'pi': 0, 'sig': 0, 'sonly': 0, 'mixed': 0, 'total': 0},
               'Class2': {'pi': 0, 'sig': 0, 'sonly': 0, 'mixed': 0, 'total': 0},
               'Class3': {'pi': 0, 'sig': 0, 'sonly': 0, 'mixed': 0, 'total': 0},
               'Class4': {'pi': 0, 'sig': 0, 'sonly': 0, 'mixed': 0, 'total': 0}}
for r in results:
    c = r['expected_class']
    class_stats[c]['total'] += 1
    if r['n_pi'] == 0:
        class_stats[c]['sonly'] += 1
    elif r['delta_pi'] > 0 and r['delta_sigma'] < 0:
        class_stats[c]['pi'] += 1
    elif r['delta_sigma'] > 0 and r['delta_pi'] < 0:
        class_stats[c]['sig'] += 1
    else:
        class_stats[c]['mixed'] += 1

print(f"\n{'Class':10} {'Description':40} {'N':>4} {'π-DOM':>8} {'σ-DOM':>8} {'σ-only':>8} {'Mixed':>6}")
print("-"*90)
for cls, name in [('Class1', 'π-bonded/conjugated/aromatic'), 
                   ('Class2', 'σ-aromaticity rings'),
                   ('Class3', 'substituent-dominant single'),
                   ('Class4', 'σ-only (no π MOs)')]:
    s = class_stats[cls]
    print(f"{cls:10} {name:40} {s['total']:>4} {s['pi']:>8} {s['sig']:>8} {s['sonly']:>8} {s['mixed']:>6}")

# Save CSV
csv_path = 'v18_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['name', 'expected_class', 'plane', 'plane_mode', 'eps',
                                       'delta_bader', 'delta_sigma', 'delta_pi', 'delta_mixed',
                                       'n_sigma', 'n_pi', 'n_mixed'])
    w.writeheader()
    for r in results:
        w.writerow({k: r.get(k) for k in w.fieldnames})

print(f"\nSaved to {csv_path}")
