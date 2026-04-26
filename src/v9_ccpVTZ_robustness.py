"""
v9：cc-pVTZ 基底 robustness 测试
================================
关键 16 化合物用 cc-pVTZ 重测，验证结论符号 (δ_σ > 0, δ_π < 0) 是否基底 robust

选择标准：
- 4 paradigm σ+π (ethylene, N2H2, formaldehyde, acetone)
- 3 退化 (acetylene, N2, ethane)  
- 3 应变/反例 (cyclopropane, allene, BH3)
- 3 芳香 (benzene, pyridine, furan)
- 2 LP-σ (N2F4, NH3)
- 1 卤代 (CHF=CHF)
"""
import sys
sys.path.insert(0, '/data/.openclaw/workspace/research/experiments/cbt/2026-04-26/loop_1')
from sigma_pi_decomposition import analyze_decomp
import warnings; warnings.filterwarnings('ignore')

KEY_TESTS = [
    # 4 paradigm
    ("Ethylene C=C", "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241", [0,0,-0.667],[0,0,0.667]),
    ("N2H2 trans N=N", "N 0 0.625 0; N 0 -0.625 0; H 0.870 1.083 0; H -0.870 -1.083 0", [0,0.625,0],[0,-0.625,0]),
    ("H2CO C=O", "C 0 0 -0.605; O 0 0 0.599; H 0.937 0 -1.198; H -0.937 0 -1.198", [0,0,-0.605],[0,0,0.599]),
    ("Acetone C=O", "C 0 0 0; O 0 0 1.221; C 1.538 0 -0.500; C -1.538 0 -0.500; H 1.916 1.018 -0.152; H 1.916 -1.018 -0.152; H 1.538 0 -1.590; H -1.916 1.018 -0.152; H -1.916 -1.018 -0.152; H -1.538 0 -1.590", [0,0,0],[0,0,1.221]),
    
    # 3 degenerate
    ("HCCH C≡C", "C 0 0 -0.601; C 0 0 0.601; H 0 0 -1.667; H 0 0 1.667", [0,0,-0.601],[0,0,0.601]),
    ("N2 N≡N", "N 0 0 -0.549; N 0 0 0.549", [0,0,-0.549],[0,0,0.549]),
    ("Ethane C-C", "C 0 0 -0.764; C 0 0 0.764; H 0 1.018 -1.160; H 0.882 -0.509 -1.160; H -0.882 -0.509 -1.160; H 0 -1.018 1.160; H -0.882 0.509 1.160; H 0.882 0.509 1.160", [0,0,-0.764],[0,0,0.764]),
    
    # 3 strained / counter
    ("Cyclopropane C-C", "C 0 0.866 0; C -0.750 -0.433 0; C 0.750 -0.433 0; H 0 1.480 0.890; H 0 1.480 -0.890; H -1.282 -0.740 0.890; H -1.282 -0.740 -0.890; H 1.282 -0.740 0.890; H 1.282 -0.740 -0.890", [0,0.866,0],[-0.750,-0.433,0]),
    ("Allene C=C", "C 0 0 0; C 0 0 1.309; C 0 0 -1.309; H 0.926 0 1.839; H -0.926 0 1.839; H 0 0.926 -1.839; H 0 -0.926 -1.839", [0,0,0],[0,0,1.309]),
    ("BH3 B-H", "B 0 0 0; H 1.193 0 0; H -0.596 1.032 0; H -0.596 -1.032 0", [0,0,0],[1.193,0,0]),
    
    # 3 aromatic
    ("Benzene C-C", "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0", [1.395,0,0],[0.698,1.208,0]),
    ("Pyridine C2-C3", "N 0 1.134 0; C 0.977 0.569 0; C -0.977 0.569 0; C 1.197 -0.820 0; C -1.197 -0.820 0; C 0 -1.599 0; H 1.737 1.222 0; H -1.737 1.222 0; H 2.196 -1.175 0; H -2.196 -1.175 0; H 0 -2.681 0", [0.977,0.569,0],[1.197,-0.820,0]),
    ("Furan C2-C3", "O 0 1.209 0; C 1.157 0.374 0; C -1.157 0.374 0; C 0.710 -0.994 0; C -0.710 -0.994 0; H 2.069 0.953 0; H -2.069 0.953 0; H 1.256 -1.989 0; H -1.256 -1.989 0", [1.157,0.374,0],[0.710,-0.994,0]),
    
    # 2 LP-σ
    ("N2F4 N-N", "N 0 0.706 0; N 0 -0.706 0; F 1.059 1.227 0; F -1.059 1.227 0; F 1.059 -1.227 0; F -1.059 -1.227 0", [0,0.706,0],[0,-0.706,0]),
    ("NH3 N-H", "N 0 0 0.067; H 0.939 0 -0.312; H -0.470 0.813 -0.312; H -0.470 -0.813 -0.312", [0,0,0.067],[0.939,0,-0.312]),
    
    # 1 halogenated
    ("trans-CHF=CHF C=C", "C 0 -0.670 0; C 0 0.670 0; F 1.140 -1.290 0; F -1.140 1.290 0; H -0.940 -1.260 0; H 0.940 1.260 0", [0,-0.670,0],[0,0.670,0]),
]

print("="*120)
print("v9: cc-pVTZ 基底 robustness 测试 (16 关键化合物)")
print("="*120)

results_pVTZ = []
for label, atoms, A, B in KEY_TESTS:
    try:
        print(f"\n>>> {label}")
        r = analyze_decomp(label, atoms, A, B, basis='cc-pVTZ')
        results_pVTZ.append(r)
    except Exception as e:
        print(f"!!! {label} FAILED: {str(e)[:100]}")

print("\n" + "="*120)
print("v9 cc-pVTZ vs 6-31G* 比较")
print("="*120)
print(f"{'分子':32} {'ε pVTZ':9} {'δ_σ pVTZ':10} {'δ_π pVTZ':10} {'符号'}")
print("-"*120)

h1_pass = 0; h1_total = 0
h2_pass = 0; h2_total = 0

for r in results_pVTZ:
    eps = r['eps']
    dpi = r['delta_pi_num']
    dsi = r['delta_sigma_num']
    
    if eps > 0.05:
        h1_total += 1
        if abs(dpi) < 0.005:
            sign_str = "σ-only"
            if dsi > 0: h1_pass += 1
        elif dsi > 0 and dpi < 0:
            sign_str = "✓"
            h1_pass += 1
        else:
            sign_str = "✗"
    elif eps < 0.01:
        h2_total += 1
        if abs(dpi + dsi) < 0.01 + 0.01*max(abs(dpi), abs(dsi)):
            sign_str = "DEGEN✓"
            h2_pass += 1
        else:
            sign_str = "DEGEN?"
    else:
        sign_str = "border"
    
    print(f"{r['name']:32} {eps:+8.4f}  {dsi:+9.4f}  {dpi:+9.4f}  {sign_str}")

print("\n" + "="*120)
print(f"v9 cc-pVTZ 假设验证：H1 {h1_pass}/{h1_total}, H2 {h2_pass}/{h2_total}")
print("="*120)
