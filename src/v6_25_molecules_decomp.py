"""
扩展验证：25 个代表化合物的 σ/π Hessian 分解
============================================

继续 2026-04-26 夜研发现：σ MOs 产生 ε，π MOs 减少 ε
扩展到 25 分子覆盖：纯 σ、双键、三键、芳香、共轭、LP、空轨道、特殊键

测试两条假说：
H1: 所有 ε > 0 系统中 δ_σ_num > 0 且 δ_π_num < 0
H2: 退化 (ε ≈ 0) 系统中 δ_π 和 δ_σ 完美抵消（数学不变性）
"""
import sys
sys.path.insert(0, '/data/.openclaw/workspace/research/experiments/cbt/2026-04-26/loop_1')
from sigma_pi_decomposition import analyze_decomp, build, find_BCP, rho_grad_hess
import numpy as np
from pyscf import gto, scf
import warnings; warnings.filterwarnings('ignore')

# 25 分子 + 完整分类
# Format: (label, atoms, A_coord, B_coord, category, expected_eps, notes)
TESTS = [
    # ── Group A: 同核 σ+π 双键 (predicts ε > 0, δ_σ > 0, δ_π < 0) ──
    ("A1 Ethylene C=C",
     "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241",
     [0,0,-0.667],[0,0,0.667], "homo σ+π", "0.45", "paradigm"),
    
    ("A2 N2H2 trans N=N",
     "N 0 0.625 0; N 0 -0.625 0; H 0.870 1.083 0; H -0.870 -1.083 0",
     [0,0.625,0],[0,-0.625,0], "homo σ+π", "NEW", "diazene"),
    
    ("A3 P2H2 trans P=P",
     "P 0 1.050 0; P 0 -1.050 0; H 1.285 1.670 0; H -1.285 -1.670 0",
     [0,1.050,0],[0,-1.050,0], "homo σ+π", "NEW", "diphosphene"),
    
    ("A4 Isobutylene C=C",
     "C 0 0.670 0; C 0 -0.670 0; H 0.920 1.265 0; H -0.920 1.265 0; C 1.000 -1.550 0; C -1.000 -1.550 0; H 1.030 -2.190 0.890; H 1.030 -2.190 -0.890; H 1.930 -0.970 0; H -1.030 -2.190 0.890; H -1.030 -2.190 -0.890; H -1.930 -0.970 0",
     [0,0.670,0],[0,-0.670,0], "homo σ+π", "NEW", "di-substituted"),
    
    # ── Group B: 异核 σ+π 双键 (predicts ε > 0) ──
    ("B1 H2CO C=O",
     "C 0 0 -0.605; O 0 0 0.599; H 0.937 0 -1.198; H -0.937 0 -1.198",
     [0,0,-0.605],[0,0,0.599], "het σ+π", "0.12", "formaldehyde"),
    
    ("B2 H2CNH C=N",
     "C 0 0 -0.640; N 0 0 0.766; H 0.942 0 -1.254; H -0.942 0 -1.254; H 0 0.962 1.249",
     [0,0,-0.640],[0,0,0.766], "het σ+π", "NEW", "imine"),
    
    ("B3 Acetone C=O",
     "C 0 0 0; O 0 0 1.221; C 1.538 0 -0.500; C -1.538 0 -0.500; H 1.916 1.018 -0.152; H 1.916 -1.018 -0.152; H 1.538 0 -1.590; H -1.916 1.018 -0.152; H -1.916 -1.018 -0.152; H -1.538 0 -1.590",
     [0,0,0],[0,0,1.221], "het σ+π", "0.09", "acetone"),
    
    # ── Group C: 三键 (predicts ε ≈ 0, degenerate, no clear σ/π split) ──
    ("C1 N2 N≡N",
     "N 0 0 -0.549; N 0 0 0.549",
     [0,0,-0.549],[0,0,0.549], "triple", "0.00", "linear"),
    
    ("C2 HCCH C≡C",
     "C 0 0 -0.601; C 0 0 0.601; H 0 0 -1.667; H 0 0 1.667",
     [0,0,-0.601],[0,0,0.601], "triple", "0.00", "acetylene"),
    
    ("C3 HCN C≡N",
     "H 0 0 -1.657; C 0 0 -0.597; N 0 0 0.563",
     [0,0,-0.597],[0,0,0.563], "triple", "0.00", ""),
    
    ("C4 CO C≡O",
     "C 0 0 -0.566; O 0 0 0.566",
     [0,0,-0.566],[0,0,0.566], "triple", "0.00", ""),
    
    # ── Group D: 芳香 + 共轭 (predicts intermediate ε) ──
    ("D1 Benzene C-C",
     "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0",
     [1.395,0,0],[0.698,1.208,0], "aromatic", "0.23", "benzene"),
    
    ("D2 Pyridine C2-C3",
     "N 0 1.134 0; C 0.977 0.569 0; C -0.977 0.569 0; C 1.197 -0.820 0; C -1.197 -0.820 0; C 0 -1.599 0; H 1.737 1.222 0; H -1.737 1.222 0; H 2.196 -1.175 0; H -2.196 -1.175 0; H 0 -2.681 0",
     [0.977,0.569,0],[1.197,-0.820,0], "aromatic", "NEW", "pyridine"),
    
    ("D3 Furan C2-C3",
     "O 0 1.209 0; C 1.157 0.374 0; C -1.157 0.374 0; C 0.710 -0.994 0; C -0.710 -0.994 0; H 2.069 0.953 0; H -2.069 0.953 0; H 1.256 -1.989 0; H -1.256 -1.989 0",
     [1.157,0.374,0],[0.710,-0.994,0], "aromatic", "NEW", "furan O-het"),
    
    ("D4 Butadiene C1-C2 (terminal)",
     "C 0 0 -1.837; C 0 0 -0.481; C 0 0 0.481; C 0 0 1.837; H 1.026 0 -2.389; H -1.026 0 -2.389; H 1.033 0 -0.033; H -1.033 0 -0.033; H 1.026 0 2.389; H -1.026 0 2.389",
     [0,0,-1.837],[0,0,-0.481], "conjugated", "0.35", "butadiene"),
    
    ("D5 Butadiene C2-C3 (central)",
     "C 0 0 -1.837; C 0 0 -0.481; C 0 0 0.481; C 0 0 1.837; H 1.026 0 -2.389; H -1.026 0 -2.389; H 1.033 0 -0.033; H -1.033 0 -0.033; H 1.026 0 2.389; H -1.026 0 2.389",
     [0,0,-0.481],[0,0,0.481], "conjugated", "low", "central single"),
    
    # ── Group E: 纯 σ 单键 (predicts ε ≈ 0, but with slight cylindrical asymmetry) ──
    ("E1 Ethane C-C",
     "C 0 0 -0.764; C 0 0 0.764; H 0 1.018 -1.160; H 0.882 -0.509 -1.160; H -0.882 -0.509 -1.160; H 0 -1.018 1.160; H -0.882 0.509 1.160; H 0.882 0.509 1.160",
     [0,0,-0.764],[0,0,0.764], "pure σ", "0.00", "sp3-sp3"),
    
    ("E2 Methane C-H",
     "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H 0.629 -0.629 -0.629; H -0.629 0.629 -0.629",
     [0,0,0],[0.629,0.629,0.629], "pure σ", "0.00", ""),
    
    ("E3 H2O O-H",
     "O 0 0 0.117; H 0.757 0 -0.469; H -0.757 0 -0.469",
     [0,0,0.117],[0.757,0,-0.469], "pure σ", "≈0.02", "polar with LP"),
    
    # ── Group F: LP-induced ε > 0 (predicts σ-only contribution, no π MO) ──
    ("F1 NH3 N-H",
     "N 0 0 0.067; H 0.939 0 -0.312; H -0.470 0.813 -0.312; H -0.470 -0.813 -0.312",
     [0,0,0.067],[0.939,0,-0.312], "LP-σ", "≈0.02", "LP on N"),
    
    ("F2 N2F4 N-N",
     "N 0 0.706 0; N 0 -0.706 0; F 1.059 1.227 0; F -1.059 1.227 0; F 1.059 -1.227 0; F -1.059 -1.227 0",
     [0,0.706,0],[0,-0.706,0], "LP-σ", "0.68", "showcase LP-induced"),
    
    ("F3 BH3 B-H",
     "B 0 0 0; H 1.193 0 0; H -0.596 1.032 0; H -0.596 -1.032 0",
     [0,0,0],[1.193,0,0], "empty p", "0.28", "empty p_z"),
    
    # ── Group G: 特殊 (allene, ketene, CO2, etc.) ──
    ("G1 Allene C=C",
     "C 0 0 0; C 0 0 1.309; C 0 0 -1.309; H 0.926 0 1.839; H -0.926 0 1.839; H 0 0.926 -1.839; H 0 -0.926 -1.839",
     [0,0,0],[0,0,1.309], "cumul", "0.54", "ε counter-example"),
    
    ("G2 Ketene C=C",
     "C 0 0 0; C 0 0 1.316; O 0 0 2.479; H 0.934 0 -0.561; H -0.934 0 -0.561",
     [0,0,0],[0,0,1.316], "cumul", "NEW", "C=C side"),
    
    ("G3 CO2 C=O",
     "C 0 0 0; O 0 0 1.161; O 0 0 -1.161",
     [0,0,0],[0,0,1.161], "linear", "0.00", "linear D∞h"),
]

# Run all and tabulate
print("="*120)
print("25 化合物 σ/π Hessian 分解扩展验证")
print("="*120)

all_results = []
for tup in TESTS:
    label, atoms, A, B, cat, exp_eps, notes = tup
    try:
        r = analyze_decomp(label, atoms, A, B)
        r['cat'] = cat
        r['exp_eps'] = exp_eps
        r['notes'] = notes
        all_results.append(r)
    except Exception as e:
        print(f"\n!!! {label} FAILED: {str(e)[:60]}")

# Summary table
print("\n" + "="*120)
print("结论汇总")
print("="*120)
print(f"{'#':5} {'分子':32} {'类':10} {'ε actual':10} {'δ_total':9} {'δ_σ':9} {'δ_π':9} {'符号正确?'}")
print("-"*120)

# Verify: H1 (δ_σ > 0 AND δ_π < 0 for ε > 0 cases)
h1_pass = 0; h1_total = 0
h2_pass = 0; h2_total = 0

for r in all_results:
    eps = r['eps']
    dpi = r['delta_pi_num']
    dsi = r['delta_sigma_num']
    dtot = r['delta_bader']
    
    if eps > 0.05:  # non-degenerate, ε > 0 case
        h1_total += 1
        sign_ok = (dsi > 0 and dpi < 0)
        if sign_ok: h1_pass += 1
        sign_str = "✓" if sign_ok else "✗"
    elif eps < 0.01:  # degenerate / cylindrical
        h2_total += 1
        # For degenerate, expect cancellation; sign of components is artifact
        sign_str = "DEGEN"
        if abs(dpi + dsi) < 0.01 * (abs(dpi) + abs(dsi)):
            h2_pass += 1
    else:
        sign_str = "border"
    
    print(f"{r['name'][:4]:5} {r['name']:32} {r.get('cat',''):10} "
          f"{eps:+9.4f}  {dtot:+8.4f} {dsi:+8.4f} {dpi:+8.4f}  {sign_str}")

print("\n" + "="*120)
print(f"假设 H1 (δ_σ>0, δ_π<0 在 ε>0 系统中): {h1_pass}/{h1_total} 通过")
print(f"假设 H2 (退化系统中 δ_π+δ_σ ≈ 0): {h2_pass}/{h2_total} 通过")
print("="*120)
