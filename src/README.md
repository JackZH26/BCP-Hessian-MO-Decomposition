# Source Code

## Core Algorithm

- **`sigma_pi_decomposition.py`** — main module with:
  - `find_BCP()`: locate bond critical point along internuclear axis
  - `rho_grad_hess()`: numerical Hessian via central differences
  - `mo_hessian_contrib_along_axis()`: per-MO contribution along principal axis
  - `classify_MO()`: σ/π classification heuristic
  - `analyze_decomp()`: full pipeline (used by all v6-v9)

## Benchmark Scripts (run sequentially)

| Script | Purpose | Compounds | Method | Basis |
|--------|---------|-----------|--------|-------|
| `v6_25_molecules_decomp.py` | Initial 25-mol benchmark | 25 | HF | 6-31G* |
| `v7_30_more_decomp.py` | Extension +28 compounds | 28 | HF | 6-31G* |
| `v8_50_more.py` | Extension +49 compounds | 49 | HF | 6-31G* |
| `v9_ccpVTZ_robustness.py` | Basis set robustness | 16 | HF | cc-pVTZ |
| `v10_b3lyp_robustness.py` | DFT/correlation robustness | 16 | B3LYP | def2-TZVP |
| `v11_ccsd_correlation.py` | Strict correlation (CCSD) | 8 | CCSD | cc-pVDZ |
| `v12_geometry_optimized.py` | Geometry optimization effect | 6 | B3LYP+opt | cc-pVTZ |
| `v13_localized_mo.py` | LMO independence | 12 | HF + Boys/PM/IBO | 6-31G* |

## Total

- 102 distinct compounds
- ~120 individual decomposition calculations
- σ/π sign rule observed in 71/72 ε > 0 cases (98.6%)
- Robust across method, basis, MO selection, geometry

## Running

Each script is self-contained. Run with:

```bash
python v6_25_molecules_decomp.py 2>&1 | tee v6_results.log
```

Computational time per compound (rough estimate):
- HF/6-31G*: 5-30 seconds
- HF/cc-pVTZ: 1-3 minutes
- B3LYP/def2-TZVP: 30 seconds - 2 minutes
- CCSD/cc-pVDZ: 1-10 minutes
- Geometry optimization: 30 seconds - 5 minutes
