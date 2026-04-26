# BCP Hessian MO Decomposition

> Per-Molecular-Orbital Decomposition of the QTAIM Bond Critical Point (BCP) Density Hessian: Algorithm and 102-Compound Benchmark Data

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/Figshare-10.6084%2Fm9.figshare.32088900-blue)](https://doi.org/10.6084/m9.figshare.32088900)

**Author:** Jian Zhou, Founder & Director, JZ Institute of Science (JZIS)
**Email:** jack@jzis.org
**ORCID:** [0009-0000-3536-9500](https://orcid.org/0009-0000-3536-9500)

---

## Summary

This repository contains the algorithm and benchmark data for systematic per-Molecular-Orbital (MO) decomposition of the BCP density Hessian eigenvalues in QTAIM (Bader's Quantum Theory of Atoms in Molecules).

We test 102 compounds across 25+ chemical bond classes and find that:

- **σ-symmetry MOs** consistently give positive contribution to δ ≡ λ₂ − λ₁ (δ_σ > 0)
- **π-symmetry MOs** consistently give negative contribution (δ_π < 0)
- This pattern holds across 4 basis sets, 3 theoretical methods, 4 MO localization schemes, and geometry optimization

The result quantitatively refines and extends qualitative pictures based on Walsh orbitals (Walsh 1947) and σ-aromaticity (Cremer 1988, Brown-Bader 2009).

---

## Quick Start

```bash
# Install dependencies
pip install pyscf numpy scipy

# Run the algorithm on ethylene
cd src
python sigma_pi_decomposition.py
```

---

## Repository Structure

```
.
├── README.md                       # This file
├── LICENSE                         # CC BY 4.0
├── requirements.txt                # Python dependencies
├── src/                            # Source code
│   ├── sigma_pi_decomposition.py   # Core algorithm (v1)
│   ├── v6_25_molecules_decomp.py   # v6: 25-mol benchmark (HF/6-31G*)
│   ├── v7_30_more_decomp.py        # v7: +28 mol benchmark
│   ├── v8_50_more.py               # v8: +49 mol benchmark
│   ├── v9_ccpVTZ_robustness.py     # v9: 16-mol cc-pVTZ HF
│   ├── v10_b3lyp_robustness.py     # v10: 16-mol B3LYP/def2-TZVP
│   ├── v11_ccsd_correlation.py    # v11: 8-mol CCSD/cc-pVDZ
│   ├── v12_geometry_optimized.py  # v12: 6-mol B3LYP optimized geometry
│   └── v13_localized_mo.py         # v13: Localized MO (Boys/PM/IBO)
├── data/                           # Geometry data (XYZ format)
├── results/                        # Raw computational outputs (logs)
├── docs/                           # Detailed reports
│   ├── ROBUSTNESS_SUMMARY.md       # 5-dimension robustness summary
│   └── FINAL_102mol_results.md     # 102-compound result table
├── xyz/                            # XYZ geometries used
└── literature/                     # Cited literature notes
```

---

## Methods

The core algorithm decomposes the BCP density Hessian as:

```
∂²ρ/∂x_q² |_BCP = Σ_i 2n_i [(∂ψ_i/∂x_q)² + ψ_i ∂²ψ_i/∂x_q²]
```

where n_i is the occupation number of MO ψ_i.

For each occupied MO, a contribution to the eigenvalue λ_q is computed numerically along the Hessian principal axis q. MOs are classified as σ or π based on geometric criteria at the BCP.

---

## Reproducibility

All computations use [PySCF](https://pyscf.org) (Python-based Simulations of Chemistry Framework) version 2.x. Geometries are provided as XYZ coordinates. All scripts are deterministic (no random seeds).

To reproduce a specific result:

```bash
# Reproduce v6 (25 molecules at HF/6-31G*)
python src/v6_25_molecules_decomp.py | tee results/v6_results.log

# Reproduce v9 (16 molecules at HF/cc-pVTZ)
python src/v9_ccpVTZ_robustness.py | tee results/v9_results.log
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{Zhou_BCP_Hessian_2026,
  author = {Zhou, Jian},
  title = {Quantifying the σ-Framework Origin of QTAIM Bond Ellipticity:
          Per-MO Hessian Decomposition Across 102 Compounds and Multiple
          Theoretical Methods},
  year = {2026},
  note = {Manuscript in preparation},
  url = {https://github.com/JackZH26/BCP-Hessian-MO-Decomposition}
}
```

Data archive: [Figshare DOI 10.6084/m9.figshare.32088900](https://doi.org/10.6084/m9.figshare.32088900)

---

## Key References

1. Bader, R. F. W. *Atoms in Molecules: A Quantum Theory*. Oxford University Press, 1990.
2. Walsh, A. D. *Trans. Faraday Soc.* **1947**, 43, 60.
3. Cremer, D.; Gauss, J.; Cremer, E. *J. Mol. Struct. THEOCHEM* **1988**, 169, 531.
4. Bader, R. F. W.; Gatti, C. *Chem. Phys. Lett.* **1998**, 287, 233.
5. Farrugia, L. J.; Macchi, P. *J. Phys. Chem. A* **2009**, 113, 10058.
6. Brown, E. C.; Bader, R. F. W.; Werstiuk, N. H. *J. Phys. Chem. A* **2009**, 113, 3254.
7. Silva Lopez, C.; de Lera, A. R. *Curr. Org. Chem.* **2011**, 15, 3576.

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Free to use with attribution.

---

## Contact

For questions or collaboration: jack@jzis.org

---

*Repository created: 2026-04-26*
*JZ Institute of Science | jzis.org*
