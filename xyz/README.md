# XYZ Geometries

The molecular geometries used in the benchmark are embedded directly in the Python scripts (src/v6_*.py through v8_*.py) as PySCF atom strings.

To extract a specific geometry as XYZ format, use:

```python
from pyscf import gto

# Example: get ethylene geometry
atoms = "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241"
mol = gto.M(atom=atoms, basis='6-31G*')

# Print as XYZ
print(mol.atom_coords() * 0.529177)  # Bohr to Angstrom
```

Geometry sources:
- v6/v7/v8: hand-typed from experimental/standard bond lengths (NIST WebBook, CCCBDB)
- v12: B3LYP/cc-pVTZ optimized

For full list of compounds and bonds analyzed, see `docs/FINAL_102mol_results.md`.
