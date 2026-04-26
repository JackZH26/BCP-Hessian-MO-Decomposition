# ⚠️ STATUS: Working Draft — NOT FOR SUBMISSION

**Last updated:** 2026-04-26

## Current Status

This manuscript (v1) is a **working draft** and **must not be submitted** in its current form.

## Reason

Independent peer review (Albert, 2026-04-26) identified a **fatal methodological circularity** in the σ/π classification rule:

The classification criterion (criterion 2 in §2.3) explicitly enforces 
`|∂ψ/∂ξ_1|² > |∂ψ/∂ξ_2|²` for π MOs. Combined with the
per-MO contribution formula (Eq. 5), this forces 
`c_i^(2) − c_i^(1) ≤ 0` for π MOs by construction.

Therefore the "δ_σ > 0, δ_π < 0 sign rule" reported as a
98.6% empirical finding is in fact a **mathematical tautology**
of the classification rule, not a genuine observation.

## What is Salvageable

✅ The 102-compound benchmark dataset (BCP properties at HF, B3LYP, CCSD)
✅ The PySCF implementation of per-MO Hessian decomposition
✅ 19 unit tests + GitHub Actions CI
✅ XYZ geometries for all compounds

❌ The "sign rule" claim
❌ The "reverses 35-year textbook" framing
❌ The current §2.3 σ/π classification rule

## Path Forward

Two possible paths exist; neither has been chosen:

**Path A (data-only paper, recommended):**
- Publish as a benchmark dataset paper (e.g. *Data in Brief*, *Sci. Data*)
- Report only ρ_BCP, ε, λ_i, ∇²ρ across compounds and methods
- Avoid σ/π decomposition entirely

**Path B (rigorous methodological rewrite):**
- Replace §2.3 classification with point-group-based projection (σ_h reflection or C_∞ rotation)
- Add ε_h convergence tests
- Add IQA literature comparison
- Re-run statistics; sign rule may or may not hold under proper classification

See `research/ESSENTIAL_BCP_Hessian_Decomposition_Failed_Pivot_2026-04-26.md`
for the complete failure analysis and lessons learned.

## Lessons for Future Work

1. **Sign rules require circularity self-audit** before claiming as empirical findings
2. **Same data → one paper.** No re-framing pivots.
3. **Independent peer review (Albert) is mandatory** before submission, not after
4. **Anomalous values cannot be excluded** without diagnosis (e.g. ε < 0 is a bug, not a data point to exclude)

## Authors' Note

This file is committed publicly to acknowledge the issue and to document the
research process honestly. Future versions of this manuscript, if any, will
correct the methodological problems described above.
