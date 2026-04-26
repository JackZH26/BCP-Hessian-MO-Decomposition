"""
Unit tests for sigma_pi_decomposition module
=============================================

Verifies:
1. Mathematical reconstruction identity (delta_pi + delta_sigma = lambda_2 - lambda_1)
2. Sign rule on paradigm cases (delta_sigma > 0, delta_pi < 0 for ε > 0)
3. Degenerate cases (ε ≈ 0 → delta_sigma + delta_pi = 0)
4. σ-only systems (BH3 has delta_pi = 0)
5. Counterexample unification (N2F4, cyclopropane)

Run with:
    pytest tests/test_decomposition.py -v

Or:
    python -m pytest tests/ -v

Tests use HF/6-31G* (fast, deterministic). Total runtime ~3-5 minutes.
"""
import sys
import os
import pytest
import numpy as np

# Make src/ importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from sigma_pi_decomposition import analyze_decomp, classify_MO


# ============================================================
# Test fixtures
# ============================================================

@pytest.fixture(scope="module")
def ethylene():
    return analyze_decomp(
        "Ethylene C=C",
        "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; "
        "H 0.921 0 1.241; H -0.921 0 1.241",
        [0, 0, -0.667], [0, 0, 0.667]
    )


@pytest.fixture(scope="module")
def ethane():
    return analyze_decomp(
        "Ethane C-C",
        "C 0 0 -0.764; C 0 0 0.764; H 0 1.018 -1.160; H 0.882 -0.509 -1.160; "
        "H -0.882 -0.509 -1.160; H 0 -1.018 1.160; H -0.882 0.509 1.160; "
        "H 0.882 0.509 1.160",
        [0, 0, -0.764], [0, 0, 0.764]
    )


@pytest.fixture(scope="module")
def acetylene():
    return analyze_decomp(
        "Acetylene C-C",
        "C 0 0 -0.601; C 0 0 0.601; H 0 0 -1.667; H 0 0 1.667",
        [0, 0, -0.601], [0, 0, 0.601]
    )


@pytest.fixture(scope="module")
def benzene():
    return analyze_decomp(
        "Benzene C-C",
        "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; "
        "C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; "
        "H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0",
        [1.395, 0, 0], [0.698, 1.208, 0]
    )


@pytest.fixture(scope="module")
def cyclopropane():
    return analyze_decomp(
        "Cyclopropane C-C",
        "C 0 0.866 0; C -0.750 -0.433 0; C 0.750 -0.433 0; "
        "H 0 1.480 0.890; H 0 1.480 -0.890; "
        "H -1.282 -0.740 0.890; H -1.282 -0.740 -0.890; "
        "H 1.282 -0.740 0.890; H 1.282 -0.740 -0.890",
        [0, 0.866, 0], [-0.750, -0.433, 0]
    )


@pytest.fixture(scope="module")
def n2f4():
    return analyze_decomp(
        "N2F4 N-N",
        "N 0 0.706 0; N 0 -0.706 0; "
        "F 1.059 1.227 0; F -1.059 1.227 0; "
        "F 1.059 -1.227 0; F -1.059 -1.227 0",
        [0, 0.706, 0], [0, -0.706, 0]
    )


@pytest.fixture(scope="module")
def bh3():
    return analyze_decomp(
        "BH3 B-H",
        "B 0 0 0; H 1.193 0 0; H -0.596 1.032 0; H -0.596 -1.032 0",
        [0, 0, 0], [1.193, 0, 0]
    )


# ============================================================
# Test 1: Mathematical reconstruction identity
# ============================================================

class TestReconstructionIdentity:
    """delta_pi + delta_sigma must equal delta_total = lambda_2 - lambda_1
    for any choice of MO classification."""

    def test_ethylene(self, ethylene):
        recon = ethylene['delta_pi_num'] + ethylene['delta_sigma_num']
        assert abs(recon - ethylene['delta_bader']) < 1e-3, (
            f"Reconstruction failed for ethylene: "
            f"sum={recon:.4f} vs delta_bader={ethylene['delta_bader']:.4f}"
        )

    def test_ethane(self, ethane):
        recon = ethane['delta_pi_num'] + ethane['delta_sigma_num']
        assert abs(recon - ethane['delta_bader']) < 1e-3

    def test_benzene(self, benzene):
        recon = benzene['delta_pi_num'] + benzene['delta_sigma_num']
        assert abs(recon - benzene['delta_bader']) < 1e-3

    def test_cyclopropane(self, cyclopropane):
        recon = cyclopropane['delta_pi_num'] + cyclopropane['delta_sigma_num']
        assert abs(recon - cyclopropane['delta_bader']) < 1e-3

    def test_n2f4(self, n2f4):
        recon = n2f4['delta_pi_num'] + n2f4['delta_sigma_num']
        assert abs(recon - n2f4['delta_bader']) < 1e-3


# ============================================================
# Test 2: Sign rule on paradigm cases (ε > 0 systems)
# ============================================================

class TestSignRule:
    """In all non-degenerate cases (ε > 0), expect delta_sigma > 0 and delta_pi < 0."""

    def test_ethylene_paradigm(self, ethylene):
        assert ethylene['eps'] > 0.4, f"Expected ε > 0.4 for ethylene, got {ethylene['eps']}"
        assert ethylene['delta_sigma_num'] > 0, (
            f"delta_sigma should be > 0 for ethylene, got {ethylene['delta_sigma_num']:.4f}"
        )
        assert ethylene['delta_pi_num'] < 0, (
            f"delta_pi should be < 0 for ethylene, got {ethylene['delta_pi_num']:.4f}"
        )

    def test_benzene_aromatic(self, benzene):
        assert benzene['eps'] > 0.15
        assert benzene['delta_sigma_num'] > 0
        assert benzene['delta_pi_num'] < 0

    def test_cyclopropane_walsh(self, cyclopropane):
        """Walsh orbital case: ε > 0 with no formal π bond.
        Expected: delta_sigma > 0 (Walsh in-plane π-like), delta_pi < 0
        (small negative from out-of-plane orbital tails)."""
        assert cyclopropane['eps'] > 0.5, "Expected large ε for cyclopropane"
        assert cyclopropane['delta_sigma_num'] > 0
        assert cyclopropane['delta_pi_num'] < 0

    def test_n2f4_lp_induced(self, n2f4):
        """N2F4: ε > 0 driven by 4 F substituents, no π bond.
        Expected: delta_sigma >> |delta_pi|."""
        assert n2f4['eps'] > 0.5, f"Expected ε > 0.5 for N2F4, got {n2f4['eps']}"
        assert n2f4['delta_sigma_num'] > 0
        assert n2f4['delta_pi_num'] < 0
        # Sigma should dominate
        assert n2f4['delta_sigma_num'] > abs(n2f4['delta_pi_num']) * 2


# ============================================================
# Test 3: Degenerate cases (ε ≈ 0)
# ============================================================

class TestDegenerate:
    """In cylindrically symmetric bonds (ε ≈ 0), delta_sigma and delta_pi should cancel."""

    def test_ethane_cylindrical(self, ethane):
        assert abs(ethane['eps']) < 0.02, f"Ethane should be ε ≈ 0, got {ethane['eps']}"
        cancellation = ethane['delta_pi_num'] + ethane['delta_sigma_num']
        assert abs(cancellation) < 0.05, (
            f"delta_pi + delta_sigma should be ~0 in degenerate case, got {cancellation:.4f}"
        )

    def test_acetylene_triple_bond(self, acetylene):
        assert abs(acetylene['eps']) < 0.02
        cancellation = acetylene['delta_pi_num'] + acetylene['delta_sigma_num']
        assert abs(cancellation) < 0.05


# ============================================================
# Test 4: σ-only system (BH3, no occupied π MOs)
# ============================================================

class TestSigmaOnly:
    """BH3 has only σ MOs, so delta_pi should be exactly zero."""

    def test_bh3_no_pi(self, bh3):
        assert bh3['eps'] > 0.2, f"Expected ε > 0.2 for BH3, got {bh3['eps']}"
        assert bh3['delta_sigma_num'] > 0
        assert abs(bh3['delta_pi_num']) < 0.005, (
            f"delta_pi should be 0 for BH3 (no π MOs), got {bh3['delta_pi_num']:.4f}"
        )


# ============================================================
# Test 5: classify_MO heuristic
# ============================================================

class TestClassifyMO:
    """Test the σ/π classification function with synthetic inputs."""

    def test_pure_pi(self):
        """ψ ≈ 0 at BCP, large gradient along λ₁ (out-of-plane) → π."""
        result = classify_MO(
            psi_at_bcp=0.001,    # ~0
            dpsi_pi=0.5,         # large
            dpsi_sigma_perp=0.05,  # small
            dpsi_bond=0.0,
            threshold_psi=0.05
        )
        assert result == 'pi'

    def test_pure_sigma(self):
        """Large ψ at BCP → σ."""
        result = classify_MO(
            psi_at_bcp=0.3,
            dpsi_pi=0.1,
            dpsi_sigma_perp=0.1,
            dpsi_bond=0.5,
            threshold_psi=0.05
        )
        assert result == 'sigma'

    def test_in_plane_with_small_psi(self):
        """ψ small but in-plane gradient larger → σ (not π)."""
        result = classify_MO(
            psi_at_bcp=0.01,
            dpsi_pi=0.1,
            dpsi_sigma_perp=0.5,  # in-plane is bigger
            dpsi_bond=0.0,
            threshold_psi=0.05
        )
        assert result == 'sigma'


# ============================================================
# Test 6: Result dict structure
# ============================================================

class TestResultStructure:
    """Verify the analyze_decomp return value has expected keys."""

    def test_keys(self, ethylene):
        required = {'name', 'eps', 'lam1', 'lam2', 'delta_bader',
                    'delta_pi_num', 'delta_sigma_num', 'delta_total_recon',
                    'classes', 'rho'}
        assert required.issubset(ethylene.keys()), (
            f"Missing keys: {required - set(ethylene.keys())}"
        )

    def test_values_are_floats(self, ethylene):
        for k in ['eps', 'lam1', 'lam2', 'delta_bader', 'delta_pi_num',
                  'delta_sigma_num', 'delta_total_recon', 'rho']:
            assert isinstance(ethylene[k], (int, float, np.floating)), (
                f"{k} should be numeric, got {type(ethylene[k])}"
            )

    def test_lambda_ordering(self, ethylene):
        """λ₁ ≤ λ₂ (both negative, λ₁ more negative)."""
        assert ethylene['lam1'] <= ethylene['lam2']
        assert ethylene['delta_bader'] >= 0  # by definition λ_2 - λ_1


# ============================================================
# Test 7: Numerical stability
# ============================================================

class TestNumericalStability:
    """Reconstruction error should be << 1% (mathematical identity)."""

    def test_reconstruction_error_below_1pct(self, ethylene, ethane, benzene,
                                              cyclopropane, n2f4):
        for r in [ethylene, ethane, benzene, cyclopropane, n2f4]:
            if abs(r['delta_bader']) > 1e-4:
                err = abs((r['delta_total_recon'] - r['delta_bader']) / r['delta_bader'])
                assert err < 0.01, (
                    f"Reconstruction error too large for {r['name']}: {err*100:.2f}%"
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
