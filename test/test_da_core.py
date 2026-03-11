"""
tests/test_da_core.py
Run with:  python tests/test_da_core.py   (from repo root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from da.core import tempering_schedule, aoei

def test_schedule_sums_to_one():
    for nt in [1,2,3,5,10]:
        for a in [0.0, 0.5, 2.0, 5.0]:
            s = tempering_schedule(nt, a)
            assert len(s) == nt
            assert abs(s.sum() - 1.0) < 1e-5, f"Nt={nt} as={a}: sum={s.sum()}"
    print("PASS  tempering_schedule: sums to 1")

def test_schedule_equal_weights_at_zero():
    for nt in [1, 3, 5, 10]:
        s = tempering_schedule(nt, 0.0)
        assert np.allclose(s, 1.0/nt, atol=1e-5), f"Not equal at as=0: {s}"
    print("PASS  tempering_schedule: equal weights when alpha_s=0")

def test_schedule_back_loaded():
    s = tempering_schedule(5, 2.0)
    assert np.all(np.diff(s) > 0), f"Not back-loaded: {s}"
    print(f"PASS  tempering_schedule: back-loaded  {np.round(s,4)}")

def test_schedule_ntemp1():
    s = tempering_schedule(1, 2.0)
    assert len(s) == 1 and abs(s[0] - 1.0) < 1e-6
    print("PASS  tempering_schedule: Ntemp=1 -> [1.0]")

def test_aoei_floor():
    rng = np.random.default_rng(0)
    for _ in range(100):
        nobs = rng.integers(1, 10)
        Ne   = 15
        R0   = np.abs(rng.standard_normal(nobs)) * 10 + 1.0
        yo   = rng.standard_normal(nobs) * 5
        hxf  = rng.standard_normal((nobs, Ne)) * 3
        r    = aoei(yo, hxf, R0)
        assert np.all(r >= R0 - 1e-6), f"Floor violated: {r} < {R0}"
    print("PASS  aoei: floor guarantee (R_tilde >= R0) for 100 random cases")

def test_aoei_no_inflation_small_dep():
    rng = np.random.default_rng(1)
    Ne = 20; R0 = np.array([25.0])
    yo = np.array([10.0])
    hxf = 9.5 + rng.standard_normal((1, Ne)) * 1.5   # spread~1.5, d~0.5
    r = aoei(yo, hxf, R0)
    assert np.allclose(r, R0, atol=0.01), f"Expected no inflation: {r}"
    print(f"PASS  aoei: no inflation for small departure  R_tilde={r[0]:.3f}")

def test_aoei_inflates_large_dep():
    rng = np.random.default_rng(2)
    Ne = 20; R0 = np.array([25.0])
    yo = np.array([50.0])
    hxf = 10.0 + rng.standard_normal((1, Ne)) * 1.0   # d~40
    r = aoei(yo, hxf, R0)
    assert r[0] > R0[0], f"Expected inflation: {r[0]} <= {R0[0]}"
    print(f"PASS  aoei: inflates large departure  R_tilde={r[0]:.1f}  R0={R0[0]:.1f}")


# ── ATEnKF helpers ─────────────────────────────────────────────────────────

def test_solve_ntemp_no_inflation():
    from da.core import _solve_ntemp
    # No inflation -> Ntemp = 1
    assert _solve_ntemp(1.0, alpha_s=1.0) == 1
    assert _solve_ntemp(0.5, alpha_s=1.0) == 1   # ratio < 1 also gives 1
    print("PASS  _solve_ntemp: ratio<=1 -> Ntemp=1")

def test_solve_ntemp_increases_with_ratio():
    from da.core import _solve_ntemp
    # Larger inflation ratio should require more steps
    nts = [_solve_ntemp(r, alpha_s=1.0) for r in [2.0, 5.0, 10.0, 50.0, 100.0]]
    assert nts == sorted(nts), f"Ntemp not monotone with ratio: {nts}"
    print(f"PASS  _solve_ntemp: monotone with inflation ratio  {nts}")

def test_solve_ntemp_respects_cap():
    from da.core import _solve_ntemp
    # Result must never exceed ntemp_max, even for absurd ratios
    for cap in [5, 10, 20]:
        nt = _solve_ntemp(1e30, alpha_s=1.0, ntemp_max=cap)
        assert nt <= cap, f"Ntemp={nt} exceeded cap={cap}"
    print("PASS  _solve_ntemp: respects ntemp_max cap")

def test_atenkf_first_step_uses_R_tilde():
    """
    For an obs where AOEI fires, the first step's oerr should equal R_tilde
    (up to floating point), i.e. R0 / alpha_1(Ntemp_j) = R_tilde_j.
    """
    from da.core import _solve_ntemp, tempering_schedule
    R0 = 25.0
    R_tilde = 200.0   # strong inflation
    ratio = R_tilde / R0
    nt = _solve_ntemp(ratio, alpha_s=1.0)
    w  = tempering_schedule(nt, alpha_s=1.0)
    # First step error: R0 / alpha_1
    oerr_first = R0 / w[0]
    # Should be >= R_tilde (alpha_1 <= R0/R_tilde)
    assert oerr_first >= R_tilde - 1e-3, \
        f"First step oerr={oerr_first:.2f} < R_tilde={R_tilde:.2f}"
    # And the next Ntemp should require one more step if ratio is higher
    print(f"PASS  ATEnKF first step: oerr_first={oerr_first:.2f} >= R_tilde={R_tilde:.2f}  Nt={nt}")

def test_atenkf_information_preserving():
    """
    sum(alpha_i) = 1  =>  sum(alpha_i / R0) = 1/R0  for any Ntemp.
    This confirms the information-preserving property.
    """
    from da.core import tempering_schedule
    R0 = 25.0
    for nt in [1, 2, 3, 5, 10, 20]:
        w = tempering_schedule(nt, alpha_s=1.0)
        total_info = (w / R0).sum()
        expected   = 1.0 / R0
        assert abs(total_info - expected) < 1e-5, \
            f"Nt={nt}: total_info={total_info:.6f} != 1/R0={expected:.6f}"
    print("PASS  ATEnKF: information-preserving property (sum alpha_i/R0 = 1/R0)")

if __name__ == "__main__":
    print("Running tests\n" + "-"*40)
    test_schedule_sums_to_one()
    test_schedule_equal_weights_at_zero()
    test_schedule_back_loaded()
    test_schedule_ntemp1()
    test_aoei_floor()
    test_aoei_no_inflation_small_dep()
    test_aoei_inflates_large_dep()
    test_solve_ntemp_no_inflation()
    test_solve_ntemp_increases_with_ratio()
    test_solve_ntemp_respects_cap()
    test_atenkf_first_step_uses_R_tilde()
    test_atenkf_information_preserving()
    print("-"*40)
    print("All tests passed.")