"""Independent shooting validation and reproducible solver-cost measurements.

The shooting benchmark is deliberately separate from the Jacobi basis. It
solves the real first-order Dirac system from both poles and matches at the
equator. The cost benchmark times basis construction, Galerkin assembly, and
dense Hermitian diagonalization independently.
"""
from __future__ import annotations

import csv
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import brentq


def _mass(theta: float | np.ndarray) -> float | np.ndarray:
    return np.cos(theta)


def _mass_derivative(theta: float) -> float:
    return -float(np.sin(theta))


def _north_data(epsilon: float, nu: float, lam: float, delta: float) -> np.ndarray:
    """Regular Frobenius data at theta=delta, through the first correction."""
    m0, m1 = float(_mass(0.0)), _mass_derivative(0.0)
    c1 = -(m0 - lam) / (epsilon * (2.0 * nu + 1.0))
    c2 = -m1 / (epsilon * (2.0 * nu + 2.0))
    u2 = 0.5 * ((m0*m0 - lam*lam) /
                (epsilon*epsilon*(2.0*nu + 1.0)) + nu/6.0)
    u = delta**nu * (1.0 + u2*delta**2)
    w = c1*delta**(nu + 1.0) + c2*delta**(nu + 2.0)
    return np.array([u, w], dtype=float)


def _south_data(epsilon: float, nu: float, lam: float, delta: float) -> np.ndarray:
    """Regular Frobenius data at theta=pi-delta, through the first correction."""
    mp, mp1 = float(_mass(np.pi)), _mass_derivative(np.pi)
    c1 = (mp + lam) / (epsilon * (2.0 * nu + 1.0))
    c2 = -mp1 / (epsilon * (2.0 * nu + 2.0))
    w2 = 0.5 * ((mp*mp - lam*lam) /
                (epsilon*epsilon*(2.0*nu + 1.0)) + nu/6.0)
    u = c1*delta**(nu + 1.0) + c2*delta**(nu + 2.0)
    w = delta**nu * (1.0 + w2*delta**2)
    return np.array([u, w], dtype=float)


def shooting_mismatch(lam: float, epsilon: float, nu: float = 0.5,
                      delta: float = 1.0e-5) -> float:
    """Normalized two-sided matching determinant for the single-wall profile.

    With the lower spinor component written as i*w, the eigenvalue equation is
        u' = nu*csc(theta)*u - (m+lambda)*w/epsilon,
        w' = -nu*csc(theta)*w - (m-lambda)*u/epsilon.
    """
    def rhs(theta: float, state: np.ndarray) -> tuple[float, float]:
        u, w = state
        m = float(_mass(theta))
        sine = float(np.sin(theta))
        return (nu*u/sine - (m + lam)*w/epsilon,
                -nu*w/sine - (m - lam)*u/epsilon)

    options = dict(method="DOP853", rtol=2.0e-12, atol=2.0e-14)
    midpoint = np.pi/2.0
    north = solve_ivp(rhs, (delta, midpoint),
                      _north_data(epsilon, nu, lam, delta), **options)
    south = solve_ivp(rhs, (np.pi-delta, midpoint),
                      _south_data(epsilon, nu, lam, delta), **options)
    if not north.success or not south.success:
        raise RuntimeError("two-sided shooting integration failed")
    left = north.y[:, -1]
    right = south.y[:, -1]
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(left[0]*right[1] - left[1]*right[0])


def shooting_eigenvalue(epsilon: float, nu: float = 0.5,
                        delta: float = 1.0e-5) -> float:
    """Locate the single-wall chiral eigenvalue by matching at theta=pi/2."""
    asymptotic_center = -nu*epsilon - nu*epsilon**2/4.0
    half_width = 0.02*epsilon
    lo, hi = asymptotic_center-half_width, asymptotic_center+half_width
    f_lo = shooting_mismatch(lo, epsilon, nu, delta)
    f_hi = shooting_mismatch(hi, epsilon, nu, delta)
    if f_lo*f_hi >= 0.0:
        raise RuntimeError(
            f"no shooting sign change in bracket [{lo:.8g}, {hi:.8g}] "
            f"for epsilon={epsilon:g}"
        )
    return float(brentq(lambda x: shooting_mismatch(x, epsilon, nu, delta),
                         lo, hi, xtol=5.0e-15, rtol=1.0e-14, maxiter=100))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_independent_benchmark(out: Path, basis_size: int = 112,
                              quadrature_order: int = 1000) -> dict:
    """Compare shooting and Galerkin eigenvalues and test pole-cutoff stability."""
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from sphere_dirac_galerkin import PoleRegularBasis, cosine_single

    nu = 0.5
    epsilons = (0.10, 0.05, 0.02, 0.01)
    basis = PoleRegularBasis.build(nu, basis_size, quadrature_order)
    rows = []
    for epsilon in epsilons:
        values = basis.solve(epsilon, cosine_single)
        target = -nu*epsilon - nu*epsilon**2/4.0
        galerkin = float(values[np.argmin(np.abs(values-target))])
        shooting = shooting_eigenvalue(epsilon, nu, 1.0e-5)
        coarse_cutoff = shooting_eigenvalue(epsilon, nu, 1.0e-4)
        rows.append({
            "epsilon": f"{epsilon:.8g}",
            "nu": f"{nu:.8g}",
            "profile": "cos(theta)",
            "galerkin_eigenvalue": f"{galerkin:.14g}",
            "shooting_eigenvalue": f"{shooting:.14g}",
            "absolute_difference": f"{abs(galerkin-shooting):.8e}",
            "relative_difference": f"{abs(galerkin-shooting)/abs(shooting):.8e}",
            "shooting_cutoff_delta_theta": "1e-5",
            "cutoff_shift_1e-4_vs_1e-5": f"{abs(coarse_cutoff-shooting):.8e}",
            "basis_size_per_component": basis_size,
            "quadrature_order": quadrature_order,
            "shooting_integrator": "DOP853",
            "shooting_rtol": "2e-12",
            "shooting_atol": "2e-14",
        })
    write_rows(out / "data" / "independent_shooting_benchmark.csv", rows)
    return {
        "max_abs_galerkin_shooting_difference": max(
            float(row["absolute_difference"]) for row in rows),
        "max_abs_shooting_cutoff_shift": max(
            float(row["cutoff_shift_1e-4_vs_1e-5"]) for row in rows),
        "epsilon_values": list(epsilons),
        "nu": nu,
        "basis_size_per_component": basis_size,
        "quadrature_order": quadrature_order,
    }


def run_cost_benchmark(out: Path, repeats: int = 7) -> dict:
    """Measure basis, assembly, and dense eigensolve time on this host."""
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from sphere_dirac_galerkin import PoleRegularBasis, cosine_single
    from threadpoolctl import threadpool_info, threadpool_limits

    cpu = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu = platform.processor() or "unknown"
    rows = []
    metadata = threadpool_info()
    blas_name = metadata[0]["internal_api"] if metadata else "unreported"
    blas_threads = metadata[0]["num_threads"] if metadata else "unreported"
    with threadpool_limits(limits=1, user_api="blas"):
        for size in (28, 56, 112, 140):
            basis_times, assembly_times, eig_times, total_times = [], [], [], []
            matrix = None
            for _ in range(repeats):
                start = time.perf_counter()
                basis = PoleRegularBasis.build(0.5, size, 1000)
                built = time.perf_counter()
                matrix = basis.hamiltonian(0.02, cosine_single)
                assembled = time.perf_counter()
                eigh(matrix, check_finite=False, driver="evd")
                solved = time.perf_counter()
                basis_times.append(built-start)
                assembly_times.append(assembled-built)
                eig_times.append(solved-assembled)
                total_times.append(solved-start)
            rows.append({
                "basis_size_per_component": size,
                "matrix_dimension": 2*size,
                "quadrature_order": 1000,
                "basis_build_median_s": f"{statistics.median(basis_times):.8g}",
                "matrix_assembly_median_s": f"{statistics.median(assembly_times):.8g}",
                "dense_eigensolve_median_s": f"{statistics.median(eig_times):.8g}",
                "total_median_s": f"{statistics.median(total_times):.8g}",
                "hamiltonian_matrix_MiB": f"{matrix.nbytes/(1024**2):.8g}",
                "repeats": repeats,
                "cpu_model": cpu,
                "visible_cpu_count": str(os.cpu_count() or 1),
                "blas": blas_name,
                "blas_threads_timed": "1",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            })
    write_rows(out / "data" / "solver_cost_benchmark.csv", rows)
    return {
        "cpu_model": cpu,
        "visible_cpu_count": os.cpu_count() or 1,
        "blas": blas_name,
        "blas_threads_timed": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "repeats_per_basis_size": repeats,
        "basis_sizes_per_component": [28, 56, 112, 140],
        "quadrature_order": 1000,
        "complex_dense_matrix_memory_scaling": "16*(2N)^2 bytes = O(N^2)",
        "dense_hermitian_eigensolve_scaling": "O((2N)^3) = O(N^3)",
    }
