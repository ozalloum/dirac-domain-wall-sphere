"""Pole-regular Jacobi-Galerkin solver for an axially symmetric Dirac operator on S^2.

The reduced sector operator acts on L^2((0, pi), dtheta; C^2):
    H_{eps,nu} = -i eps/R sigma_1 d_theta
                    + eps nu/(R sin(theta)) sigma_2 + m(theta) sigma_3.

The spinor endpoint powers encode the smooth global domain at both poles.
Dependencies: NumPy and SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.special import eval_jacobi, roots_legendre

Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
MassProfile = Callable[[Array], Array]


def validate_sector(nu: float) -> float:
    """Validate and return a nonzero half-integer angular momentum."""
    twice = 2.0 * float(nu)
    if not np.isfinite(twice) or abs(twice - round(twice)) > 1e-12:
        raise ValueError("nu must be a half-integer")
    if abs(round(twice)) % 2 != 1:
        raise ValueError("nu must be a nonzero half-integer (±1/2, ±3/2, ...)")
    if nu == 0:
        raise ValueError("nu must be nonzero")
    return float(nu)


def endpoint_exponents(nu: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ((north,south) upper powers, (north,south) lower powers).

    These powers are for the amplitudes after removing the spherical measure.
    """
    nu = validate_sector(nu)
    kappa, sign = abs(nu), 1.0 if nu > 0 else -1.0
    upper = (kappa + (1.0 - sign) / 2.0,
             kappa + (1.0 + sign) / 2.0)
    lower = (kappa + (1.0 + sign) / 2.0,
             kappa + (1.0 - sign) / 2.0)
    return upper, lower


@dataclass(frozen=True)
class PoleRegularBasis:
    """Orthonormal Jacobi bases and their derivatives for one angular sector."""
    nu: float
    size: int
    theta: Array
    weights: Array
    upper: Array
    upper_derivative: Array
    upper_norms: Array
    lower: Array
    lower_derivative: Array
    lower_norms: Array

    @classmethod
    def build(cls, nu: float, size: int, quadrature_order: int | None = None):
        nu = validate_sector(nu)
        if size < 2:
            raise ValueError("size must be at least 2")
        if quadrature_order is None:
            quadrature_order = max(500, 8 * size)
        if quadrature_order < 2 * size + 20:
            raise ValueError("quadrature_order should exceed 2*size by a safe margin")
        x, w = roots_legendre(int(quadrature_order))
        theta = 0.5 * np.pi * (x + 1.0)
        weights = 0.5 * np.pi * w
        upper_exp, lower_exp = endpoint_exponents(nu)
        upper, dupper, unorms = _component_basis(*upper_exp, size, theta, weights)
        lower, dlower, vnorms = _component_basis(*lower_exp, size, theta, weights)
        return cls(nu, size, theta, weights, upper, dupper, unorms,
                   lower, dlower, vnorms)

    def hamiltonian(self, epsilon: float, mass: MassProfile, radius: float = 1.0) -> Array:
        """Assemble the Hermitian 2N x 2N Galerkin matrix."""
        if not (epsilon > 0.0 and radius > 0.0):
            raise ValueError("epsilon and radius must be positive")
        m = np.asarray(mass(self.theta), dtype=float)
        if m.shape != self.theta.shape or not np.all(np.isfinite(m)):
            raise ValueError("mass(theta) must return finite values with theta's shape")
        w = self.weights
        u, v = self.upper, self.lower
        uu = (u * w) @ (m[None, :] * u).T
        vv = (v * w) @ (m[None, :] * v).T
        derivative = (u * w) @ (self.lower_derivative
                               + (self.nu / np.sin(self.theta))[None, :] * v).T
        upper_right = -1j * epsilon / radius * derivative
        matrix = np.block([[uu.astype(complex), upper_right],
                           [upper_right.conj().T, -vv.astype(complex)]])
        return 0.5 * (matrix + matrix.conj().T)

    def solve(self, epsilon: float, mass: MassProfile, radius: float = 1.0,
              vectors: bool = False):
        """Compute the full real spectrum, optionally with Galerkin coefficients."""
        h = self.hamiltonian(epsilon, mass, radius)
        vals, vecs = eigh(h, check_finite=False, driver="evd")
        return (vals, vecs) if vectors else vals

    def reconstruct(self, coefficients: ComplexArray, theta: Array) -> tuple[ComplexArray, ComplexArray]:
        """Reconstruct the two reduced spinor components on an interior grid."""
        theta = np.asarray(theta, dtype=float)
        if np.any(theta <= 0.0) or np.any(theta >= np.pi):
            raise ValueError("reconstruction points must lie strictly between the poles")
        upper_exp, lower_exp = endpoint_exponents(self.nu)
        u, _, _ = _component_basis(*upper_exp, self.size, theta, self.weights,
                                   normalization=self.upper_norms)
        v, _, _ = _component_basis(*lower_exp, self.size, theta, self.weights,
                                   normalization=self.lower_norms)
        c = coefficients[:self.size]
        d = coefficients[self.size:]
        return c @ u, d @ v


def _component_basis(a: float, b: float, size: int, theta: Array,
                     quadrature_weights: Array,
                     normalization: Array | None = None) -> tuple[Array, Array, Array]:
    """Normalized sin(theta/2)^a cos(theta/2)^b Jacobi basis."""
    alpha, beta = a - 0.5, b - 0.5
    x, sint = np.cos(theta), np.sin(theta)
    factor = np.sin(theta / 2.0) ** a * np.cos(theta / 2.0) ** b
    polys = np.array([eval_jacobi(k, alpha, beta, x) for k in range(size)])
    dpolys = np.zeros_like(polys)
    for k in range(1, size):
        dpolys[k] = (-0.5 * sint * (k + alpha + beta + 1.0)
                     * eval_jacobi(k - 1, alpha + 1.0, beta + 1.0, x))
    raw = factor[None, :] * polys
    dfactor = 0.5 * factor * (a / np.tan(theta / 2.0) - b * np.tan(theta / 2.0))
    derivative = dfactor[None, :] * polys + factor[None, :] * dpolys
    norms = (np.sqrt(np.sum(quadrature_weights[None, :] * raw**2, axis=1))
             if normalization is None else normalization)
    return raw / norms[:, None], derivative / norms[:, None], norms


def cosine_single(theta: Array, mu: float = 1.0) -> Array:
    """One equatorial wall: m(theta)=mu*cos(theta)."""
    return mu * np.cos(theta)


def cosine_double(theta: Array, mu: float = 1.0) -> Array:
    """Reflection-symmetric pair of walls: m(theta)=mu*cos(2 theta)."""
    return mu * np.cos(2.0 * theta)
