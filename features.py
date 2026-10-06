"""Deterministic derived features (no labels, no torch) -- safe for worker processes."""
import numpy as np


# --------------------------------------------------------------------------
# Physics features from CST reconstruction
# Assumed convention: coeffs 1-4 lower surface, 5-8 upper surface, Bernstein
# order 3, class function N1=0.5, N2=1.0, zero trailing-edge thickness.
# --------------------------------------------------------------------------
_TH = np.linspace(0, np.pi, 161)
_XC = 0.5 * (1 - np.cos(_TH))           # cosine-spaced chord stations
_BERN = np.stack([np.array([1, 3, 3, 1])[k] * _XC**k * (1 - _XC)**(3 - k) for k in range(4)], 1)
_CLASS = np.sqrt(_XC) * (1 - _XC)


def physics_features(X):
    aoa, Al, Au = X[:, 0], X[:, 1:5], X[:, 5:9]
    yl = _CLASS * (Al @ _BERN.T)
    yu = _CLASS * (Au @ _BERN.T)
    t, z = yu - yl, 0.5 * (yu + yl)
    dz = np.gradient(z, _XC, axis=1)
    # thin-airfoil zero-lift angle: a0 = -(1/pi) * int dz/dx (cos th - 1) dth
    integrand = dz * (np.cos(_TH) - 1)
    a0 = -np.trapezoid(integrand, _TH, axis=1) / np.pi
    a0_deg = np.degrees(a0)
    # thin-airfoil pitching-moment-ish term (A1 - A2) proxy
    A1 = 2 / np.pi * np.trapezoid(dz * np.cos(_TH), _TH, axis=1)
    a_eff = np.radians(aoa) - a0
    feats = np.column_stack([
        t.max(1), _XC[t.argmax(1)], z.max(1), z.min(1), _XC[np.abs(z).argmax(1)],
        0.5 * (Au[:, 0] ** 2 + Al[:, 0] ** 2),          # LE radius ~ A0^2/2
        np.arctan(Au[:, 3]) - np.arctan(Al[:, 3]),      # TE wedge-angle proxy
        dz[:, -2],                                      # TE camber slope
        a0_deg, A1, np.degrees(a_eff), 2 * np.pi * a_eff,  # thin-airfoil Cl
        t.mean(1),
    ])
    return np.column_stack([X, feats])
