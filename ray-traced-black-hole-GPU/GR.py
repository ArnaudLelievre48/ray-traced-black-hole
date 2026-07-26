import cupy as cp


def schwarzschild_christoffel(lightrays, M):
    """
    Symboles de Christoffel de Schwarzschild pour tous les rayons.

    lightrays[y, x] = [t, r, theta, phi, ut, ur, utheta, uphi]
    Gamma[y, x, alpha, mu, nu] = Γ^alpha_{mu nu}
    """
    Gamma = cp.zeros(lightrays.shape[:2] + (4, 4, 4), dtype=lightrays.dtype)

    r = lightrays[:, :, 1]
    theta = lightrays[:, :, 2]

    f = 1.0 - 2.0 * M / r
    sin_theta = cp.sin(theta)
    cos_theta = cp.cos(theta)

    # Γ^t_{tr} = Γ^t_{rt}
    Gamma[:, :, 0, 0, 1] = M / (r**2 * f)
    Gamma[:, :, 0, 1, 0] = M / (r**2 * f)

    # Γ^r_{tt}, Γ^r_{rr}, Γ^r_{θθ}, Γ^r_{φφ}
    Gamma[:, :, 1, 0, 0] = M * f / r**2
    Gamma[:, :, 1, 1, 1] = -M / (r**2 * f)
    Gamma[:, :, 1, 2, 2] = -r * f
    Gamma[:, :, 1, 3, 3] = -r * f * sin_theta**2

    # Γ^θ_{rθ} = Γ^θ_{θr}, Γ^θ_{φφ}
    Gamma[:, :, 2, 1, 2] = 1.0 / r
    Gamma[:, :, 2, 2, 1] = 1.0 / r
    Gamma[:, :, 2, 3, 3] = -sin_theta * cos_theta

    # Γ^φ_{rφ} = Γ^φ_{φr}
    Gamma[:, :, 3, 1, 3] = 1.0 / r
    Gamma[:, :, 3, 3, 1] = 1.0 / r

    # Γ^φ_{θφ} = Γ^φ_{φθ} = cot(theta)
    cot_theta = cp.where(
        cp.abs(sin_theta) > 1e-12,
        cos_theta / sin_theta,
        0.0,
    )
    Gamma[:, :, 3, 2, 3] = cot_theta
    Gamma[:, :, 3, 3, 2] = cot_theta

    return Gamma


def geodesic_rhs(lightrays, M):
    """
    d x^mu = u^mu dλ
    d u^mu = -Γ^mu_{αβ} u^α u^β dλ

    => d/dλ [x^mu, u^mu] = [u^mu, -Γ^mu_{αβ} u^α u^β]
    """
    dlightrays = cp.empty_like(lightrays)
    Gamma = schwarzschild_christoffel(lightrays, M)
    u = lightrays[:, :, 4:8]

    # dx^mu/dλ = u^mu
    dlightrays[:, :, 0:4] = u

    # du^mu/dλ = - Γ^mu_{αβ} u^α u^β
    dlightrays[:, :, 4:8] = -cp.einsum(
        "...mab,...a,...b->...m",
        Gamma,
        u,
        u,
    )

    return dlightrays


def rk4_step(lightrays, h, M):
    """Un pas RK4.

    h peut être :
        - un scalaire ;
        - une matrice (height, width), avec un pas différent par rayon.

    Dans le deuxième cas on l'étend en (height, width, 1) pour multiplier
    toutes les composantes [t,r,theta,phi,ut,ur,utheta,uphi] du même rayon.
    """
    if hasattr(h, "ndim") and h.ndim == 2:
        h = h[:, :, None]

    k1 = geodesic_rhs(lightrays, M)
    k2 = geodesic_rhs(lightrays + 0.5 * h * k1, M)
    k3 = geodesic_rhs(lightrays + 0.5 * h * k2, M)
    k4 = geodesic_rhs(lightrays + h * k3, M)
    return lightrays + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
