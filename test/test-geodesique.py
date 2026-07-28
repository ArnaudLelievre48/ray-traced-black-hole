import numpy as np
import matplotlib.pyplot as plt


def circular_uphi(r, M):
    """u^phi=dphi/dtau pour une orbite circulaire massive de Schwarzschild.

    Omega=dphi/dt=sqrt(M/r^3), u^t=1/sqrt(1-3M/r), donc
    u^phi=Omega*u^t.

    Stable seulement pour r > 6M ; entre 3M et 6M, circulaire mais instable.
    """
    if r <= 3.0 * M:
        raise ValueError("No timelike circular orbit for r <= 3M")
    omega = np.sqrt(M / r**3)
    ut = 1.0 / np.sqrt(1.0 - 3.0 * M / r)
    return omega * ut


def init_massive_particle(r, phi, uphi, M, ur=0.0):
    """Initialise une particule massive dans le plan orbital.

    X = [t, r, phi, u^t, u^r, u^phi]

    Normalisation massive :
        -f (u^t)^2 + f^-1 (u^r)^2 + r^2 (u^phi)^2 = -1

    Donc :
        u^t = sqrt((1 + (u^r)^2/f + r^2 (u^phi)^2) / f)
    """
    f = 1.0 - 2.0 * M / r
    ut = np.sqrt((1.0 + ur**2 / f + r**2 * uphi**2) / f)
    return np.array([0.0, r, phi, ut, ur, uphi], dtype=np.float64)


def timelike_norm(X, M):
    """Doit rester proche de -1 pour une particule massive."""
    r = X[1]
    f = 1.0 - 2.0 * M / r
    ut, ur, uphi = X[3], X[4], X[5]
    return -f * ut**2 + ur**2 / f + r**2 * uphi**2


def geodesic_dX(X, M):
    """Membre de droite de la géodésique plane.

    X = [t, r, phi, ut, ur, uphi]
    dX = [dt, dr, dphi, dut, dur, duphi]
    """
    r = X[1]
    ut = X[3]
    ur = X[4]
    uphi = X[5]

    dX = np.zeros_like(X)

    if (not np.isfinite(X).all()) or r <= 2.05 * M:
        return dX

    f = 1.0 - 2.0 * M / r

    # dx^alpha/dlambda = u^alpha
    dX[0] = ut
    dX[1] = ur
    dX[2] = uphi

    # du^alpha/dlambda = -Gamma^alpha_{mu nu} u^mu u^nu
    dX[3] = -2.0 * M / (r**2 * f) * ut * ur
    dX[4] = (
        -M * f / r**2 * ut**2
        + M / (r**2 * f) * ur**2
        + r * f * uphi**2
    )
    dX[5] = -2.0 / r * ur * uphi

    return dX


def rk4_step(X, M, h=0.1):
    """Avance X d'un pas RK4."""
    k1 = geodesic_dX(X, M)
    k2 = geodesic_dX(X + 0.5 * h * k1, M)
    k3 = geodesic_dX(X + 0.5 * h * k2, M)
    k4 = geodesic_dX(X + h * k3, M)

    return X + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def position_radiale2cartesian(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.array([x, y])


def black_hole_schwarzchild_radius(M):
    r = 2.0 * M
    phi = np.linspace(0, 2*np.pi, 500)
    return np.stack([r*np.cos(phi), r*np.sin(phi)], axis=1)


r = 50.0
phi = -np.pi / 2
M = 2.0

# Pour une particule massive en orbite quasi-circulaire, mieux vaut partir de
# l'uphi circulaire GR plutôt que d'en choisir un au hasard.
uphi = circular_uphi(r, M)

STEPS = 50_000
PARTICLES = 10
H = 0.2

POSITIONS = np.full((PARTICLES, STEPS, 2), np.nan, dtype=np.float64)

horizon = black_hole_schwarzchild_radius(M)
plt.plot(horizon[:, 0], horizon[:, 1], color="black")

for i in range(POSITIONS.shape[0]):
    # Petit bruit autour de l'orbite circulaire.
    uphi_rand = uphi * (1.0 + 0.05 * (np.random.rand() - 0.5))
    X = init_massive_particle(r, phi, uphi_rand, M)
    print("initial norm", i, timelike_norm(X, M))

    POSITIONS[i, 0] = position_radiale2cartesian(X[1], X[2])

    last_j = 0
    for j in range(1, POSITIONS.shape[1]):
        X = rk4_step(X, M, h=H)
        if X[1] <= 2.05*M or not np.isfinite(X).all():
            break
        POSITIONS[i, j] = position_radiale2cartesian(X[1], X[2])
        last_j = j

    plt.plot(POSITIONS[i, :last_j+1, 0], POSITIONS[i, :last_j+1, 1])

plt.gca().set_aspect("equal", adjustable="box")
plt.show()
