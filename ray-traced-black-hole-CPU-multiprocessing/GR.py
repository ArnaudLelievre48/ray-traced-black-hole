import numpy as np
import matplotlib.pyplot as plt


# Convention d'indices :
# 0 = t
# 1 = r
# 2 = theta
# 3 = phi


def schwarzschild_metric(r, theta, M):
    g = np.zeros((4, 4), dtype=np.float64)

    f = 1.0 - 2.0 * M / r

    g[0, 0] = -f
    g[1, 1] = 1.0 / f
    g[2, 2] = r**2
    g[3, 3] = r**2 * np.sin(theta)**2

    return g


def schwarzschild_christoffel(r, theta, M):
    Gamma = np.zeros((4, 4, 4), dtype=np.float64)

    f = 1.0 - 2.0 * M / r
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Gamma^t_{tr} = Gamma^t_{rt}
    Gamma[0, 0, 1] = M / (r**2 * f)
    Gamma[0, 1, 0] = M / (r**2 * f)

    # Gamma^r_{..}
    Gamma[1, 0, 0] = M * f / r**2
    Gamma[1, 1, 1] = -M / (r**2 * f)
    Gamma[1, 2, 2] = -r * f
    Gamma[1, 3, 3] = -r * f * sin_theta**2

    # Gamma^theta_{..}
    Gamma[2, 1, 2] = 1.0 / r
    Gamma[2, 2, 1] = 1.0 / r
    Gamma[2, 3, 3] = -sin_theta * cos_theta

    # Gamma^phi_{..}
    Gamma[3, 1, 3] = 1.0 / r
    Gamma[3, 3, 1] = 1.0 / r

    if abs(sin_theta) > 1e-12:
        Gamma[3, 2, 3] = cos_theta / sin_theta
        Gamma[3, 3, 2] = cos_theta / sin_theta

    return Gamma


# u^mu = d x^mu / d lambda
# state = [t, r, theta, phi, ut, ur, utheta, uphi]
def geodesic_rhs(state, M):
    """
    Version explicite des équations géodésiques Schwarzschild.

    Même équation que la version Christoffel générale, mais sans allouer Gamma
    et sans triples boucles Python à chaque sous-step RK4.
    state = [t, r, theta, phi, ut, ur, utheta, uphi]
    """
    t, r, theta, phi, ut, ur, utheta, uphi = state

    f = 1.0 - 2.0 * M / r
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    dstatedlambda = np.empty(8, dtype=np.float64)

    # dx^mu/dlambda = u^mu
    dstatedlambda[0] = ut
    dstatedlambda[1] = ur
    dstatedlambda[2] = utheta
    dstatedlambda[3] = uphi

    # du^mu/dlambda = -Gamma^mu_{alpha beta} u^alpha u^beta
    dstatedlambda[4] = -2.0 * M / (r**2 * f) * ut * ur

    dstatedlambda[5] = (
        - M * f / r**2 * ut**2
        + M / (r**2 * f) * ur**2
        + r * f * utheta**2
        + r * f * sin_theta**2 * uphi**2
    )

    dstatedlambda[6] = (
        -2.0 / r * ur * utheta
        + sin_theta * cos_theta * uphi**2
    )

    if abs(sin_theta) > 1e-12:
        cot_theta = cos_theta / sin_theta
    else:
        cot_theta = 0.0

    dstatedlambda[7] = (
        -2.0 / r * ur * uphi
        -2.0 * cot_theta * utheta * uphi
    )

    return dstatedlambda

def photon_impact_parameter(state, M):
    """
    Paramètre d'impact b = L/E pour un photon Schwarzschild.

    Pour une géodésique 3D :
        E = f ut
        L² = r⁴ (utheta² + sin²(theta) uphi²)
    """
    r = state[1]
    theta = state[2]
    ut = state[4]
    utheta = state[6]
    uphi = state[7]

    f = 1.0 - 2.0 * M / r
    E = f * ut
    L2 = r**4 * (utheta**2 + np.sin(theta)**2 * uphi**2)

    if E == 0 or not np.isfinite(E) or L2 < 0:
        return np.inf

    return np.sqrt(L2) / abs(E)


# ds^2 = 0 <=> g_mu_nu u^mu u^nu = 0
def null_condition(state, M):
    """Retourne g_mu_nu u^mu u^nu. Pour un photon, ça doit rester proche de 0."""
    t, r, theta, phi = state[:4]
    u = state[4:]
    g = schwarzschild_metric(r, theta, M)
    return u @ g @ u


# methode de rungue-kutta 4 pour forward integration avec un maximum de stabilité
def rk4_step(state, h, M):
    k1 = geodesic_rhs(state, M)
    k2 = geodesic_rhs(state + 0.5 * h * k1, M)
    k3 = geodesic_rhs(state + 0.5 * h * k2, M)
    k4 = geodesic_rhs(state + h * k3, M)
    return state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)



# --------------------
# espace de test
# --------------------


def initial_photon_equatorial(r0, phi0, b, M=1.0, inward=True):
    """
    Condition initiale photon dans le plan équatorial theta=pi/2.

    On utilise les constantes de mouvement :
        E = f ut
        L = r² uphi
        b = L/E

    On fixe E=1, donc L=b.
    La condition photon ds²=0 donne ur.
    """
    theta0 = np.pi / 2
    f = 1.0 - 2.0 * M / r0

    E = 1.0
    L = b

    ut = E / f
    uphi = L / r0**2

    ur_squared = E**2 - f * L**2 / r0**2
    if ur_squared < 0:
        raise ValueError(f"Impossible: ur²={ur_squared} < 0 for r0={r0}, b={b}")

    ur = np.sqrt(ur_squared)
    if inward:
        ur *= -1.0

    return np.array([
        0.0,      # t
        r0,       # r
        theta0,   # theta
        phi0,     # phi
        ut,
        ur,
        0.0,      # utheta
        uphi,
    ], dtype=np.float64)


def integrate_photon_equatorial(r0, b, M=1.0, h=0.05, max_steps=20_000, r_escape=60.0):
    """Intègre un photon équatorial lancé depuis r0 vers le trou noir."""
    state = initial_photon_equatorial(r0=r0, phi0=0.0, b=b, M=M, inward=True)
    states = [state.copy()]

    escaped_after_turning = False
    has_turned = False
    previous_ur = state[5]

    for _ in range(max_steps):
        r = state[1]

        if r <= 2.05 * M:
            break

        if previous_ur < 0 and state[5] > 0:
            has_turned = True

        if has_turned and r > r_escape:
            escaped_after_turning = True
            break

        previous_ur = state[5]
        state = rk4_step(state, h, M)
        states.append(state.copy())

        if not np.all(np.isfinite(state)):
            break

    return np.array(states), escaped_after_turning


def spherical_equatorial_to_cartesian(states):
    r = states[:, 1]
    phi = states[:, 3]
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def plot_photon_geodesics_2d(output_path="gr_photon_geodesics_debug.png"):
    M = 1.0
    r0 = 30.0
    b_crit = 3.0 * np.sqrt(3.0) * M

    b_values = [3.5, 4.5, b_crit * 0.995, b_crit * 1.005, 6.0, 7.0]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Horizon r=2M
    horizon = plt.Circle((0, 0), 2*M, color="black", alpha=0.9, label="horizon r=2M")
    ax.add_patch(horizon)

    # Sphère photonique r=3M, cercle pointillé dans le plan équatorial
    photon_sphere = plt.Circle((0, 0), 3*M, color="orange", fill=False, linestyle="--", alpha=0.8, label="photon sphere r=3M")
    ax.add_patch(photon_sphere)

    for b in b_values:
        states, escaped = integrate_photon_equatorial(r0=r0, b=b, M=M)
        x, y = spherical_equatorial_to_cartesian(states)
        label = f"b={b:.3f}" + (" escape" if escaped else " fall")
        ax.plot(x, y, label=label)

        nc0 = null_condition(states[0], M)
        nc_end = null_condition(states[-1], M)
        print(f"b={b:.6f}, steps={len(states)}, r_end={states[-1,1]:.3f}, null start={nc0:.3e}, null end={nc_end:.3e}")

    ax.set_aspect("equal")
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Photon geodesics in Schwarzschild, equatorial plane")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved {output_path}")


if __name__ == "__main__":
    plot_photon_geodesics_2d()

