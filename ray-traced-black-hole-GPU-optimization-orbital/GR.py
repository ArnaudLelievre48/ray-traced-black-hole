import cupy as cp


def schwarzschild_christoffel(M, r):
    """Christoffel de Schwarzschild dans le plan orbital.

    Coordonnées : x^alpha = [t, r, beta]
    Gamma[ray_id, alpha, mu, nu] = Γ^alpha_{mu nu}

    r peut être un scalaire ou un tableau (RAYS_NUMBER,).
    """
    r = cp.asarray(r, dtype=cp.float64)
    f = 1.0 - 2.0 * M / r

    Gamma = cp.zeros(r.shape + (3, 3, 3), dtype=cp.float64)

    # Γ^t_{tr} = Γ^t_{rt}
    Gamma[..., 0, 0, 1] = M / (r**2 * f)
    Gamma[..., 0, 1, 0] = M / (r**2 * f)

    # Γ^r_{tt}, Γ^r_{rr}, Γ^r_{beta beta}
    Gamma[..., 1, 0, 0] = M * f / r**2
    Gamma[..., 1, 1, 1] = -M / (r**2 * f)
    Gamma[..., 1, 2, 2] = -r * f

    # Γ^beta_{r beta} = Γ^beta_{beta r}
    Gamma[..., 2, 1, 2] = 1.0 / r
    Gamma[..., 2, 2, 1] = 1.0 / r

    return Gamma


# calcule dx^alpha / d_lambda = u^alpha,
# puis du^alpha / d_lambda = -Γ^alpha_{mu nu} u^mu u^nu
def geodesic_dX(X, M):
    """Membre de droite de la géodésique plane.

    X[ray_id] = [t, r, beta, ut, ur, ubeta]
    dX[ray_id] = [dt, dr, dbeta, dut, dur, dubeta]
    """
    r = X[:, 1]
    ut = X[:, 3]
    ur = X[:, 4]
    ubeta = X[:, 5]

    # Schwarzschild est singulier en r=2M.
    # Si un rayon est déjà capturé / non fini, on met dX=0 pour éviter de
    # calculer des termes explosifs. orbital_geodesic le désactivera ensuite.
    capture_radius = 2.05 * M
    valid = cp.isfinite(X).all(axis=1) & (r > capture_radius)

    dX = cp.zeros_like(X)

    rv = r[valid]
    utv = ut[valid]
    urv = ur[valid]
    ubetav = ubeta[valid]
    f = 1.0 - 2.0 * M / rv

    # dx^alpha/dlambda = u^alpha
    dX[valid, 0] = utv
    dX[valid, 1] = urv
    dX[valid, 2] = ubetav

    # du^alpha/dlambda = -Gamma^alpha_{mu nu} u^mu u^nu
    # Version explicite du même calcul que Gamma + einsum, mais sans allouer
    # Gamma[ray_id,3,3,3] à chaque sous-pas RK4.
    dX[valid, 3] = -2.0 * M / (rv**2 * f) * utv * urv
    dX[valid, 4] = (
        -M * f / rv**2 * utv**2
        + M / (rv**2 * f) * urv**2
        + rv * f * ubetav**2
    )
    dX[valid, 5] = -2.0 / rv * urv * ubetav

    return dX


def rk4_step(X, h, M):
    """Avance X d'un pas RK4.

    h peut être :
        - scalaire ;
        - tableau (RAYS_NUMBER,) ;
        - tableau (RAYS_NUMBER, 1).

    En interne on veut h.shape == (RAYS_NUMBER, 1), pour que chaque rayon ait
    son propre pas mais que ce pas multiplie les 6 composantes de son état.
    """
    h = cp.asarray(h, dtype=X.dtype)
    if h.ndim == 0:
        pass
    elif h.ndim == 1:
        h = h[:, None]
    elif h.ndim == 2 and h.shape[1] == 1:
        pass
    else:
        raise ValueError("h doit être un scalaire, (RAYS_NUMBER,), ou (RAYS_NUMBER, 1)")

    k1 = geodesic_dX(X, M)
    k2 = geodesic_dX(X + 0.5 * h * k1, M)
    k3 = geodesic_dX(X + 0.5 * h * k2, M)
    k4 = geodesic_dX(X + h * k3, M)

    return X + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# beta = 0 => vers le trou noir
# beta = pi => à l'opposé du trou noir
def orbital_geodesic(r, M, RAYS_NUMBER=1000, MAX_STEPS=10_000, r_escape=None, h_step=1):
    # le photon respecte ds^2 = -f(r) dt^2 + f(r)^-1 dr^2 + r^2 dbeta^2
    f = 1.0 - 2.0 * M / r
    beta = cp.linspace(0, cp.pi, RAYS_NUMBER, dtype=cp.float64)

    ur = -cp.cos(beta)
    ubeta = cp.sin(beta) / r
    ut = cp.sqrt((ur**2 / f + r**2 * ubeta**2) / f)

    # X[ray_id] = [t, r, beta, ut, ur, ubeta]
    X = cp.stack([
        cp.zeros_like(beta),
        cp.full_like(beta, r),
        beta,
        ut,
        ur,
        ubeta,
    ], axis=1).astype(cp.float64)

    # Matrice-colonne : un pas par rayon, broadcastable avec X.shape == (N, 6).
    # Plus tard tu pourras remplacer chaque h[ray_id, 0] indépendamment.
    h = h_step * (1.0 - 0.95 * cp.sin(0.5 * beta))[:, None]

    if r_escape is None:
        r_escape = r

    active = cp.ones(X.shape[0], dtype=cp.bool_)
    capture_radius = 2.05 * M

    for step in range(MAX_STEPS):
        if not bool(active.any().get()):
            break

        # On n'intègre que les rayons actifs. Sinon les rayons déjà échappés
        # continuent vers r énorme pendant MAX_STEPS et peuvent overflow.
        X[active] = rk4_step(X[active], h[active], M)

        r_current = X[:, 1]
        ur_current = X[:, 4]

        invalid = ~cp.isfinite(X).all(axis=1)
        captured = r_current <= capture_radius
        escaped = (r_current >= r_escape) & (ur_current > 0.0)

        # Nettoyage des états capturés : le dernier pas RK4 peut passer sous
        # l'horizon en coordonnées de Schwarzschild, donc on fige proprement.
        X[captured, 1] = 2.0 * M
        X[captured, 3:6] = 0.0

        active = ~(invalid | captured | escaped)

    return X






def orbital_geodesic_fast(r, M, RAYS_NUMBER=1000, MAX_STEPS=10_000, r_escape=None, h_step=0.5, check_interval=25):
    """Version performance de orbital_geodesic.

    Même état :
        X[ray_id] = [t, r, beta, ut, ur, ubeta]

    Différence avec orbital_geodesic :
        - garde tous les rayons dans un tableau (N,6) fixe ;
        - préalloue k1,k2,k3,k4,tmp ;
        - calcule le RHS en place ;
        - évite X[active] = rk4_step(...) qui réalloue/compacte à chaque step ;
        - sync CPU active.any().get() seulement tous les check_interval steps.

    C'est moins lisible, donc on garde les fonctions classiques au-dessus comme référence.
    """
    if r_escape is None:
        r_escape = r

    f0 = 1.0 - 2.0 * M / r
    beta0 = cp.linspace(0.0, cp.pi, RAYS_NUMBER, dtype=cp.float64)

    ur0 = -cp.cos(beta0)
    ubeta0 = cp.sin(beta0) / r
    ut0 = cp.sqrt((ur0**2 / f0 + r**2 * ubeta0**2) / f0)

    X = cp.empty((RAYS_NUMBER, 6), dtype=cp.float64)
    X[:, 0] = 0.0
    X[:, 1] = r
    X[:, 2] = beta0
    X[:, 3] = ut0
    X[:, 4] = ur0
    X[:, 5] = ubeta0

    # h.shape == (N,1), pour multiplier correctement les 6 composantes.
    # Petit pas près de beta=pi, là où sin(beta/2)=1 ; grand pas près de beta=0.
    h = h_step * (1.0 - 0.9 * cp.sin(0.5 * beta0))[:, None]

    active = cp.ones(RAYS_NUMBER, dtype=cp.bool_)
    capture_radius = 2.05 * M

    k1 = cp.empty_like(X)
    k2 = cp.empty_like(X)
    k3 = cp.empty_like(X)
    k4 = cp.empty_like(X)
    tmp = cp.empty_like(X)

    def rhs_inplace(Y, out):
        rY = Y[:, 1]
        ut = Y[:, 3]
        ur = Y[:, 4]
        ubeta = Y[:, 5]

        valid = active & cp.isfinite(Y).all(axis=1) & (rY > capture_radius)

        # Évite les divisions dangereuses pour les rayons inactifs/capturés.
        r_safe = cp.where(valid, rY, 1.0)
        ut_safe = cp.where(valid, ut, 0.0)
        ur_safe = cp.where(valid, ur, 0.0)
        ubeta_safe = cp.where(valid, ubeta, 0.0)

        f = 1.0 - 2.0 * M / r_safe
        inv_r = 1.0 / r_safe
        inv_r2 = inv_r * inv_r

        out[:, 0] = ut_safe
        out[:, 1] = ur_safe
        out[:, 2] = ubeta_safe
        out[:, 3] = -2.0 * M * inv_r2 / f * ut_safe * ur_safe
        out[:, 4] = (
            -M * f * inv_r2 * ut_safe**2
            + M * inv_r2 / f * ur_safe**2
            + r_safe * f * ubeta_safe**2
        )
        out[:, 5] = -2.0 * inv_r * ur_safe * ubeta_safe

    for step in range(MAX_STEPS):
        if step % check_interval == 0 and not bool(active.any().get()):
            break

        rhs_inplace(X, k1)

        cp.multiply(k1, h, out=tmp)
        tmp *= 0.5
        tmp += X
        rhs_inplace(tmp, k2)

        cp.multiply(k2, h, out=tmp)
        tmp *= 0.5
        tmp += X
        rhs_inplace(tmp, k3)

        cp.multiply(k3, h, out=tmp)
        tmp += X
        rhs_inplace(tmp, k4)

        # tmp = k1 + 2*k2 + 2*k3 + k4, sans allouer 2*k2/2*k3.
        tmp[...] = k1
        tmp += k2
        tmp += k2
        tmp += k3
        tmp += k3
        tmp += k4
        tmp *= h / 6.0
        X += tmp

        r_current = X[:, 1]
        ur_current = X[:, 4]

        invalid = ~cp.isfinite(X).all(axis=1)
        captured = r_current <= capture_radius
        escaped = (r_current >= r_escape) & (ur_current > 0.0)

        X[captured, 1] = 2.0 * M
        X[captured, 3:6] = 0.0

        active &= ~(invalid | captured | escaped)

    return X
