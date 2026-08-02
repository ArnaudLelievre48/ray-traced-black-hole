import cupy as cp
import numpy as np


_ORBITAL_GEODESIC_FUSED_SOURCE = r"""
#include <cuda_runtime.h>

__device__ __forceinline__ void geodesic_rhs(
    const double* state,
    double* derivative,
    const double mass,
    const double capture_radius
) {
    const double radius = state[1];
    bool valid = radius > capture_radius;
    #pragma unroll
    for (int component = 0; component < 6; ++component) {
        valid = valid && isfinite(state[component]);
    }
    if (!valid) {
        #pragma unroll
        for (int component = 0; component < 6; ++component) derivative[component] = 0.0;
        return;
    }

    const double ut = state[3];
    const double ur = state[4];
    const double ubeta = state[5];
    const double inverse_radius = 1.0 / radius;
    const double inverse_radius2 = inverse_radius * inverse_radius;
    const double f = 1.0 - 2.0 * mass * inverse_radius;

    derivative[0] = ut;
    derivative[1] = ur;
    derivative[2] = ubeta;
    derivative[3] = -2.0 * mass * inverse_radius2 / f * ut * ur;
    derivative[4] =
        -mass * f * inverse_radius2 * ut * ut
        + mass * inverse_radius2 / f * ur * ur
        + radius * f * ubeta * ubeta;
    derivative[5] = -2.0 * inverse_radius * ur * ubeta;
}

extern "C" __global__
void orbital_geodesic_fused(
    double* __restrict__ output,
    const double initial_radius,
    const double mass,
    const int ray_count,
    const int max_steps,
    const double escape_radius,
    const double maximum_step,
    const int step_update_interval
) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= ray_count) return;

    const double pi = 3.1415926535897932384626433832795;
    const double beta0 = pi * (double)ray / (double)(ray_count - 1);
    const double f0 = 1.0 - 2.0 * mass / initial_radius;
    const double ur0 = -cos(beta0);
    const double ubeta0 = sin(beta0) / initial_radius;

    double state[6];
    state[0] = 0.0;
    state[1] = initial_radius;
    state[2] = beta0;
    state[3] = sqrt(
        (ur0 * ur0 / f0 + initial_radius * initial_radius * ubeta0 * ubeta0) / f0
    );
    state[4] = ur0;
    state[5] = ubeta0;

    const double capture_radius = 2.05 * mass;
    const double direction_ratio = fmin(fmax(beta0 / 0.20, 0.0), 1.0);
    const double direction_factor = 0.5 + 0.5 * direction_ratio;
    double step_size = maximum_step;
    double k1[6], k2[6], k3[6], k4[6], temporary[6];

    for (int step = 0; step < max_steps; ++step) {
        if (step % step_update_interval == 0) {
            const double radius_factor = fmin(fmax(
                (state[1] - capture_radius) / (12.0 * mass - capture_radius),
                0.0
            ), 1.0);
            const double factor = fmin(radius_factor, direction_factor);
            step_size = 0.05 + (maximum_step - 0.05) * factor;
        }

        geodesic_rhs(state, k1, mass, capture_radius);
        #pragma unroll
        for (int component = 0; component < 6; ++component) {
            temporary[component] = state[component] + 0.5 * step_size * k1[component];
        }
        geodesic_rhs(temporary, k2, mass, capture_radius);
        #pragma unroll
        for (int component = 0; component < 6; ++component) {
            temporary[component] = state[component] + 0.5 * step_size * k2[component];
        }
        geodesic_rhs(temporary, k3, mass, capture_radius);
        #pragma unroll
        for (int component = 0; component < 6; ++component) {
            temporary[component] = state[component] + step_size * k3[component];
        }
        geodesic_rhs(temporary, k4, mass, capture_radius);

        bool finite = true;
        #pragma unroll
        for (int component = 0; component < 6; ++component) {
            state[component] += (step_size / 6.0) * (
                k1[component] + 2.0 * k2[component] + 2.0 * k3[component] + k4[component]
            );
            finite = finite && isfinite(state[component]);
        }
        if (!finite) break;
        if (state[1] <= capture_radius) {
            state[1] = 2.0 * mass;
            state[3] = 0.0;
            state[4] = 0.0;
            state[5] = 0.0;
            break;
        }
        if (state[1] >= escape_radius && state[4] > 0.0) break;
    }

    double* destination = output + ray * 6;
    #pragma unroll
    for (int component = 0; component < 6; ++component) {
        destination[component] = state[component];
    }
}
"""

_ORBITAL_GEODESIC_FUSED_KERNEL = cp.RawKernel(
    _ORBITAL_GEODESIC_FUSED_SOURCE,
    "orbital_geodesic_fused",
    options=("--std=c++11",),
)

_DISK_CROSSING_FUSED_SOURCE = _ORBITAL_GEODESIC_FUSED_SOURCE + r"""
extern "C" __global__
void disk_crossing_samples_fused(
    double* __restrict__ beta_grid,
    double* __restrict__ beta_samples,
    double* __restrict__ radius_samples,
    const double initial_radius,
    const double mass,
    const int ray_count,
    const int max_steps,
    const double inner_radius,
    const double outer_radius,
    const int maximum_samples,
    const int sample_interval,
    const double step_size,
    const double maximum_beta
) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= ray_count) return;

    const double pi = 3.1415926535897932384626433832795;
    const double beta0 = pi * (double)ray / (double)(ray_count - 1);
    beta_grid[ray] = beta0;
    const int sample_offset = ray * maximum_samples;
    for (int sample = 0; sample < maximum_samples; ++sample) {
        beta_samples[sample_offset + sample] = nan("");
        radius_samples[sample_offset + sample] = nan("");
    }

    const double f0 = 1.0 - 2.0 * mass / initial_radius;
    const double ur0 = -cos(beta0);
    const double ubeta0 = sin(beta0) / initial_radius;
    double state[6];
    state[0] = 0.0;
    state[1] = initial_radius;
    state[2] = beta0;
    state[3] = sqrt(
        (ur0 * ur0 / f0 + initial_radius * initial_radius * ubeta0 * ubeta0) / f0
    );
    state[4] = ur0;
    state[5] = ubeta0;

    const double capture_radius = 2.05 * mass;
    int stored_samples = 0;
    double k1[6], k2[6], k3[6], k4[6], temporary[6];

    for (int step = 0; step < max_steps; ++step) {
        if (step % sample_interval == 0
                && state[1] >= inner_radius
                && state[1] <= outer_radius
                && state[2] >= 0.0
                && state[2] <= maximum_beta
                && stored_samples < maximum_samples) {
            beta_samples[sample_offset + stored_samples] = state[2];
            radius_samples[sample_offset + stored_samples] = state[1];
            ++stored_samples;
        }

        geodesic_rhs(state, k1, mass, capture_radius);
        #pragma unroll
        for (int component = 0; component < 6; ++component) {
            temporary[component] = state[component] + 0.5 * step_size * k1[component];
        }
        geodesic_rhs(temporary, k2, mass, capture_radius);
        #pragma unroll
        for (int component = 0; component < 6; ++component) {
            temporary[component] = state[component] + 0.5 * step_size * k2[component];
        }
        geodesic_rhs(temporary, k3, mass, capture_radius);
        #pragma unroll
        for (int component = 0; component < 6; ++component) {
            temporary[component] = state[component] + step_size * k3[component];
        }
        geodesic_rhs(temporary, k4, mass, capture_radius);

        bool finite = true;
        #pragma unroll
        for (int component = 0; component < 6; ++component) {
            state[component] += (step_size / 6.0) * (
                k1[component] + 2.0 * k2[component] + 2.0 * k3[component] + k4[component]
            );
            finite = finite && isfinite(state[component]);
        }
        if (!finite
                || state[1] <= capture_radius
                || (state[1] >= initial_radius && state[4] > 0.0)
                || state[2] > maximum_beta) {
            break;
        }
    }
}
"""

_DISK_CROSSING_FUSED_KERNEL = cp.RawKernel(
    _DISK_CROSSING_FUSED_SOURCE,
    "disk_crossing_samples_fused",
    options=("--std=c++11",),
)


def orbital_geodesic_fused(
    r,
    M,
    RAYS_NUMBER=1000,
    MAX_STEPS=20_000,
    r_escape=None,
    h_step=2,
    h_update_interval=16,
):
    """Intègre chaque rayon dans un kernel CUDA persistant, sans boucle Python."""
    if RAYS_NUMBER < 2:
        raise ValueError("RAYS_NUMBER doit être supérieur ou égal à 2")
    if h_update_interval <= 0:
        raise ValueError("h_update_interval doit être strictement positif")
    if r_escape is None:
        r_escape = r

    output = cp.empty((RAYS_NUMBER, 6), dtype=cp.float64)
    threads = 128
    _ORBITAL_GEODESIC_FUSED_KERNEL(
        ((RAYS_NUMBER + threads - 1) // threads,),
        (threads,),
        (
            output,
            np.float64(r),
            np.float64(M),
            np.int32(RAYS_NUMBER),
            np.int32(MAX_STEPS),
            np.float64(r_escape),
            np.float64(h_step),
            np.int32(h_update_interval),
        ),
    )
    return output


def orbital_disk_crossing_samples_fused(
    r,
    M,
    RAYS_NUMBER=1000,
    MAX_STEPS=10_000,
    r_inner=None,
    r_outer=None,
    max_samples=32,
    sample_interval=4,
    h_step=0.8,
    max_beta_turns=4,
):
    """Construit toute la LUT disque dans un seul lancement CUDA."""
    if RAYS_NUMBER < 2:
        raise ValueError("RAYS_NUMBER doit être supérieur ou égal à 2")
    if sample_interval <= 0 or max_samples <= 0:
        raise ValueError("sample_interval et max_samples doivent être positifs")
    if r_inner is None:
        r_inner = 6.0 * M
    if r_outer is None:
        r_outer = 30.0 * M

    beta_grid = cp.empty(RAYS_NUMBER, dtype=cp.float64)
    beta_samples = cp.empty((RAYS_NUMBER, max_samples), dtype=cp.float64)
    radius_samples = cp.empty_like(beta_samples)
    threads = 128
    # L'ancienne expression `step % 0.25 == 0` échantillonnait de fait à
    # chaque pas. Le kernel travaille avec un intervalle entier explicite.
    sample_every = max(1, int(round(sample_interval)))
    _DISK_CROSSING_FUSED_KERNEL(
        ((RAYS_NUMBER + threads - 1) // threads,),
        (threads,),
        (
            beta_grid,
            beta_samples,
            radius_samples,
            np.float64(r),
            np.float64(M),
            np.int32(RAYS_NUMBER),
            np.int32(MAX_STEPS),
            np.float64(r_inner),
            np.float64(r_outer),
            np.int32(max_samples),
            np.int32(sample_every),
            np.float64(h_step),
            np.float64(max_beta_turns * 2.0 * np.pi),
        ),
    )
    return beta_grid, beta_samples, radius_samples


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


def direction_step_factor(beta0):
    """Facteur directionnel pré-calculable pour h adaptatif rapide.

    beta=0 vise le trou noir : on réduit le pas, mais pas jusqu'au minimum.
    Hors du cône beta_safe, le facteur vaut 1.
    """
    beta_safe = 0.20  # ~11.5 degrés
    factor = cp.clip(beta0 / beta_safe, 0.0, 1.0)
    return 0.5 + 0.5 * factor


def adaptive_step_from_radius_and_beta(X, beta0, M, h_max):
    """Version lisible : h absolu entre 0.05 et h_max.

    Heuristique volontairement cheap : linéaire en r, direction prévisible en beta,
    pas de puissances. Le but est que le calcul de h ne devienne pas le bottleneck.
    """
    h = cp.empty((X.shape[0], 1), dtype=X.dtype)
    update_adaptive_step_inplace(h, X, direction_step_factor(beta0), M, h_max)
    return h


def update_adaptive_step_inplace(h, X, direction_factor, M, h_max):
    """Update rapide de h, in-place.

    h varie entre 0.05 et h_max typiquement 0.1--0.5.
    direction_factor est pré-calculé une seule fois depuis beta0.
    """
    r = X[:, 1]

    h_min = 0.05
    r_near = 2.05 * M
    r_far = 12.0 * M

    radius_factor = cp.clip((r - r_near) / (r_far - r_near), 0.0, 1.0)
    factor = cp.minimum(radius_factor, direction_factor)

    h[:, 0] = h_min + (h_max - h_min) * factor


# beta = 0 => vers le trou noir
# beta = pi => à l'opposé du trou noir
def circular_timelike_ubeta(r, M):
    """u^beta=dβ/dτ pour une orbite circulaire massive Schwarzschild.

    Omega=dβ/dt=sqrt(M/r^3), u^t=1/sqrt(1-3M/r), donc
    u^beta=Omega*u^t. Stable seulement pour r > 6M.
    """
    omega = cp.sqrt(M / r**3)
    ut = 1.0 / cp.sqrt(1.0 - 3.0 * M / r)
    return omega * ut


def orbital_geodesic(r, M, RAYS_NUMBER=1000, MAX_STEPS=10_000, r_escape=None, h_step=1, isParticle=False):
    # le photon respecte ds^2 = -f(r) dt^2 + f(r)^-1 dr^2 + r^2 dbeta^2
    f = 1.0 - 2.0 * M / r
    beta = cp.linspace(0, cp.pi, RAYS_NUMBER, dtype=cp.float64)

    if isParticle:
        # Pour des particules massives, beta est une coordonnée orbitale, pas
        # l'angle initial du rayon photonique. On place donc les particules sur
        # un anneau complet avec une vitesse proche de l'orbite circulaire.
        beta = cp.linspace(0.0, 2.0 * cp.pi, RAYS_NUMBER, endpoint=False, dtype=cp.float64)
        ur = cp.zeros_like(beta)
        ubeta_circ = circular_timelike_ubeta(r, M)
        ubeta = ubeta_circ * (1.0 + 0.05 * (cp.random.rand(RAYS_NUMBER, dtype=cp.float64) - 0.5))
        # Normalisation massive : -f ut² + f⁻¹ ur² + r² ubeta² = -1.
        ut = cp.sqrt((1.0 + ur**2 / f + r**2 * ubeta**2) / f)
    else:
        ur = -cp.cos(beta)
        ubeta = cp.sin(beta) / r
        # Normalisation photonique : -f ut² + f⁻¹ ur² + r² ubeta² = 0.
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

    # Ancienne heuristique direction seule, gardée pour référence :
    # h = h_step * (1.0 - 0.95 * cp.sin(0.5 * beta))[:, None]
    # Problème : elle ne regardait pas la distance courante au trou noir.
    h = adaptive_step_from_radius_and_beta(X, beta, M, h_step)

    if r_escape is None:
        r_escape = r

    active = cp.ones(X.shape[0], dtype=cp.bool_)
    capture_radius = 2.05 * M

    for step in range(MAX_STEPS):
        if not bool(active.any().get()):
            break

        # h dépend du r courant : ralentit quand le rayon approche du trou noir.
        if step % 300 == 0:
            h = adaptive_step_from_radius_and_beta(X, beta, M, h_step)

        # On n'intègre que les rayons actifs. Sinon les rayons déjà échappés
        # continuent vers r énorme pendant MAX_STEPS et peuvent overflow.
        X[active] = rk4_step(X[active], h[active], M)

        r_current = X[:, 1]
        ur_current = X[:, 4]

        invalid = ~cp.isfinite(X).all(axis=1)
        captured = r_current <= capture_radius
        escaped = (r_current >= r_escape) & (ur_current > 0.0) & (not isParticle)

        # Nettoyage des états capturés : le dernier pas RK4 peut passer sous
        # l'horizon en coordonnées de Schwarzschild, donc on fige proprement.
        X[captured, 1] = 2.0 * M
        X[captured, 3:6] = 0.0

        active = ~(invalid | captured | escaped)

    return X






def orbital_geodesic_fast(
    r,
    M,
    RAYS_NUMBER=1000,
    MAX_STEPS=20_000,
    r_escape=None,
    h_step=2,
    check_interval=25,
    h_update_interval=16,
    isParticle=False,
    use_fused=True,
):
    """Version performance de orbital_geodesic.

    Même état :
        X[ray_id] = [t, r, beta, ut, ur, ubeta]

    Différence avec orbital_geodesic :
        - garde tous les rayons dans un tableau (N,6) fixe ;
        - préalloue k1,k2,k3,k4,tmp ;
        - calcule le RHS en place ;
        - évite X[active] = rk4_step(...) qui réalloue/compacte à chaque step ;
        - sync CPU active.any().get() seulement tous les check_interval steps ;
        - update du pas adaptatif seulement tous les h_update_interval steps.

    C'est moins lisible, donc on garde les fonctions classiques au-dessus comme référence.
    """
    if use_fused and not isParticle:
        return orbital_geodesic_fused(
            r,
            M,
            RAYS_NUMBER=RAYS_NUMBER,
            MAX_STEPS=MAX_STEPS,
            r_escape=r_escape,
            h_step=h_step,
            h_update_interval=h_update_interval,
        )

    if r_escape is None:
        r_escape = r

    f0 = 1.0 - 2.0 * M / r
    beta0 = cp.linspace(0.0, cp.pi, RAYS_NUMBER, dtype=cp.float64)

    if isParticle:
        # Pour des particules massives, beta0 est la position angulaire autour
        # de l'anneau. On initialise ur=0 et u^beta proche de l'orbite circulaire.
        beta0 = cp.linspace(0.0, 2.0 * cp.pi, RAYS_NUMBER, endpoint=False, dtype=cp.float64)
        ur0 = cp.zeros_like(beta0)
        ubeta_circ = circular_timelike_ubeta(r, M)
        ubeta0 = ubeta_circ * (1.0 + 0.05 * (cp.random.rand(RAYS_NUMBER, dtype=cp.float64) - 0.5))
        ut0 = cp.sqrt((1.0 + ur0**2 / f0 + r**2 * ubeta0**2) / f0)
    else:
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
    # Ancienne heuristique direction seule, gardée pour référence :
    # h = h_step * (1.0 - 0.9 * cp.sin(0.5 * beta0))[:, None]
    # Problème : beta=0 vers le trou noir avait un gros pas, et la distance
    # courante au trou noir n'était pas prise en compte.
    h = cp.empty((RAYS_NUMBER, 1), dtype=cp.float64)
    direction_factor = direction_step_factor(beta0)
    update_adaptive_step_inplace(h, X, direction_factor, M, h_step)

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

        # Calculer h à chaque step coûtait plus cher que prévu sur GPU.
        # On l'actualise seulement de temps en temps : r évolue continûment,
        # donc h n'a pas besoin d'être recalculé à chaque sous-pas RK4.
        if step % h_update_interval == 0:
            update_adaptive_step_inplace(h, X, direction_factor, M, h_step)

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
        escaped = (r_current >= r_escape) & (ur_current > 0.0) & (not isParticle)

        X[captured, 1] = 2.0 * M
        X[captured, 3:6] = 0.0

        active &= ~(invalid | captured | escaped)

    return X


def orbital_disk_crossing_samples(
    r,
    M,
    RAYS_NUMBER=1000,
    MAX_STEPS=10_000,
    r_inner=None,
    r_outer=None,
    max_samples=32,
    sample_interval=4,
    h_step=0.8,
    max_beta_turns=4,
    use_fused=True,
):
    """LUT auxiliaire pour le disque d'accrétion.

    On garde quelques valeurs de beta_coord lorsque chaque photon passe dans
    l'anneau radial du disque :

        r_inner <= r(lambda) <= r_outer

    Le rendu peut ensuite tester si ces beta_coord correspondent à un crossing
    du plan du disque z=z_BH, sans réintégrer la géodésique par pixel.

    Retour :
        beta_grid.shape == (RAYS_NUMBER,)
        beta_samples.shape == (RAYS_NUMBER, max_samples)
        r_samples.shape == (RAYS_NUMBER, max_samples)

    beta_samples vaut NaN quand aucun sample n'est disponible.
    """
    if use_fused:
        return orbital_disk_crossing_samples_fused(
            r,
            M,
            RAYS_NUMBER=RAYS_NUMBER,
            MAX_STEPS=MAX_STEPS,
            r_inner=r_inner,
            r_outer=r_outer,
            max_samples=max_samples,
            sample_interval=sample_interval,
            h_step=h_step,
            max_beta_turns=max_beta_turns,
        )

    if r_inner is None:
        r_inner = 6.0 * M
    if r_outer is None:
        r_outer = 30.0 * M

    beta_grid = cp.linspace(0.0, cp.pi, RAYS_NUMBER, dtype=cp.float64)
    f0 = 1.0 - 2.0 * M / r

    ur0 = -cp.cos(beta_grid)
    ubeta0 = cp.sin(beta_grid) / r
    ut0 = cp.sqrt((ur0**2 / f0 + r**2 * ubeta0**2) / f0)

    X = cp.empty((RAYS_NUMBER, 6), dtype=cp.float64)
    X[:, 0] = 0.0
    X[:, 1] = r
    X[:, 2] = beta_grid
    X[:, 3] = ut0
    X[:, 4] = ur0
    X[:, 5] = ubeta0

    beta_samples = cp.full((RAYS_NUMBER, max_samples), cp.nan, dtype=cp.float64)
    r_samples = cp.full((RAYS_NUMBER, max_samples), cp.nan, dtype=cp.float64)
    sample_counts = cp.zeros(RAYS_NUMBER, dtype=cp.int32)

    active = cp.ones(RAYS_NUMBER, dtype=cp.bool_)
    capture_radius = 2.05 * M
    r_escape = r
    max_beta = max_beta_turns * 2.0 * cp.pi

    for step in range(MAX_STEPS):
        if step % 25 == 0 and not bool(active.any().get()):
            break

        if step % sample_interval == 0:
            r_current = X[:, 1]
            beta_current = X[:, 2]
            in_disk_radial_zone = (
                active
                & (r_current >= r_inner)
                & (r_current <= r_outer)
                & (beta_current >= 0.0)
                & (beta_current <= max_beta)
                & (sample_counts < max_samples)
            )
            idx = cp.nonzero(in_disk_radial_zone)[0]
            if idx.size > 0:
                slots = sample_counts[idx]
                beta_samples[idx, slots] = beta_current[idx]
                r_samples[idx, slots] = r_current[idx]
                sample_counts[idx] += 1

        X[active] = rk4_step(X[active], h_step, M)

        r_current = X[:, 1]
        ur_current = X[:, 4]
        beta_current = X[:, 2]

        invalid = ~cp.isfinite(X).all(axis=1)
        captured = r_current <= capture_radius
        escaped = (r_current >= r_escape) & (ur_current > 0.0)
        too_many_turns = beta_current > max_beta

        active &= ~(invalid | captured | escaped | too_many_turns)

    return beta_grid, beta_samples, r_samples
