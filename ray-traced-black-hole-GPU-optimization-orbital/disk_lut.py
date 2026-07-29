import time

import cupy as cp
import numpy as np

import GR
import func


DISK_COLOR = cp.asarray([1.0, 0.42, 0.05], dtype=cp.float32)


def compute_radiuses_factors(M, factor=1.7):
    L = factor*2*np.sqrt(3)*M
    r_min_factor = ( (L**2) / (2.0*M) ) * (1 - np.sqrt(1 - ( (12 * M**2) / (L**2) ) ) ) / M
    r_max_factor = ( (L**2) / (2.0*M) ) * (1 + np.sqrt(1 - ( (12 * M**2) / (L**2) ) ) ) / M
    return r_min_factor, r_max_factor

def compute_disk_crossing_lut(
    r,
    M,
    rays_number,
    max_steps,
    max_samples=256,
    sample_interval=0.25,
    h_step=0.5,
    max_beta_turns=8,
):
    r_inner_factor, r_outer_factor = compute_radiuses_factors(M)
    print(r_inner_factor, r_outer_factor)
    """Calcule la LUT disque : beta_initial -> beta_coord dans l'anneau radial.

    Cette LUT est indépendante de l'orientation de caméra : elle dépend seulement
    de r_camera et de M, comme la LUT de déviation principale.
    """
    t0 = time.perf_counter()
    print(f"computing disk crossing LUT r={r:.3f}...")
    beta_grid, beta_samples, r_samples = GR.orbital_disk_crossing_samples(
        r,
        M,
        RAYS_NUMBER=rays_number,
        MAX_STEPS=max_steps,
        r_inner=r_inner_factor * M,
        r_outer=r_outer_factor * M,
        max_samples=max_samples,
        sample_interval=sample_interval,
        h_step=h_step,
        max_beta_turns=max_beta_turns,
    )
    cp.cuda.Stream.null.synchronize()
    print(f"disk crossing LUT done in {time.perf_counter() - t0:.3f}s")
    return beta_grid, beta_samples, r_samples


def disk_plane_crossing_beta(e_r0, e_side, blackhole_z=0.0):
    """Angle beta où le plan orbital du pixel croise le plan du disque z=z_BH.

    pos_world = BH + r * (cos(beta) e_r0 + sin(beta) e_side)
    Pour le disque dans le plan (x,y), on veut pos_world.z == BH.z :

        cos(beta) e_r0.z + sin(beta) e_side.z = 0

    Solution modulo pi : beta_disk = atan2(-e_r0.z, e_side.z) mod pi.
    blackhole_z est volontairement inutilisé ici : après soustraction de BH, il
    disparaît de l'équation. Il reste dans la signature pour rendre l'intention claire.
    """
    del blackhole_z
    return cp.mod(cp.arctan2(-e_r0[2], e_side[..., 2]), cp.pi)


def angular_distance_mod_pi(beta, beta_ref):
    """Distance angulaire à beta_ref + k*pi."""
    return cp.abs(cp.mod(beta - beta_ref + 0.5 * cp.pi, cp.pi) - 0.5 * cp.pi)


def circular_orbit_specific_energy(r, M):
    """Énergie spécifique E/m d'une orbite circulaire massive Schwarzschild.

    e = (1 - 2M/r) / sqrt(1 - 3M/r)

    Le binding 1-e donne une échelle d'énergie disponible. Ce n'est pas un
    modèle d'accrétion hydrodynamique, mais c'est une base GR simple et locale.
    """
    r_safe = cp.maximum(r, 3.001 * M)
    return (1.0 - 2.0 * M / r_safe) / cp.sqrt(1.0 - 3.0 * M / r_safe)


def blackbody_rgb_from_temperature(T):
    """Approximation RGB d'un corps noir, T en Kelvin visuels.

    Approximation de Tanner Helland, vectorisée CuPy. Elle est suffisante pour
    obtenir une couleur cohérente visuellement sans intégrer Planck sur les
    courbes CIE à chaque frame.
    """
    T = cp.clip(T, 1000.0, 40000.0) / 100.0

    red = cp.where(
        T <= 66.0,
        255.0,
        329.698727446 * cp.power(T - 60.0, -0.1332047592),
    )
    green = cp.where(
        T <= 66.0,
        99.4708025861 * cp.log(T) - 161.1195681661,
        288.1221695283 * cp.power(T - 60.0, -0.0755148492),
    )
    blue = cp.where(
        T >= 66.0,
        255.0,
        cp.where(T <= 19.0, 0.0, 138.5177312231 * cp.log(T - 10.0) - 305.0447927307),
    )

    rgb = cp.stack([red, green, blue], axis=-1)
    return cp.clip(rgb / 255.0, 0.0, 1.0).astype(cp.float32)


def disk_blackbody_emission_from_radius(r, M, r_inner, r_outer):
    """Couleur/intensité du disque ne dépendant que de r.

    Modèle disque mince newtonien/Schwarzschild simplifié :
        F(r) ∝ r^-3 * (1 - sqrt(r_in/r))

    Avantages pour notre rendu :
        - F=0 au bord interne : pas de matière lumineuse collée à l'ombre ;
        - maximum un peu après r_in ;
        - décroissance externe naturelle + extinction douce à r_out ;
        - T ∝ F^(1/4), puis RGB corps noir.

    Ce n'est pas Novikov-Thorne complet, mais c'est beaucoup plus cohérent que
    "tout rayon traversé brille" ou que binding(r) seul.
    """
    r_in = cp.maximum(cp.asarray(r_inner, dtype=cp.float64), 6.0 * M)
    r_out = cp.asarray(r_outer, dtype=cp.float64)
    r_safe = cp.maximum(r, 1e-6)

    in_disk = (r_safe >= r_in) & (r_safe <= r_out)

    # Flux disque mince : zéro sous r_in, zéro exactement à r_in, puis pic vers
    # r = (49/36) r_in avant de décroître comme r^-3.
    flux = cp.where(
        in_disk,
        r_safe**-3 * cp.maximum(1.0 - cp.sqrt(r_in / r_safe), 0.0),
        0.0,
    )

    # r_outer est un bord artificiel de notre disque fini. Le flux disque mince
    # infini ne s'y annule pas tout seul, donc on ajoute une extinction douce
    # sur le dernier quart radial du disque pour éviter un cut brutal lumineux.
    outer_fade_width = 0.25 * cp.maximum(r_out - r_in, 1e-6)
    x_outer = cp.clip((r_out - r_safe) / outer_fade_width, 0.0, 1.0)
    outer_taper = x_outer * x_outer * (3.0 - 2.0 * x_outer)  # smoothstep
    flux *= outer_taper

    r_peak = (49.0 / 36.0) * r_in
    flux_peak = r_peak**-3 * cp.maximum(1.0 - cp.sqrt(r_in / r_peak), 1e-12)
    # Le pic analytique est avant le taper externe pour nos tailles usuelles ;
    # il reste une bonne normalisation visuelle.
    F_norm = cp.clip(flux / (flux_peak + 1e-30), 0.0, 1.0)

    # Corps noir : F ∝ T^4, donc T ∝ F^(1/4). On remappe vers des Kelvin
    # visuels et on applique un redshift gravitationnel observé très simple.
    T_norm = cp.power(F_norm, 0.25)
    T_COLD = 1300.0
    T_HOT = 7200.0
    g_redshift = cp.sqrt(cp.clip(1.0 - 2.0 * M / r_safe, 0.05, 1.0))
    T_visual = (T_COLD + (T_HOT - T_COLD) * T_norm) * g_redshift

    color = blackbody_rgb_from_temperature(T_visual)

    # Warm tint léger : garde l'esprit corps noir, mais évite un disque trop bleu.
    WARM_TINT = cp.asarray([1.10, 0.92, 0.72], dtype=cp.float32)
    color = cp.clip(color * WARM_TINT, 0.0, 1.0)

    # Intensité surfacique : proportionnelle au flux normalisé, pas affine.
    # Donc sous r_in, au bord interne et au bord externe : 0.
    intensity = 1.65 * F_norm
    return color, intensity.astype(cp.float32)


def compute_disk_overlay_from_lut(
    camera,
    blackhole,
    disk_beta_grid,
    disk_beta_samples,
    disk_r_samples,
    beta_tolerance=0.1,
    max_beta_turns=8,
    emission=1.25,
    b_min_factor=0.0,
    b_max_factor=40.0,
):
    """Overlay disque depuis la LUT de crossing, sans réintégrer de géodésique.

    Pour chaque pixel :
        1. calcule beta_initial et le plan orbital du pixel ;
        2. interpole les beta_coord stockés par la LUT disque ;
        3. teste beta_coord ~= beta_disk + k*pi jusqu'à max_beta_turns ;
        4. retourne une image RGB d'émission orange.
    """
    M = blackhole["MASS"]
    width = int(camera["width"])
    height = int(camera["height"])

    camera_pos = cp.asarray([camera["x"], camera["y"], camera["z"]], dtype=cp.float64)
    blackhole_pos = cp.asarray([blackhole["x"], blackhole["y"], blackhole["z"]], dtype=cp.float64)

    e_center = func.normalize(blackhole_pos - camera_pos, axis=0)
    e_r0 = -e_center

    directions = func.camera_pixel_directions(camera)
    cos_beta = cp.clip(cp.sum(directions * e_center[None, None, :], axis=2), -1.0, 1.0)
    beta_pixel = cp.arccos(cos_beta)
    sin_beta = cp.sqrt(cp.maximum(1.0 - cos_beta**2, 0.0))

    # Filtre impact parameter : garde les rayons qui peuvent traverser le disque.
    # L'ancien filtre b <= 10M était beaucoup trop restrictif : il ne gardait
    # que la zone très proche du trou noir, alors que la LUT disque actuelle
    # stocke des crossings jusqu'à R_outer~100M.
    r_cam = cp.linalg.norm(camera_pos - blackhole_pos)
    f_cam = 1.0 - 2.0 * M / r_cam
    b = r_cam * sin_beta / cp.sqrt(f_cam)
    candidate = (b >= b_min_factor * M) & (b <= b_max_factor * M)

    side = directions - cos_beta[:, :, None] * e_center[None, None, :]
    fallback = cp.cross(e_center, cp.asarray([0.0, 0.0, 1.0], dtype=cp.float64))
    if float(cp.linalg.norm(fallback).get()) < 1e-12:
        fallback = cp.cross(e_center, cp.asarray([0.0, 1.0, 0.0], dtype=cp.float64))
    fallback = fallback / cp.linalg.norm(fallback)

    e_side = cp.where(
        sin_beta[:, :, None] > 1e-12,
        side / cp.maximum(sin_beta[:, :, None], 1e-12),
        fallback[None, None, :],
    )
    e_side = func.normalize(e_side, axis=2)

    beta_disk = disk_plane_crossing_beta(e_r0, e_side, blackhole_pos[2])

    beta_flat = beta_pixel.ravel()
    beta_disk_flat = beta_disk.ravel()
    candidate_flat = candidate.ravel()

    max_beta = max_beta_turns * 2.0 * cp.pi
    best_diff = cp.full(beta_flat.shape, cp.inf, dtype=cp.float64)
    hit = cp.zeros(beta_flat.shape, dtype=cp.bool_)

    disk_beta_grid = cp.asarray(disk_beta_grid, dtype=cp.float64)
    disk_beta_samples = cp.asarray(disk_beta_samples, dtype=cp.float64)
    disk_r_samples = cp.asarray(disk_r_samples, dtype=cp.float64)

    r_inner_factor, r_outer_factor = compute_radiuses_factors(M)
    r_inner = r_inner_factor * M
    r_outer = r_outer_factor * M
    overlay_flat = cp.zeros((beta_flat.size, 3), dtype=cp.float32)

    for sample_id in range(disk_beta_samples.shape[1]):
        beta_path = cp.interp(beta_flat, disk_beta_grid, disk_beta_samples[:, sample_id])
        r_path = cp.interp(beta_flat, disk_beta_grid, disk_r_samples[:, sample_id])
        valid = cp.isfinite(beta_path) & cp.isfinite(r_path) & (beta_path >= 0.0) & (beta_path <= max_beta)
        diff = angular_distance_mod_pi(beta_path, beta_disk_flat)
        diff_safe = cp.where(valid, diff, cp.inf)
        sample_hit = candidate_flat & valid & (diff_safe < beta_tolerance)
        hit |= sample_hit
        best_diff = cp.minimum(best_diff, diff_safe)

        closeness = cp.clip(1.0 - diff_safe / beta_tolerance, 0.0, 1.0).astype(cp.float32)
        r_safe = cp.where(valid, r_path, r_outer)
        color, bb_intensity = disk_blackbody_emission_from_radius(r_safe, M, r_inner, r_outer)
        sample_overlay = (
            emission
            * closeness[:, None]
            * bb_intensity[:, None]
            * color
            * sample_hit[:, None].astype(cp.float32)
        )
        overlay_flat = cp.maximum(overlay_flat, sample_overlay)

    overlay = overlay_flat.reshape(height, width, 3)
    return overlay.astype(cp.float32)
