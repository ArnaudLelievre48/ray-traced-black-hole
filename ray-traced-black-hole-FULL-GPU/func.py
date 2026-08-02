import cupy as cp
import imageio.v3 as iio
import numpy as np


SKYBOX = None


def load_skybox(path):
    """Charge une skybox equirectangulaire en float32 RGB dans [0, 1], puis l'envoie sur GPU."""
    global SKYBOX

    skybox_cpu = iio.imread(path).astype("float32")

    if skybox_cpu.max() > 1.0:
        skybox_cpu /= 255.0

    if skybox_cpu.ndim == 3 and skybox_cpu.shape[2] == 4:
        skybox_cpu = skybox_cpu[:, :, :3]

    SKYBOX = cp.asarray(skybox_cpu, dtype=cp.float32)
    return SKYBOX


def normalize(v, axis=-1, eps=1e-15):
    norm = cp.linalg.norm(v, axis=axis, keepdims=True)
    safe_norm = cp.where(norm > eps, norm, 1.0)
    out = v / safe_norm
    return cp.where(norm > eps, out, 0.0)


def camera_pixel_directions(CAMERA):
    """Direction monde de chaque pixel, sans GR.

    Retour : directions[height, width, 3]
    Convention d'orientation reprise de l'ancienne version :
        angle_vertical, angle_horizontal définissent le forward en sphériques.
    """
    width = CAMERA["width"]
    height = CAMERA["height"]
    aspect_ratio = width / height

    pixel_x = cp.arange(width, dtype=cp.float64)[None, :]
    pixel_y = cp.arange(height, dtype=cp.float64)[:, None]

    pixel_offset_x = (pixel_x + 0.5 - width / 2) / width
    pixel_offset_y = (pixel_y + 0.5 - height / 2) / height

    angle_vertical = CAMERA["angle_vertical"]
    angle_horizontal = CAMERA["angle_horizontal"]

    sin_av = cp.sin(angle_vertical)
    cos_av = cp.cos(angle_vertical)
    sin_ah = cp.sin(angle_horizontal)
    cos_ah = cp.cos(angle_horizontal)

    forward = cp.stack([
        sin_av * cos_ah,
        sin_av * sin_ah,
        cos_av,
    ]).astype(cp.float64)
    forward = forward / cp.linalg.norm(forward)

    e_theta_cam = cp.stack([
        cos_av * cos_ah,
        cos_av * sin_ah,
        -sin_av,
    ]).astype(cp.float64)
    e_phi_cam = cp.stack([
        -sin_ah,
        cos_ah,
        cp.asarray(0.0),
    ]).astype(cp.float64)

    right = e_phi_cam
    up = -e_theta_cam

    screen_width = CAMERA.get("camera_virtual_screen_width", 1.0)
    screen_height = screen_width / aspect_ratio
    distance_from_virtual_screen = CAMERA.get(
        "distance_from_virtual_screen",
        screen_width / (2.0 * cp.tan(CAMERA["FOV"] / 2.0)),
    )

    screen_x = pixel_offset_x * screen_width
    screen_y = pixel_offset_y * screen_height

    directions = (
        distance_from_virtual_screen * forward[None, None, :]
        + screen_x[:, :, None] * right[None, None, :]
        - screen_y[:, :, None] * up[None, None, :]
    )

    return normalize(directions, axis=2)


def bilinear_skybox_sample(directions, skybox=None):
    """Sample equirectangulaire bilinéaire depuis des directions monde.

    directions.shape = (..., 3)
    retour.shape = (..., 3)
    """
    if skybox is None:
        skybox = SKYBOX
    if skybox is None:
        raise ValueError("Skybox non chargée : appelle load_skybox(...) avant.")

    directions = normalize(directions, axis=-1)
    dx = directions[..., 0]
    dy = directions[..., 1]
    dz = cp.clip(directions[..., 2], -1.0, 1.0)

    phi = cp.arctan2(dy, dx)
    theta = cp.arccos(dz)

    u = ((phi + cp.pi) / (2.0 * cp.pi)) % 1.0
    v = cp.clip(theta / cp.pi, 0.0, 1.0)

    height, width, _ = skybox.shape

    x = u * width
    y = v * (height - 1)

    x_floor = cp.floor(x)
    y_floor = cp.floor(y)

    x0 = x_floor.astype(cp.int64) % width
    x1 = (x0 + 1) % width
    y0 = y_floor.astype(cp.int64)
    y1 = cp.clip(y0 + 1, 0, height - 1)

    tx = (x - x_floor)[..., None]
    ty = (y - y_floor)[..., None]

    c00 = skybox[y0, x0]
    c10 = skybox[y0, x1]
    c01 = skybox[y1, x0]
    c11 = skybox[y1, x1]

    c0 = (1.0 - tx) * c00 + tx * c10
    c1 = (1.0 - tx) * c01 + tx * c11
    return ((1.0 - ty) * c0 + ty * c1).astype(cp.float32)


def render_skybox_from_orbital_lut(CAMERA, BLACKHOLE, beta_grid, final_states, skybox=None):
    """Rend l'image via LUT orbitale Schwarzschild.

    beta_grid[i] : angle initial échantillonné, avec beta=0 vers le trou noir.
    final_states[i] = [t, r, beta_coord, ut, ur, ubeta] après intégration.

    Important : dans GR.orbital_geodesic actuel, la coordonnée beta initiale vaut
    beta_grid. La déviation géométrique utilisée ici est donc :
        delta = beta_final - beta_initial

    Pour chaque pixel :
        direction_pixel -> beta_pixel + plan orbital local
        interpolation delta(beta_pixel), ur(beta_pixel), ubeta(beta_pixel)
        reconstruction de la direction finale monde
        sampling skybox
    """
    if skybox is None:
        skybox = SKYBOX
    if skybox is None:
        raise ValueError("Skybox non chargée : appelle load_skybox(...) avant.")

    M = BLACKHOLE["MASS"]
    beta_grid = cp.asarray(beta_grid, dtype=cp.float64)
    final_states = cp.asarray(final_states, dtype=cp.float64)

    camera_pos = cp.asarray([CAMERA["x"], CAMERA["y"], CAMERA["z"]], dtype=cp.float64)
    blackhole_pos = cp.asarray([BLACKHOLE["x"], BLACKHOLE["y"], BLACKHOLE["z"]], dtype=cp.float64)

    # e_center : caméra -> trou noir, donc beta=0 correspond à direction_pixel=e_center.
    e_center = normalize(blackhole_pos - camera_pos, axis=0)
    # e_r0 : trou noir -> caméra, position radiale initiale du photon.
    e_r0 = -e_center

    directions = camera_pixel_directions(CAMERA)
    cos_beta = cp.clip(cp.sum(directions * e_center[None, None, :], axis=2), -1.0, 1.0)
    beta_pixel = cp.arccos(cos_beta)
    sin_beta = cp.sqrt(cp.maximum(1.0 - cos_beta**2, 0.0))

    # Direction latérale du plan orbital : direction = cos(beta)*e_center + sin(beta)*e_side.
    side = directions - cos_beta[:, :, None] * e_center[None, None, :]

    # Fallback pour beta≈0 ou beta≈pi, où le plan orbital est dégénéré.
    fallback_z = cp.cross(e_center, cp.asarray([0.0, 0.0, 1.0], dtype=cp.float64))
    fallback_y = cp.cross(e_center, cp.asarray([0.0, 1.0, 0.0], dtype=cp.float64))
    # Sélection sur le GPU : l'ancien `.get()` imposait une synchronisation
    # GPU->CPU à chaque frame, même avant l'étape d'affichage.
    fallback = cp.where(cp.linalg.norm(fallback_z) >= 1e-12, fallback_z, fallback_y)
    fallback = fallback / cp.linalg.norm(fallback)

    e_side = cp.where(
        sin_beta[:, :, None] > 1e-12,
        side / cp.maximum(sin_beta[:, :, None], 1e-12),
        fallback[None, None, :],
    )
    e_side = normalize(e_side, axis=2)

    beta_flat = beta_pixel.ravel()

    beta_final_grid = final_states[:, 2]
    delta_grid = beta_final_grid - beta_grid
    r_final_grid = final_states[:, 1]
    ur_final_grid = final_states[:, 4]
    ubeta_final_grid = final_states[:, 5]
    captured_grid = r_final_grid <= 2.0 * M

    delta = cp.interp(beta_flat, beta_grid, delta_grid).reshape(beta_pixel.shape)
    r_final = cp.interp(beta_flat, beta_grid, r_final_grid).reshape(beta_pixel.shape)
    ur_final = cp.interp(beta_flat, beta_grid, ur_final_grid).reshape(beta_pixel.shape)
    ubeta_final = cp.interp(beta_flat, beta_grid, ubeta_final_grid).reshape(beta_pixel.shape)
    captured = (cp.interp(beta_flat, beta_grid, captured_grid.astype(cp.float64)) > 0.5).reshape(beta_pixel.shape)

    cos_delta = cp.cos(delta)
    sin_delta = cp.sin(delta)

    # Base locale finale dans le plan orbital.
    e_r_final = (
        cos_delta[:, :, None] * e_r0[None, None, :]
        + sin_delta[:, :, None] * e_side
    )
    e_beta_final = (
        -sin_delta[:, :, None] * e_r0[None, None, :]
        + cos_delta[:, :, None] * e_side
    )

    final_direction = (
        ur_final[:, :, None] * e_r_final
        + (r_final * ubeta_final)[:, :, None] * e_beta_final
    )
    final_direction = normalize(final_direction, axis=2)

    image = bilinear_skybox_sample(final_direction, skybox)
    image[captured] = cp.asarray([0.0, 0.0, 0.0], dtype=cp.float32)

    return image, beta_pixel, final_direction


def camera_position(camera):
    return np.array([camera["x"], camera["y"], camera["z"]], dtype=np.float64)

def blackhole_position(blackhole):
    return np.array([blackhole["x"], blackhole["y"], blackhole["z"]], dtype=np.float64)

def camera_radius(camera, blackhole):
    return np.linalg.norm(camera_position(camera) - blackhole_position(blackhole))

def camera_basis(camera):
    """Retourne forward/right/up en NumPy, même convention que func.camera_pixel_directions."""
    av = camera["angle_vertical"]
    ah = camera["angle_horizontal"]

    forward = np.array([
        np.sin(av) * np.cos(ah),
        np.sin(av) * np.sin(ah),
        np.cos(av),
    ], dtype=np.float64)
    forward /= np.linalg.norm(forward)

    e_theta = np.array([
        np.cos(av) * np.cos(ah),
        np.cos(av) * np.sin(ah),
        -np.sin(av),
    ], dtype=np.float64)
    e_phi = np.array([-np.sin(ah), np.cos(ah), 0.0], dtype=np.float64)

    right = e_phi
    up = -e_theta
    return forward, right, up
