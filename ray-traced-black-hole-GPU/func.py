import cupy as cp
import matplotlib.pyplot as plt


SKYBOX = None


def load_skybox(path):
    """Charge une skybox equirectangulaire en float32 RGB dans [0, 1], puis l'envoie sur GPU."""
    global SKYBOX

    skybox = plt.imread(path).astype("float32")

    if skybox.max() > 1.0:
        skybox /= 255.0

    if skybox.ndim == 3 and skybox.shape[2] == 4:
        skybox = skybox[:, :, :3]

    SKYBOX = cp.asarray(skybox, dtype=cp.float32)
    return SKYBOX


def skybox_color(lightrays, M=None):
    """Retourne l'image RGB de la skybox pour tous les rayons.

    Convention :
        lightrays[y, x] = [t, r, theta, phi, ut, ur, utheta, uphi]

    Skybox à l'infini : on lit la skybox avec la direction finale du photon :
        v = u^r e_r + r u^theta e_theta + r sin(theta) u^phi e_phi

    Si M est fourni, les rayons avec r <= 2M sont noirs.
    Les rayons non finis sont aussi noirs.
    """
    if SKYBOX is None:
        raise ValueError("Skybox non chargée : appelle func.load_skybox(...) avant skybox_color.")

    r = lightrays[:, :, 1]
    theta = lightrays[:, :, 2]
    phi = lightrays[:, :, 3]

    ur = lightrays[:, :, 5]
    utheta = lightrays[:, :, 6]
    uphi = lightrays[:, :, 7]

    sin_theta = cp.sin(theta)
    cos_theta = cp.cos(theta)
    sin_phi = cp.sin(phi)
    cos_phi = cp.cos(phi)

    direction_x = (
        ur * sin_theta * cos_phi
        + r * utheta * cos_theta * cos_phi
        - r * sin_theta * uphi * sin_phi
    )
    direction_y = (
        ur * sin_theta * sin_phi
        + r * utheta * cos_theta * sin_phi
        + r * sin_theta * uphi * cos_phi
    )
    direction_z = (
        ur * cos_theta
        - r * utheta * sin_theta
    )

    norm = cp.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
    valid = cp.isfinite(norm) & (norm > 0)

    direction_x = cp.where(valid, direction_x / norm, 0.0)
    direction_y = cp.where(valid, direction_y / norm, 0.0)
    direction_z = cp.where(valid, direction_z / norm, 1.0)

    sky_phi = cp.arctan2(direction_y, direction_x)
    sky_theta = cp.arccos(cp.clip(direction_z, -1.0, 1.0))

    u = ((sky_phi + cp.pi) / (2.0 * cp.pi)) % 1.0
    v = cp.clip(sky_theta / cp.pi, 0.0, 1.0)

    height, width, _ = SKYBOX.shape

    px = (u * width).astype(cp.int64) % width
    py = (v * (height - 1)).astype(cp.int64)

    image = SKYBOX[py, px].astype(cp.float32)

    if M is not None:
        captured = r <= 2.0 * M
        image[captured] = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)

    invalid = ~cp.isfinite(lightrays).all(axis=2)
    image[invalid] = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)

    return image


def init_lightrays(CAMERA, BLACKHOLE):
    """Initialise tous les rayons d'un coup sur GPU.

    Retour :
        LIGHTRAYS.shape = (height, width, 8)
        LIGHTRAYS[y, x] = [t, r, theta, phi, ut, ur, utheta, uphi]
    """
    width = CAMERA["width"]
    height = CAMERA["height"]
    M = BLACKHOLE["MASS"]

    pos = cp.array([
        CAMERA["x"] - BLACKHOLE["x"],
        CAMERA["y"] - BLACKHOLE["y"],
        CAMERA["z"] - BLACKHOLE["z"],
    ], dtype=cp.float64)

    r0 = cp.linalg.norm(pos)
    if bool(r0 <= 2.0 * M):
        raise ValueError("Camera is inside or on the black hole horizon")

    x0, y0, z0 = pos
    theta0 = cp.arccos(cp.clip(z0 / r0, -1.0, 1.0))
    phi0 = cp.arctan2(y0, x0)

    # Grille de pixels : shape (height, width)
    pixel_x = cp.arange(width, dtype=cp.float64)[None, :]
    pixel_y = cp.arange(height, dtype=cp.float64)[:, None]

    aspect_ratio = width / height
    pixel_offset_x = (pixel_x + 0.5 - width / 2) / width
    pixel_offset_y = (pixel_y + 0.5 - height / 2) / height

    angle_vertical = CAMERA["angle_vertical"]
    angle_horizontal = CAMERA["angle_horizontal"]

    forward = cp.stack([
        cp.sin(angle_vertical) * cp.cos(angle_horizontal),
        cp.sin(angle_vertical) * cp.sin(angle_horizontal),
        cp.cos(angle_vertical),
    ]).astype(cp.float64)
    forward = forward / cp.linalg.norm(forward)

    sin_av = cp.sin(angle_vertical)
    cos_av = cp.cos(angle_vertical)
    sin_ah = cp.sin(angle_horizontal)
    cos_ah = cp.cos(angle_horizontal)

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

    direction = (
        distance_from_virtual_screen * forward[None, None, :]
        + screen_x[:, :, None] * right[None, None, :]
        - screen_y[:, :, None] * up[None, None, :]
    )
    direction = direction / cp.linalg.norm(direction, axis=2, keepdims=True)

    sin_theta0 = cp.sin(theta0)
    cos_theta0 = cp.cos(theta0)
    sin_phi0 = cp.sin(phi0)
    cos_phi0 = cp.cos(phi0)

    e_r = cp.stack([
        sin_theta0 * cos_phi0,
        sin_theta0 * sin_phi0,
        cos_theta0,
    ]).astype(cp.float64)
    e_theta = cp.stack([
        cos_theta0 * cos_phi0,
        cos_theta0 * sin_phi0,
        -sin_theta0,
    ]).astype(cp.float64)
    e_phi = cp.stack([
        -sin_phi0,
        cos_phi0,
        cp.asarray(0.0),
    ]).astype(cp.float64)

    direction_r = cp.sum(direction * e_r[None, None, :], axis=2)
    direction_theta = cp.sum(direction * e_theta[None, None, :], axis=2)
    direction_phi = cp.sum(direction * e_phi[None, None, :], axis=2)

    ur = direction_r
    utheta = direction_theta / r0
    uphi = cp.where(
        cp.abs(sin_theta0) > 1e-12,
        direction_phi / (r0 * sin_theta0),
        0.0,
    )

    f = 1.0 - 2.0 * M / r0
    spatial_norm = (
        ur**2 / f
        + r0**2 * utheta**2
        + r0**2 * sin_theta0**2 * uphi**2
    )
    ut = cp.sqrt(spatial_norm / f)

    lightrays = cp.empty((height, width, 8), dtype=cp.float64)
    lightrays[:, :, 0] = 0.0
    lightrays[:, :, 1] = r0
    lightrays[:, :, 2] = theta0
    lightrays[:, :, 3] = phi0
    lightrays[:, :, 4] = ut
    lightrays[:, :, 5] = ur
    lightrays[:, :, 6] = utheta
    lightrays[:, :, 7] = uphi

    return lightrays
