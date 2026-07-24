import numpy as np
import matplotlib.pyplot as plt


def load_skybox(path):
    """Charge une skybox equirectangulaire en float32 RGB dans [0, 1]."""
    skybox = plt.imread(path)
    skybox = skybox.astype(np.float32)

    if skybox.max() > 1.0:
        skybox /= 255.0

    if skybox.ndim == 3 and skybox.shape[2] == 4:
        skybox = skybox[:, :, :3]

    return skybox


def skybox_color(direction, skybox):
    """Renvoie la couleur de la skybox dans la direction 3D donnée.

    Échantillonnage bilinéaire + wrap horizontal pour éviter les coutures
    trop visibles dans l'image.
    """
    direction = np.array(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    dx, dy, dz = direction / norm

    phi = np.arctan2(dy, dx)                  # [-pi, pi]
    theta = np.arccos(np.clip(dz, -1.0, 1.0)) # [0, pi]

    # u wrappe horizontalement, v est clampé verticalement.
    u = ((phi + np.pi) / (2 * np.pi)) % 1.0
    v = np.clip(theta / np.pi, 0.0, 1.0)

    height, width, _ = skybox.shape

    x = u * width
    y = v * (height - 1)

    x0 = int(np.floor(x)) % width
    x1 = (x0 + 1) % width
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, height - 1)

    tx = x - np.floor(x)
    ty = y - y0

    c00 = skybox[y0, x0]
    c10 = skybox[y0, x1]
    c01 = skybox[y1, x0]
    c11 = skybox[y1, x1]

    c0 = (1.0 - tx) * c00 + tx * c10
    c1 = (1.0 - tx) * c01 + tx * c11
    return ((1.0 - ty) * c0 + ty * c1).astype(np.float32)


def skybox_color_from_position(origin, direction, skybox, sphere_radius):
    """
    Version skybox sur sphère finie centrée en (0,0,0).

    On intersecte le rayon origin + t*direction avec la sphère de rayon
    sphere_radius, puis on utilise la direction centre->point_intersection
    pour lire la skybox. Contrairement à skybox_color(direction), ça prend
    donc en compte la position de la caméra/rayon.
    """
    origin = np.array(origin, dtype=np.float32)
    direction = np.array(direction, dtype=np.float32)

    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    direction = direction / norm

    # Résout |origin + t direction|² = sphere_radius².
    b = 2.0 * np.dot(origin, direction)
    c = np.dot(origin, origin) - sphere_radius**2
    discriminant = b*b - 4.0*c

    if discriminant < 0:
        # Fallback : pas d'intersection numérique, comportement skybox infinie.
        return skybox_color(direction, skybox)

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    # On veut l'intersection devant le rayon.
    if t2 >= 0:
        t = t2
    elif t1 >= 0:
        t = t1
    else:
        return skybox_color(direction, skybox)

    hit_point = origin + t * direction
    return skybox_color(hit_point, skybox)

def pos_to_map(x, y, z, MAP):
    """Convertit une position monde (x,y,z), centrée en 0, vers MAP[iz, iy, ix]."""
    shape_z, shape_y, shape_x, _ = MAP.shape
    return (int(z + shape_z/2), int(y + shape_y/2), int(x + shape_x/2))


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def distance(pos1, pos2):
    x1,y1,z1 = pos1
    x2,y2,z2 = pos2
    return np.sqrt( (x1-x2)**2 +  (y1-y2)**2 + (z1-z2)**2 )


def cartesian_to_spherical(x, y=None, z=None):
    """Convertit (x,y,z) cartésien vers (r, theta, phi)."""
    if y is None and z is None:
        x, y, z = x

    r = np.sqrt(x*x + y*y + z*z)
    if r == 0:
        return 0.0, 0.0, 0.0

    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta=None, phi=None):
    """Convertit (r,theta,phi) sphérique vers (x,y,z) cartésien."""
    if theta is None and phi is None:
        r, theta, phi = r

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z], dtype=np.float64)


def spherical_basis(theta, phi):
    """Base sphérique orthonormée locale (e_r, e_theta, e_phi) en cartésien."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    e_r = np.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta,
    ], dtype=np.float64)

    e_theta = np.array([
        cos_theta * cos_phi,
        cos_theta * sin_phi,
        -sin_theta,
    ], dtype=np.float64)

    e_phi = np.array([
        -sin_phi,
        cos_phi,
        0.0,
    ], dtype=np.float64)

    return e_r, e_theta, e_phi


def cartesian_to_spherical_gr(x, y=None, z=None):
    """
    Coordonnées sphériques pour l'intégration GR, avec axe polaire = axe x.

    Avec l'axe z classique, les rayons de la colonne centrale passent près de
    theta=0/pi : les coordonnées sphériques deviennent singulières et le terme
    cot(theta) de l'équation géodésique explose. Schwarzschild est sphériquement
    symétrique, donc on peut choisir un autre axe polaire sans changer la physique.
    """
    if y is None and z is None:
        x, y, z = x

    r = np.sqrt(x*x + y*y + z*z)
    if r == 0:
        return 0.0, 0.0, 0.0

    theta = np.arccos(np.clip(x / r, -1.0, 1.0))
    phi = np.arctan2(z, y)
    return r, theta, phi


def spherical_to_cartesian_gr(r, theta=None, phi=None):
    """Inverse de cartesian_to_spherical_gr : axe polaire = axe x."""
    if theta is None and phi is None:
        r, theta, phi = r

    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(theta) * np.sin(phi)
    return np.array([x, y, z], dtype=np.float64)


def spherical_basis_gr(theta, phi):
    """Base sphérique orthonormée GR avec axe polaire = axe x."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    e_r = np.array([
        cos_theta,
        sin_theta * cos_phi,
        sin_theta * sin_phi,
    ], dtype=np.float64)

    e_theta = np.array([
        -sin_theta,
        cos_theta * cos_phi,
        cos_theta * sin_phi,
    ], dtype=np.float64)

    e_phi = np.array([
        0.0,
        -sin_phi,
        cos_phi,
    ], dtype=np.float64)

    return e_r, e_theta, e_phi


def spatial_direction_from_gr_state(state, basis=None):
    """
    Convertit la vitesse spatiale coordonnée Schwarzschild
    (ur, utheta, uphi) en direction cartésienne.

    state = [t, r, theta, phi, ut, ur, utheta, uphi]
    """
    r = state[1]
    theta = state[2]
    phi = state[3]
    ur = state[5]
    utheta = state[6]
    uphi = state[7]

    e_r, e_theta, e_phi = spherical_basis(theta, phi)

    # Les composantes physiques associées sont :
    # radial: ur, polaire: r*utheta, azimutale: r*sin(theta)*uphi.
    direction = (
        ur * e_r
        + (r * utheta) * e_theta
        + (r * np.sin(theta) * uphi) * e_phi
    )

    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    direction = direction / norm

    if basis is not None:
        direction = basis @ direction
        norm = np.linalg.norm(direction)
        if norm == 0 or not np.isfinite(norm):
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        direction = direction / norm

    return direction


def plot_map_scatter(MAP, output_path="map_scatter_debug.png"):
    """Plot 3D des voxels non noirs de MAP pour vérifier leur répartition spatiale."""
    occupied = np.linalg.norm(MAP[:, :, :, :3], axis=3) > 0.001
    iz, iy, ix = np.where(occupied)

    if len(ix) == 0:
        print("No occupied voxels to plot")
        return

    shape_z, shape_y, shape_x, _ = MAP.shape
    x = ix - shape_x / 2
    y = iy - shape_y / 2
    z = iz - shape_z / 2
    colors = MAP[iz, iy, ix, 0] # rouge

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=colors, s=8, alpha=0.8)

    limit = max(shape_x, shape_y, shape_z) / 2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"MAP occupied voxels: {len(ix)}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved {output_path} with {len(ix)} occupied voxels")



