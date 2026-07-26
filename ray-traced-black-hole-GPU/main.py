import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

# local import
import func
import GR


BLACKHOLE = {
    "MASS": 1.0,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
}

CAMERA = {
    "FOV": np.deg2rad(75),
    "x": 0.0,
    "y": -75.0,
    "z": 0.0,
    "angle_vertical": np.pi / 2,
    "angle_horizontal": np.pi / 2,
    "width": 1000,
    "height": 750,
    "camera_virtual_screen_width": 1.0,
}

CAMERA["distance_from_virtual_screen"] = (
    CAMERA["camera_virtual_screen_width"] / (2.0 * np.tan(CAMERA["FOV"] / 2.0))
)

SKYBOX = func.load_skybox("source/skybox.png")

camera_position = cp.array([CAMERA["x"], CAMERA["y"], CAMERA["z"]], dtype=cp.float64)
blackhole_position = cp.array([BLACKHOLE["x"], BLACKHOLE["y"], BLACKHOLE["z"]], dtype=cp.float64)
COMPUTATION_DISK_RADIUS = 1.5 * cp.linalg.norm(camera_position - blackhole_position)

MAX_STEPS = 10_000

# Pas adaptatif par rayon.
# H_MAX : pas grossier loin du trou noir / hors zone sensible.
# H_MIN : pas prudent près du trou noir ou près de l'axe central uphi≈0.
H_MAX = 0.25
H_MIN = 0.005

# Rayon de transition : entre r=2M et r=ADAPT_RADIUS_FACTOR*M,
# le pas passe progressivement de H_MIN à H_MAX.
ADAPT_RADIUS_FACTOR = 25.0

# Si |u^phi| est plus petit que ce seuil, on diminue le pas.
# La colonne centrale a typiquement u^phi≈0, donc c'est elle qu'on protège.
UPHI_SAFE = 5e-4

# Zone verticale sensible autour de phi=pi/2.
# On garde un petit pas dans cette bande, même si u^phi n'est pas exactement nul.
PHI_VERTICAL_CENTER = np.pi / 2
PHI_VERTICAL_SAFE = np.deg2rad(2.0)

# Rayons qui foncent vers le trou noir : on les détecte avec un impact parameter
# approximatif b=L/E. Si ur<0 et b est petit/proche de b_crit, on baisse le pas.
BH_FOCUS_B_FACTOR = 8.0

# LIGHTRAYS[y, x] = [t, r, theta, phi, ut, ur, utheta, uphi]
LIGHTRAYS = func.init_lightrays(CAMERA, BLACKHOLE)
SAFE_LIGHTRAYS = LIGHTRAYS.copy()

active = cp.ones(LIGHTRAYS.shape[:2], dtype=cp.bool_)

step = 1
percent = 0

while bool(active.any().get()) and step < MAX_STEPS:
    # On ne veut pas intégrer les rayons déjà terminés, mais pour rester GPU-friendly
    # on garde un tableau rectangulaire complet et on remplace les inactifs par un
    # état sûr avant RK4.
    step_input = cp.where(active[:, :, None], LIGHTRAYS, SAFE_LIGHTRAYS)

    r_input = step_input[:, :, 1]
    theta_input = step_input[:, :, 2]
    phi_input = step_input[:, :, 3]
    ut_input = step_input[:, :, 4]
    ur_input = step_input[:, :, 5]
    utheta_input = step_input[:, :, 6]
    uphi_input = step_input[:, :, 7]

    # 1) Proximité trou noir : petit pas près de r=2M, grand pas loin.
    radius_factor = cp.clip(
        (r_input - 2.0 * BLACKHOLE["MASS"])
        / ((ADAPT_RADIUS_FACTOR - 2.0) * BLACKHOLE["MASS"]),
        0.0,
        1.0,
    )
    # Puissance 2 : on reste prudent près du trou noir, puis le pas augmente
    # franchement quand on s'en éloigne.
    h_radius = H_MIN + (H_MAX - H_MIN) * radius_factor**2

    # 2) Axe central : si u^phi≈0, on réduit le pas pour éviter les grosses
    # divergences numériques observées autour de phi=pi/2.
    # L'axe uphi≈0 est surtout dangereux près du trou noir. Loin du trou noir,
    # les rayons vont quasi tout droit, donc radius_factor relâche cette contrainte.
    uphi_factor = cp.clip(cp.abs(uphi_input) / UPHI_SAFE + radius_factor, 0.0, 1.0)
    h_axis = H_MIN + (H_MAX - H_MIN) * uphi_factor

    # 3) Bande verticale phi≈pi/2 : plus robuste que seulement u^phi≈0.
    # Distance angulaire avec wrap 2pi.
    dphi_vertical = cp.abs(
        (phi_input - PHI_VERTICAL_CENTER + cp.pi) % (2.0 * cp.pi) - cp.pi
    )
    vertical_factor = cp.clip(dphi_vertical / PHI_VERTICAL_SAFE + 0.5 * radius_factor, 0.0, 1.0)
    h_vertical = H_MIN + (H_MAX - H_MIN) * vertical_factor

    # 4) Rayons entrants vers le trou noir : impact parameter b=L/E.
    # Pour Schwarzschild, E=f*u^t et L≈r²*sqrt((u^theta)^2 + sin²(theta)(u^phi)^2).
    sin_theta_input = cp.sin(theta_input)
    f_input = 1.0 - 2.0 * BLACKHOLE["MASS"] / r_input
    E = f_input * ut_input
    L = r_input**2 * cp.sqrt(utheta_input**2 + sin_theta_input**2 * uphi_input**2)
    b = cp.where(cp.abs(E) > 1e-12, L / cp.abs(E), cp.inf)

    b_crit = 3.0 * cp.sqrt(cp.asarray(3.0)) * BLACKHOLE["MASS"]
    b_focus_max = BH_FOCUS_B_FACTOR * BLACKHOLE["MASS"]
    b_factor = cp.clip((b - b_crit) / (b_focus_max - b_crit), 0.0, 1.0)

    # Si le rayon ne rentre pas, on ne force pas ce critère.
    # S'il rentre et a petit b, h_blackhole≈H_MIN.
    inward = ur_input < 0.0
    blackhole_factor = cp.where(inward, cp.clip(b_factor + 0.35 * radius_factor, 0.0, 1.0), 1.0)
    h_blackhole = H_MIN + (H_MAX - H_MIN) * blackhole_factor

    # On prend le pas le plus prudent des deux critères.
    h_matrix = cp.minimum(cp.minimum(h_radius, h_axis), cp.minimum(h_vertical, h_blackhole))

    next_lightrays = GR.rk4_step(step_input, h_matrix, BLACKHOLE["MASS"])
    LIGHTRAYS = cp.where(active[:, :, None], next_lightrays, LIGHTRAYS)

    r = LIGHTRAYS[:, :, 1]
    ur = LIGHTRAYS[:, :, 5]

    invalid = ~cp.isfinite(LIGHTRAYS).all(axis=2)
    captured = r <= 2.0 * BLACKHOLE["MASS"]
    escaped = (r >= COMPUTATION_DISK_RADIUS) & (ur > 0.0)

    active = active & ~(invalid | captured | escaped)
    step += 1
    if int(100 * step / MAX_STEPS) - percent > 0:
        percent = int(100 * step/ MAX_STEPS)
        remaining = int(active.sum().get())
        print(f"MAX {percent} % | step={step} active={remaining}", flush=True)

image = func.skybox_color(LIGHTRAYS, BLACKHOLE["MASS"])

# Les rayons non terminés après MAX_STEPS sont considérés non résolus -> noir.
image[active] = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)

image_cpu = cp.asnumpy(image)
plt.imshow(image_cpu)
plt.axis("off")
plt.show()
