import time

import cupy as cp
import numpy as np
import pygame
import imageio.v3 as iio
import os

import disk_lut
import func
from disk_cache import DiskLUTCache
from display import OpenGLImageDisplay
from lut_cache import OrbitalLUTCache


BLACKHOLE = {
    "MASS": 1.0,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
}

CAMERA = {
    "FOV": np.deg2rad(75),
    "x": 0.0,
    "y": -70.0,
    "z": 8.0,
    "angle_vertical": np.pi / 2 + np.deg2rad(7),
    "angle_horizontal": np.pi / 2,
    "width": 1280,
    "height": 720,
    "camera_virtual_screen_width": 1.0,
}
CAMERA["distance_from_virtual_screen"] = (
    CAMERA["camera_virtual_screen_width"] / (2.0 * np.tan(CAMERA["FOV"] / 2.0))
)

RAYS_NUMBER = 1000
MAX_STEPS = 250_000
#LUT_DR = 1.0
LUT_DR = 0.1
STARTUP_PRECOMPUTE_MARGIN = 5.0
IDLE_PREFETCH_MARGIN = 3.0
ENABLE_DISK_OVERLAY = False

ANGLE_SPEED = np.deg2rad(90.0)  # rad/s
MOVE_SPEED = 10.0                # unités de coordonnée / s
MOUSE_SENSITIVITY = 0.0005        # rad/pixel

OUT_DIR = "video_frames"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_disk_lut_for_current_camera(disk_cache):
    if not ENABLE_DISK_OVERLAY:
        return None

    current_r = func.camera_radius(CAMERA, BLACKHOLE)
    return disk_cache.get(current_r)


def render_frame(skybox, beta_grid, final_states, disk_crossing_lut=None):
    """Rend l'image courante depuis CAMERA, sans recalculer la géodésique."""
    t0 = time.perf_counter()
    image_gpu, _, _ = func.render_skybox_from_orbital_lut(
        CAMERA,
        BLACKHOLE,
        beta_grid,
        final_states,
        skybox,
    )
    if ENABLE_DISK_OVERLAY and disk_crossing_lut is not None:
        disk_beta_grid, disk_beta_samples, disk_r_samples = disk_crossing_lut
        overlay = disk_lut.compute_disk_overlay_from_lut(
            CAMERA,
            BLACKHOLE,
            disk_beta_grid,
            disk_beta_samples,
            disk_r_samples,
        )
        image_gpu = cp.clip(image_gpu + overlay, 0.0, 1.0)

    image_cpu = cp.asnumpy(image_gpu)
    frame_rgb = np.clip(image_cpu * 255.0, 0.0, 255.0).astype(np.uint8)
    print(f"frame rendered in {time.perf_counter() - t0:.3f}s")
    return frame_rgb


def move_camera(delta):
    CAMERA["x"] += delta[0]
    CAMERA["y"] += delta[1]
    CAMERA["z"] += delta[2]


def handle_continuous_input(dt):
    """Lit l'état clavier courant.

    Contrairement aux KEYDOWN ponctuels, pygame.key.get_pressed() permet :
        - maintenir une touche enfoncée ;
        - combiner plusieurs touches, ex: z+d, z+space, left+up.

    Retourne (orientation_changed, position_changed, should_quit).
    """
    keys = pygame.key.get_pressed()

    orientation_changed = False
    position_changed = False
    should_quit = keys[pygame.K_ESCAPE]
    toggle_disk = keys[pygame.K_m]

    # Rotation continue.
    d_angle_h = 0.0
    d_angle_v = 0.0

    if keys[pygame.K_LEFT] or keys[pygame.K_j]:
        d_angle_h -= ANGLE_SPEED * dt
    if keys[pygame.K_RIGHT] or keys[pygame.K_l]:
        d_angle_h += ANGLE_SPEED * dt
    if keys[pygame.K_UP] or keys[pygame.K_i]:
        d_angle_v -= ANGLE_SPEED * dt
    if keys[pygame.K_DOWN] or keys[pygame.K_k]:
        d_angle_v += ANGLE_SPEED * dt

    if d_angle_h != 0.0 or d_angle_v != 0.0:
        CAMERA["angle_horizontal"] += d_angle_h
        CAMERA["angle_vertical"] = np.clip(
            CAMERA["angle_vertical"] + d_angle_v,
            1e-3,
            np.pi - 1e-3,
        )
        orientation_changed = True

    # Déplacement continu AZERTY : zqsd + espace/shift.
    forward, right, up = func.camera_basis(CAMERA)
    move = np.zeros(3, dtype=np.float64)

    if keys[pygame.K_z]:
        move += forward
    if keys[pygame.K_s]:
        move -= forward
    if keys[pygame.K_d]:
        move += right
    if keys[pygame.K_q]:
        move -= right
    if keys[pygame.K_SPACE]:
        move += up
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        move -= up

    move_norm = np.linalg.norm(move)
    if move_norm > 0.0:
        # Normalisation : z+d ne va pas sqrt(2) fois plus vite que z seul.
        move_camera(MOVE_SPEED * dt * move / move_norm)
        position_changed = True

    return orientation_changed, position_changed, should_quit, toggle_disk


def capture_mouse():
    """Capture la souris dans la fenêtre pour pouvoir regarder autour."""
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()  # flush, évite un saut initial


def handle_mouse_look():
    """Tourne la caméra avec le mouvement relatif de la souris."""
    dx, dy = pygame.mouse.get_rel()

    if dx == 0 and dy == 0:
        return False

    CAMERA["angle_horizontal"] += dx * MOUSE_SENSITIVITY

    # dy positif = souris vers le bas. Le signe ci-dessous donne un feeling FPS :
    # souris vers le haut -> regarde vers le haut.
    CAMERA["angle_vertical"] = np.clip(
        CAMERA["angle_vertical"] + dy * MOUSE_SENSITIVITY,
        1e-3,
        np.pi - 1e-3,
    )

    return True


def main():
    global ENABLE_DISK_OVERLAY

    skybox = func.load_skybox("source/skybox.png")

    lut_cache = OrbitalLUTCache(
        M=BLACKHOLE["MASS"],
        rays_number=RAYS_NUMBER,
        max_steps=MAX_STEPS,
        dr=LUT_DR,
    )
    disk_cache = DiskLUTCache(
        M=BLACKHOLE["MASS"],
        rays_number=RAYS_NUMBER,
        max_steps=MAX_STEPS,
        dr=LUT_DR,
    )

    # Démarrage volontairement un peu plus long : on prend un peu d'avance.
    # Le reste de la marge +/-20 sera rempli pendant les temps morts.
    current_r = func.camera_radius(CAMERA, BLACKHOLE)
    lut_cache.precompute_margin_around(current_r, margin=STARTUP_PRECOMPUTE_MARGIN)
    beta_grid, final_states = lut_cache.get_interpolated(current_r)
    disk_crossing_lut = compute_disk_lut_for_current_camera(disk_cache)
    frame = render_frame(skybox, beta_grid, final_states, disk_crossing_lut)

    display = OpenGLImageDisplay(
        CAMERA["width"],
        CAMERA["height"],
        title="Ray-traced black hole",
    )
    display.update_image(frame)
    capture_mouse()

    running = True
    previous_toggle_disk = False
    last_time = time.perf_counter()
    while running:
        now = time.perf_counter()
        dt = min(now - last_time, 0.05)  # évite un gros saut après un calcul bloquant
        last_time = now

        image_dirty = False
        geodesic_dirty = False
        had_input = False

        for event in display.poll_events():
            if event.type == pygame.QUIT:
                running = False

        orientation_changed, position_changed, should_quit, toggle_disk = handle_continuous_input(dt)
        mouse_changed = handle_mouse_look()
        if should_quit:
            running = False
        if toggle_disk and not previous_toggle_disk:
            ENABLE_DISK_OVERLAY = not ENABLE_DISK_OVERLAY
            print(f"disk overlay: {'on' if ENABLE_DISK_OVERLAY else 'off'}")
            if ENABLE_DISK_OVERLAY and disk_crossing_lut is None:
                disk_crossing_lut = compute_disk_lut_for_current_camera(disk_cache)
            image_dirty = True
            had_input = True
        previous_toggle_disk = toggle_disk

        if orientation_changed or mouse_changed:
            image_dirty = True
            had_input = True
        if position_changed:
            geodesic_dirty = True
            image_dirty = True
            had_input = True

        if geodesic_dirty:
            current_r = func.camera_radius(CAMERA, BLACKHOLE)
            beta_grid, final_states = lut_cache.get_interpolated(current_r)
            disk_crossing_lut = compute_disk_lut_for_current_camera(disk_cache)

        if image_dirty:
            frame = render_frame(skybox, beta_grid, final_states, disk_crossing_lut)
            display.update_image(frame)

        # Si rien n'a changé, on ne recalcule pas l'image : on redessine juste
        # la texture déjà en VRAM + le FPS.
        display.draw()
        display.tick(240)

        # Temps mort : prépare une seule LUT voisine. Si ça prend ~1s, ce n'est
        # pas grave : ça arrive seulement quand l'utilisateur ne touche à rien.
        if not had_input and not image_dirty:
            current_r = func.camera_radius(CAMERA, BLACKHOLE)
            lut_cache.prefetch_one_missing_with_margin(
                current_r,
                margin=IDLE_PREFETCH_MARGIN,
            )

    display.close()


if __name__ == "__main__":
    main()
