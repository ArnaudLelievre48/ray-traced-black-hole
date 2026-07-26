import time

import cupy as cp
import numpy as np
import pygame

import GR
import func
from display import OpenGLImageDisplay


BLACKHOLE = {
    "MASS": 1.0,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
}

CAMERA = {
    "FOV": np.deg2rad(75),
    "x": 0.0,
    "y": -50.0,
    "z": 0.0,
    "angle_vertical": np.pi / 2,
    "angle_horizontal": np.pi / 2,
    "width": 1280,
    "height": 720,
    "camera_virtual_screen_width": 1.0,
}

CAMERA["distance_from_virtual_screen"] = (
    CAMERA["camera_virtual_screen_width"] / (2.0 * np.tan(CAMERA["FOV"] / 2.0))
)

RAYS_NUMBER = 500
MAX_STEPS = 5_000

ANGLE_STEP = np.deg2rad(2.0)
MOVE_STEP = 2.0


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


def compute_lut():
    """Calcule la LUT GR pour la distance caméra-trou noir actuelle."""
    r = camera_radius(CAMERA, BLACKHOLE)
    beta_grid = cp.linspace(0.0, cp.pi, RAYS_NUMBER, dtype=cp.float64)

    t0 = time.perf_counter()
    print(f"computing deviation LUT at r={r:.3f}...")
    final_states = GR.orbital_geodesic_fast(
        r,
        BLACKHOLE["MASS"],
        RAYS_NUMBER=RAYS_NUMBER,
        MAX_STEPS=MAX_STEPS,
    )
    cp.cuda.Stream.null.synchronize()
    print(f"LUT done in {time.perf_counter() - t0:.3f}s")

    return beta_grid, final_states


def render_frame(skybox, beta_grid, final_states):
    """Rend l'image courante depuis CAMERA, sans recalculer la géodésique."""
    t0 = time.perf_counter()
    image_gpu, _, _ = func.render_skybox_from_orbital_lut(
        CAMERA,
        BLACKHOLE,
        beta_grid,
        final_states,
        skybox,
    )
    image_cpu = cp.asnumpy(image_gpu)
    frame_rgb = np.clip(image_cpu * 255.0, 0.0, 255.0).astype(np.uint8)
    print(f"frame rendered in {time.perf_counter() - t0:.3f}s")
    return frame_rgb


def move_camera(delta):
    CAMERA["x"] += delta[0]
    CAMERA["y"] += delta[1]
    CAMERA["z"] += delta[2]


def handle_keydown(event):
    """Retourne (orientation_changed, position_changed, should_quit)."""
    orientation_changed = False
    position_changed = False
    should_quit = False

    forward, right, up = camera_basis(CAMERA)

    if event.key == pygame.K_ESCAPE:
        should_quit = True

    # Rotation clavier pour l'instant.
    elif event.key in (pygame.K_LEFT, pygame.K_j):
        CAMERA["angle_horizontal"] -= ANGLE_STEP
        orientation_changed = True
    elif event.key in (pygame.K_RIGHT, pygame.K_l):
        CAMERA["angle_horizontal"] += ANGLE_STEP
        orientation_changed = True
    elif event.key in (pygame.K_UP, pygame.K_i):
        CAMERA["angle_vertical"] = np.clip(
            CAMERA["angle_vertical"] - ANGLE_STEP,
            1e-3,
            np.pi - 1e-3,
        )
        orientation_changed = True
    elif event.key in (pygame.K_DOWN, pygame.K_k):
        CAMERA["angle_vertical"] = np.clip(
            CAMERA["angle_vertical"] + ANGLE_STEP,
            1e-3,
            np.pi - 1e-3,
        )
        orientation_changed = True

    # Déplacement AZERTY : zqsd.
    # Ici on recalcule la LUT, comme demandé, car la position change.
    elif event.key == pygame.K_z:
        move_camera(MOVE_STEP * forward)
        position_changed = True
    elif event.key == pygame.K_s:
        move_camera(-MOVE_STEP * forward)
        position_changed = True
    elif event.key == pygame.K_d:
        move_camera(MOVE_STEP * right)
        position_changed = True
    elif event.key == pygame.K_q:
        move_camera(-MOVE_STEP * right)
        position_changed = True
    elif event.key == pygame.K_SPACE:
        move_camera(MOVE_STEP * up)
        position_changed = True
    elif event.key == pygame.K_LSHIFT:
        move_camera(-MOVE_STEP * up)
        position_changed = True

    return orientation_changed, position_changed, should_quit


def main():
    skybox = func.load_skybox("source/skybox.png")
    beta_grid, final_states = compute_lut()
    frame = render_frame(skybox, beta_grid, final_states)

    display = OpenGLImageDisplay(
        CAMERA["width"],
        CAMERA["height"],
        title="Ray-traced black hole",
    )
    display.update_image(frame)

    running = True
    while running:
        image_dirty = False
        geodesic_dirty = False

        for event in display.poll_events():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                orientation_changed, position_changed, should_quit = handle_keydown(event)

                if should_quit:
                    running = False
                if orientation_changed:
                    image_dirty = True
                if position_changed:
                    geodesic_dirty = True
                    image_dirty = True

        if geodesic_dirty:
            beta_grid, final_states = compute_lut()

        if image_dirty:
            frame = render_frame(skybox, beta_grid, final_states)
            display.update_image(frame)

        # Si rien n'a changé, on ne recalcule pas l'image : on redessine juste
        # la texture déjà en VRAM + le FPS.
        display.draw()
        display.tick(240)

    display.close()


if __name__ == "__main__":
    main()
